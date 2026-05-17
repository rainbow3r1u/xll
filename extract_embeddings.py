#!/usr/bin/env python3
"""
提取 Kronos-base 内部 Context Vector (embedding)
每256天BTC窗口 → 832维隐状态 → 压缩保存
GPU推理, 不训练, 不会过拟合
"""
import os, sys, json, numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kronos_model.kronos import Kronos, KronosTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")

# ─── 1. 加载模型 ───
LOCAL_MODEL = os.path.join(os.path.dirname(__file__), 'kronos_pretrained/Kronos-base')
LOCAL_TOKENIZER = os.path.join(os.path.dirname(__file__), 'kronos_pretrained/Kronos-Tokenizer-base')

print("加载 Kronos-base...")
model = Kronos.from_pretrained(LOCAL_MODEL).to(DEVICE)
tokenizer = KronosTokenizer.from_pretrained(LOCAL_TOKENIZER).to(DEVICE)
model.eval()

# ─── 2. 加载BTC数据 ───
with open(os.path.join(os.path.dirname(__file__), 'btc_klines.json')) as f:
    kls = json.load(f)

df = pd.DataFrame([{
    'datetime': pd.to_datetime(k['t'], unit='ms', utc=True),
    'open': k['o'], 'high': k['h'], 'low': k['l'],
    'close': k['c'], 'volume': k['v'],
    'quote_vol': k.get('q', k['v'] * k['c'])
} for k in kls])
df = df.set_index('datetime').sort_index()

cols = ['open', 'high', 'low', 'close', 'volume', 'quote_vol']
data = df[cols].values.astype(np.float32)

# Z-score归一化
mean = data.mean(axis=0, keepdims=True)
std = data.std(axis=0, keepdims=True) + 1e-8
data_norm = (data - mean) / std

CONTEXT_DAYS = 256
MIN_START = 200  # 至少200天历史才开始提取
MAX_WINDOWS = 1500  # 最多处理1500个窗口

print(f"BTC数据: {len(data)}天, 提取窗口: {MIN_START}→{min(len(data), MIN_START+MAX_WINDOWS)}")

# ─── 3. 提取 Context Vector ───
embeddings = {}  # {timestamp_int: [embedding_list]}
timestamps = df.index

pbar = tqdm(range(MIN_START, min(len(data), MIN_START + MAX_WINDOWS)),
            desc="提取 embedding")
for i in pbar:
    if i < CONTEXT_DAYS:
        continue

    ts = int(timestamps[i].timestamp())
    ctx_data = data_norm[i - CONTEXT_DAYS:i]  # (256, 6)

    x = torch.FloatTensor(ctx_data).unsqueeze(0).to(DEVICE)  # (1, 256, 6)

    with torch.no_grad():
        tokens = tokenizer.encode(x, half=True)
        s1_ids, s2_ids = tokens[0], tokens[1]

        # 构建时间戳
        x_stamp = torch.zeros(1, CONTEXT_DAYS, 3, device=DEVICE)
        x_stamp[:, :, 0] = torch.arange(CONTEXT_DAYS)

        s1_logits, context = model.decode_s1(s1_ids, s2_ids, x_stamp)

        # context: (1, 832) — 取最后一个位置
        vec = context[:, -1, :].squeeze(0).cpu().numpy()  # (832,)

        # 再用 decode_s2 取 post 部分的信息
        # 取 s1 最后一个 token 作为输入
        last_s1 = s1_ids[:, -1:]  # (1, 1)
        s2_logits = model.decode_s2(context[:, -1:, :], last_s1)
        # s2_logits[-1] 是预测的s2分布

    embeddings[str(ts)] = vec.tolist()

print(f"\n提取完成: {len(embeddings)} 个时间点, 每点 {len(vec)} 维")

# ─── 4. PCA 压缩到 20 维 ───
# 用 sklearn PCA, 保留95%方差 → 自动选维度
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

all_ts = sorted(embeddings.keys())
all_vecs = np.array([embeddings[ts] for ts in all_ts])

print(f"全量: {all_vecs.shape}")

# 标准化
scaler = StandardScaler()
vecs_scaled = scaler.fit_transform(all_vecs)

# PCA 保留 95% 方差
pca_full = PCA(n_components=0.95, svd_solver='full')
vecs_pca_full = pca_full.fit_transform(vecs_scaled)
print(f"PCA 95%方差: {pca_full.n_components_} 维")

# 同时输出固定20维的版本
pca20 = PCA(n_components=20)
vecs_pca20 = pca20.fit_transform(vecs_scaled)
print(f"PCA 20维: 方差保留 {pca20.explained_variance_ratio_.sum()*100:.1f}%")

# ─── 5. 保存 ───
output = {
    'embeddings': {ts: vecs_pca20[i].tolist() for i, ts in enumerate(all_ts)},
    'pca_components': pca20.components_.tolist(),
    'pca_mean': pca20.mean_.tolist(),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'variance_retained': float(pca20.explained_variance_ratio_.sum()),
    'n_dim': 20,
    'total_timestamps': len(all_ts),
}

outfile = os.path.join(os.path.dirname(__file__), 'kronos_embeddings.json')
with open(outfile, 'w') as f:
    json.dump(output, f)

print(f"\n保存: {outfile} ({os.path.getsize(outfile)/1024:.0f}KB)")
print(f"服务器使用:")
print(f"  1. 复制到 /home/myuser/websocket_new/data/kronos_embeddings.json")
print(f"  2. daily_predictor.py 加载: 20维替代原来3维kronos特征")
print(f"  3. 新特征8维: 前5个PCA分量 + 3个统计值(dir/vol/long)")
