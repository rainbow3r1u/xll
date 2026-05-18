#!/usr/bin/env python3
"""
提取 Kronos-base 内部 Context Vector (embedding)
每256天BTC窗口 → 832维隐状态 → PCA压缩 → 20维保存

用法:
  python extract_embeddings.py            # 全新提取 (首次)
  python extract_embeddings.py --update   # 增量更新 (每周一次, GPU推理)

周常: 每周在GPU服务器跑一次 --update, 传回替换 kronos_embeddings.json
"""
import os, sys, json, argparse, numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kronos_model.kronos import Kronos, KronosTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--update', action='store_true', help='增量更新模式')
parser.add_argument('--refit-pca', action='store_true', help='强制重算PCA')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")

OUTPUT = os.path.join(os.path.dirname(__file__), 'kronos_embeddings.json')
LOCAL_MODEL = os.path.join(os.path.dirname(__file__), 'kronos_pretrained/Kronos-base')
LOCAL_TOKENIZER = os.path.join(os.path.dirname(__file__), 'kronos_pretrained/Kronos-Tokenizer-base')

# 加载模型
print("加载 Kronos-base...")
model = Kronos.from_pretrained(LOCAL_MODEL).to(DEVICE)
tokenizer = KronosTokenizer.from_pretrained(LOCAL_TOKENIZER).to(DEVICE)
model.eval()

# 加载BTC数据
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
mean = data.mean(axis=0, keepdims=True)
std = data.std(axis=0, keepdims=True) + 1e-8
data_norm = (data - mean) / std
timestamps = df.index

CONTEXT_DAYS = 256
MIN_START = 200

# 增量模式: 只处理新时间戳
existing_ts = set()
old_pca = None
old_scaler = None
if args.update and os.path.exists(OUTPUT):
    print(f"[增量] 加载现有 embeddings: {OUTPUT}")
    with open(OUTPUT) as f:
        old = json.load(f)
    existing_ts = set(old['embeddings'].keys())
    old_pca = (old.get('pca_components'), old.get('pca_mean'))
    old_scaler = (old.get('scaler_mean'), old.get('scaler_scale'))
    print(f"  已有 {len(existing_ts)} 个时间点")

# 找出需要提取的窗口
to_extract = []
for i in range(MIN_START, len(data)):
    ts = int(timestamps[i].timestamp())
    ts_str = str(ts)
    if ts_str not in existing_ts:
        to_extract.append(i)

if not to_extract:
    print("所有时间点已存在, 无需更新")
    sys.exit(0)

print(f"需提取: {len(to_extract)} 个新窗口")

# 提取新embeddings
new_embeddings = {}
pbar = tqdm(to_extract, desc="提取 embedding")
for i in pbar:
    ts = int(timestamps[i].timestamp())
    ctx_data = data_norm[i - CONTEXT_DAYS:i]

    x = torch.FloatTensor(ctx_data).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        tokens = tokenizer.encode(x, half=True)
        s1_ids, s2_ids = tokens[0], tokens[1]

        # 时间戳: [minute, hour, weekday, day, month] 5列
        window_dates = timestamps[i - CONTEXT_DAYS:i]
        x_stamp = torch.zeros(1, CONTEXT_DAYS, 5, dtype=torch.long, device=DEVICE)
        x_stamp[:, :, 0] = torch.LongTensor([d.minute for d in window_dates])
        x_stamp[:, :, 1] = torch.LongTensor([d.hour for d in window_dates])
        x_stamp[:, :, 2] = torch.LongTensor([d.weekday() for d in window_dates])
        x_stamp[:, :, 3] = torch.LongTensor([d.day for d in window_dates])
        x_stamp[:, :, 4] = torch.LongTensor([d.month for d in window_dates])

        s1_logits, context = model.decode_s1(s1_ids, s2_ids, x_stamp)
        vec = context[:, -1, :].squeeze(0).cpu().numpy()

    new_embeddings[str(ts)] = vec.tolist()

print(f"新提取: {len(new_embeddings)} 个")

# 合并
if args.update and existing_ts:
    all_embeddings = {k: v for k, v in old['embeddings'].items()}
else:
    all_embeddings = {}
all_embeddings.update(new_embeddings)

all_ts = sorted(all_embeddings.keys())
all_vecs = np.array([all_embeddings[ts] for ts in all_ts])
print(f"全量: {all_vecs.shape}")

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if args.update and old_pca is not None and not args.refit_pca:
    # 用旧PCA投影新点
    print("[增量] 使用已有PCA变换")
    scaler = StandardScaler()
    scaler.mean_ = np.array(old_scaler[0])
    scaler.scale_ = np.array(old_scaler[1])
    vecs_scaled = scaler.transform(all_vecs)
    pca = PCA(n_components=len(old_pca[0]))
    pca.components_ = np.array(old_pca[0])
    pca.mean_ = np.array(old_pca[1])
    vecs_pca20 = pca.transform(vecs_scaled)
    variance = float(np.var(vecs_pca20).sum() / np.var(vecs_scaled).sum() * 100)
    # rough estimate
    variance = old.get('variance_retained', 77.9)
else:
    # 重新拟合PCA
    print("[重算] PCA...")
    scaler = StandardScaler()
    vecs_scaled = scaler.fit_transform(all_vecs)
    pca = PCA(n_components=20)
    vecs_pca20 = pca.fit_transform(vecs_scaled)
    variance = float(pca.explained_variance_ratio_.sum())

print(f"PCA 20维: 方差保留 {variance*100:.1f}%")

# 保存
output = {
    'embeddings': {ts: vecs_pca20[i].tolist() for i, ts in enumerate(all_ts)},
    'pca_components': pca.components_.tolist(),
    'pca_mean': pca.mean_.tolist(),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'variance_retained': variance,
    'n_dim': 20,
    'total_timestamps': len(all_ts),
    'last_updated': str(pd.Timestamp.now()),
}

with open(OUTPUT, 'w') as f:
    json.dump(output, f)

print(f"\n保存: {OUTPUT} ({os.path.getsize(OUTPUT)/1024:.0f}KB)")
print(f"时间点: {len(all_ts)} (新增 {len(new_embeddings)})")
