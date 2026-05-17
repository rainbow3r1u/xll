#!/usr/bin/env python3
"""
Kronos BTC 微调 — AutoDL GPU 训练脚本
对 Kronos-small 做 BTC 定制化微调，提升预测准确率

原理:
  Kronos 是时序Transformer，把K线压成离散token，然后自回归预测未来token
  微调 = 用BTC历史数据做teacher forcing，让模型更懂BTC的价格模式

用法 (AutoDL):
  pip install torch pandas numpy requests huggingface_hub tqdm
  python train.py --epochs 20 --context_days 256 --pred_days 7
"""
import os, sys, json, math, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests

# ─── 参数 ───
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--context_days', type=int, default=256, help='输入上下文天数')
parser.add_argument('--pred_days', type=int, default=7, help='要预测的未来天数')
parser.add_argument('--output_dir', type=str, default='./kronos-btc-finetuned')
parser.add_argument('--data_path', type=str, default='', help='本地BTC CSV (空则从币安拉)')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 1. 数据准备
# ═══════════════════════════════════════════════════════════════
# 加载BTC数据: 优先本地JSON → 本地CSV → 币安API
local_json = os.path.join(os.path.dirname(__file__), 'btc_klines.json')
if os.path.exists(local_json):
    print(f"从本地JSON加载: {local_json}")
    with open(local_json) as f:
        kls = json.load(f)
    df = pd.DataFrame([{
        'datetime': pd.to_datetime(k['t'], unit='ms', utc=True),
        'open': k['o'], 'high': k['h'], 'low': k['l'],
        'close': k['c'], 'volume': k['v']
    } for k in kls])
    df = df.set_index('datetime').sort_index()
elif args.data_path and os.path.exists(args.data_path):
    print(f"从本地CSV加载: {args.data_path}")
    df = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
else:
    print("从币安拉取BTC日线...")
    r = requests.get('https://fapi.binance.com/fapi/v1/klines',
        params={'symbol': 'BTCUSDT', 'interval': '1d', 'limit': 1500}, timeout=30)
    kls = r.json()
    df = pd.DataFrame([{
        'datetime': pd.to_datetime(int(k[0]), unit='ms', utc=True),
        'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]),
        'close': float(k[4]), 'volume': float(k[5])
    } for k in kls])
    df = df.set_index('datetime').sort_index()

print(f"数据: {len(df)} 天, {df.index[0].date()} → {df.index[-1].date()}")

# Kronos-Tokenizer-base 需要6维: OHLCV + quote_volume
# btc_klines.json 里有 q 字段 = quote volume
btc_kls = []
with open(os.path.join(os.path.dirname(__file__), 'btc_klines.json')) as f:
    btc_kls = json.load(f)
df = pd.DataFrame([{
    'datetime': pd.to_datetime(k['t'], unit='ms', utc=True),
    'open': k['o'], 'high': k['h'], 'low': k['l'],
    'close': k['c'], 'volume': k['v'], 'quote_vol': k.get('q', k['v'] * k['c'])
} for k in btc_kls])
df = df.set_index('datetime').sort_index()

# 6维: OHLCV + quote_vol
cols = ['open', 'high', 'low', 'close', 'volume', 'quote_vol']
data = df[cols].values.astype(np.float32)

# 归一化 (按序列做z-score)
mean = data.mean(axis=0, keepdims=True)
std = data.std(axis=0, keepdims=True) + 1e-8
data_norm = (data - mean) / std

# 构建滑动窗口
class BTCWindowDataset(Dataset):
    def __init__(self, data, context_days, pred_days):
        self.data = data
        self.context = context_days
        self.pred = pred_days
        self.n = len(data) - context_days - pred_days

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        ctx = self.data[idx:idx + self.context]          # (context, 5)
        tgt = self.data[idx + self.context:
                        idx + self.context + self.pred]   # (pred, 5)
        return torch.FloatTensor(ctx), torch.FloatTensor(tgt)

# 分割
total = len(data)
n_val = int(total * 0.15)
train_data = data_norm[:-n_val]
val_data = data_norm[-n_val - args.context_days - args.pred_days:]

train_ds = BTCWindowDataset(train_data, args.context_days, args.pred_days)
val_ds = BTCWindowDataset(val_data, args.context_days, args.pred_days)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True)
print(f"训练: {len(train_ds)} 窗口  验证: {len(val_ds)} 窗口")

# ═══════════════════════════════════════════════════════════════
# 2. 加载 Kronos (优先本地预下载, 兼容HuggingFace)
# ═══════════════════════════════════════════════════════════════
print("\n加载 Kronos-base (最大版本, 405MB)...")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kronos_model.kronos import Kronos, KronosTokenizer

# 优先本地, 否则HuggingFace下载
LOCAL_MODEL = os.path.join(os.path.dirname(__file__), 'kronos_pretrained/Kronos-base')
LOCAL_TOKENIZER = os.path.join(os.path.dirname(__file__), 'kronos_pretrained/Kronos-Tokenizer-base')

if os.path.exists(LOCAL_MODEL):
    print(f"  加载本地模型: {LOCAL_MODEL}")
    model = Kronos.from_pretrained(LOCAL_MODEL)
else:
    print("  从HuggingFace加载 Kronos-base...")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

if os.path.exists(LOCAL_TOKENIZER):
    print(f"  加载本地Tokenizer: {LOCAL_TOKENIZER}")
    tokenizer = KronosTokenizer.from_pretrained(LOCAL_TOKENIZER)
else:
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

tokenizer.to(DEVICE)
model.to(DEVICE)

# 冻结tokenizer (它的工作是通用时序压缩，不需要改)
for p in tokenizer.parameters():
    p.requires_grad = False
print("Tokenizer 已冻结")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")

# ═══════════════════════════════════════════════════════════════
# 3. 训练循环 — Teacher Forcing
# ═══════════════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(args.epochs):
    # ── Train ──
    model.train()
    train_loss = 0
    train_steps = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for ctx, tgt in pbar:
        ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)  # (B, ctx_days, 5), (B, pred_days, 5)
        B = ctx.size(0)

        # Tokenize context → s1_ids, s2_ids
        with torch.no_grad():
            ctx_tokens = tokenizer.encode(ctx, half=True)  # [s1_ids, s2_ids]
            tgt_tokens = tokenizer.encode(tgt, half=True)

        s1_ctx, s2_ctx = ctx_tokens[0], ctx_tokens[1]      # (B, ctx_len)
        s1_tgt = tgt_tokens[0]                               # (B, pred_len)

        # Teacher forcing: feed context s1 tokens, predict s1 for each position
        # Concat context and target s1, use teacher forcing on the whole sequence
        s1_full = torch.cat([s1_ctx, s1_tgt], dim=1)        # (B, ctx_len + pred_len)
        s2_full = torch.cat([s2_ctx, tgt_tokens[1]], dim=1)

        # Forward with teacher forcing: predict next-s1 at each position
        s1_logits, _ = model(
            s1_full[:, :-1],          # input: all but last
            s2_full[:, :-1],          # post tokens
            use_teacher_forcing=True,
            s1_targets=s1_full[:, 1:]  # target: all but first (shifted)
        )
        # s1_logits: (B, total_len-1, vocab_size)

        loss = criterion(
            s1_logits[:, -args.pred_days:].reshape(-1, s1_logits.size(-1)),
            s1_tgt.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        train_steps += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_loss /= train_steps
    history['train_loss'].append(train_loss)

    # ── Val ──
    model.eval()
    val_loss = 0
    val_steps = 0
    with torch.no_grad():
        for ctx, tgt in val_loader:
            ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)
            B = ctx.size(0)

            ctx_tokens = tokenizer.encode(ctx, half=True)
            tgt_tokens = tokenizer.encode(tgt, half=True)
            s1_ctx, s2_ctx = ctx_tokens[0], ctx_tokens[1]
            s1_tgt = tgt_tokens[0]

            s1_full = torch.cat([s1_ctx, s1_tgt], dim=1)
            s2_full = torch.cat([s2_ctx, tgt_tokens[1]], dim=1)

            s1_logits, _ = model(
                s1_full[:, :-1], s2_full[:, :-1],
                use_teacher_forcing=True,
                s1_targets=s1_full[:, 1:]
            )
            loss = criterion(
                s1_logits[:, -args.pred_days:].reshape(-1, s1_logits.size(-1)),
                s1_tgt.reshape(-1)
            )
            val_loss += loss.item()
            val_steps += 1

    val_loss /= val_steps
    history['val_loss'].append(val_loss)

    scheduler.step()

    print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        print(f"  → 保存最佳模型到 {args.output_dir}")

# ═══════════════════════════════════════════════════════════════
# 4. 保存
# ═══════════════════════════════════════════════════════════════
with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
    json.dump(history, f)

print(f"\n{'='*60}")
print(f"微调完成! 最佳验证loss: {best_val_loss:.4f}")
print(f"模型: {args.output_dir}")
print(f"")
print(f"部署步骤:")
print(f"  1. 打包: tar czf kronos-btc.tar.gz {args.output_dir}")
print(f"  2. 传到服务器: scp kronos-btc.tar.gz user@server:~/")
print(f"  3. 在服务器解压到: /home/myuser/websocket_new/kronos_model/kronos-btc-finetuned/")
print(f"  4. 修改 kronos_features.py: KRONOS_MODEL_NAME = 'kronos-btc-finetuned'")
print(f"{'='*60}")
