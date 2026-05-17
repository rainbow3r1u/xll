# Kronos BTC 微调 — AutoDL 训练包

## 已包含（无需联网）

```
kronos_finetune/
├── train.py                         # 训练脚本
├── btc_klines.json                  # BTC 2001天日线 (187KB)
├── kronos_pretrained/               # 预训练模型 (已下载, 无需HF)
│   ├── Kronos-base/                 # 最大版本, 391MB
│   └── Kronos-Tokenizer-base/       # Tokenizer, 16MB
├── kronos_model/                    # 模型代码
│   ├── kronos.py                    # Kronos Transformer 定义
│   ├── module.py                    # BSQuantizer 量化模块
│   └── __init__.py
├── requirements.txt
└── README.md
```

## AutoDL 步骤

### 1. 上传
把整个 `kronos_finetune/` 目录（约 410MB）上传到 AutoDL。

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 训练
```bash
# 默认参数即可 (20 epochs, 256天上下文, 预测7天, Kronos-base)
python train.py

# 自定义参数
python train.py --epochs 30 --context_days 512 --pred_days 3 --lr 1e-4
```

参数说明:
- `--epochs`      训练轮数 (10-30)
- `--context_days` 输入上下文天数 (128/256/512)
- `--pred_days`    预测未来天数 (2/3/5/7)
- `--lr`           学习率 (默认 5e-5)
- `--output_dir`   输出目录 (默认 ./kronos-btc-finetuned)

### 4. 下载结果
训练完成后，`kronos-btc-finetuned/` 目录包含微调后的模型。

### 5. 部署回服务器
```bash
# 把微调后的模型放到服务器
scp -r kronos-btc-finetuned/ user@server:~/websocket_new/kronos_model/

# 修改 kronos_features.py 第23行:
KRONOS_MODEL_NAME = "./kronos-btc-finetuned"
```

## 原理

Kronos 是时序 Transformer，分两步：

1. **Tokenizer (编码器)**: 把 K 线 (OHLCV) 压缩成离散 token，类似图片→像素块
2. **Kronos 模型**: 在 token 空间自回归预测——给前 256 天的 token，预测未来 token

微调 = 用 BTC 历史做 teacher forcing，让 Transformer decoder 更适配 BTC 的价格模式。
冻结 tokenizer（压缩是通用的），只训练 decoder 的预测能力。

## 模型选择

- Kronos-base (已打包): 参数量最大，效果最好，需要 GPU
- Kronos-small: 24.7M 参数，CPU 可用
- Tokenizer-base: 通用时序压缩器，无需微调
