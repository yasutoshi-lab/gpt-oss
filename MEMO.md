# MEMO GPT-OSS

## UV 使い方

### UV インストール(macOS, Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### UV 初期化

```bash
# .tomlファイルが存在する場合はinitできない
uv init
```

### UV 仮想環境作成&有効化

```bash
# uv venv --python 3.12
uv venv --python 3.10

source .venv/bin/activate
```

### ホストに指定バージョンのPythonが無い場合

```bash
# uv venv --python 3.12
uv venv --python 3.10

uv python install 3.10
```

### UV 環境同期

```bash
# .to,lファイルが存在する場合に利用
uv sync
```

### UV ライブラリ追加

```bash
uv pip install 'ライブラリ'
```

### UV インストールライブラリ確認

```bash
uv pip list
```

### UV pythonバージョン確認

```bash
source .venv/bin/activate

python --version
```


## GPT-OSS-COMPACT 

### 環境確認

#### GPU

```bash
# NVIDIA ドライバー確認
nvidia-smi

# NVIDIA CUDA TOOLKIT 確認
nvcc --version

# GPU数確認
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 検証環境

```txt
OS: Linux ubuntu 6.8.0-60-generic (Ubuntu 22.04)
Architecture: x86_64
GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16376 MiB)
Python: 3.10.12
PyTorch: 2.6.0+cu124
CUDA: 12.4
uv: 0.8.4
```

### ライブラリ

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### コンパクトモデルの構築

```bash
python add_documents/src/model_scratch.py
```

### サンプルデータセット

- 件数：500件

```txt
"Hello world, this is a test.",
"The quick brown fox jumps over the lazy dog.",
"Machine learning is fascinating.",
"GPT models are powerful language models.",
"Artificial intelligence will change the world.",
"Data science unlocks hidden patterns in data.",
"Neural networks can approximate complex functions.",
```

### パラメーター

```json
{
  // --- アーキテクチャ ---
  "num_hidden_layers": 6,          // Transformer ブロック数
  "num_experts": 16,               // MoE 専門家ネットワーク数
  "experts_per_token": 4,          // 1 トークン当たりルーティングされる専門家数 (Top-K)
  "vocab_size": 32000,             // トークナイザ語彙サイズ
  "hidden_size": 1536,             // 隠れ状態の次元数（モデル幅）
  "intermediate_size": 3072,       // FFN 中間層サイズ (= hidden_size × 2)
  "swiglu_limit": 7.0,             // SwiGLU 活性化での clamp 上限

  // --- アテンション ---
  "head_dim": 64,                  // 1 ヘッド当たりの次元数
  "num_attention_heads": 24,       // マルチヘッド注意のヘッド数
  "num_key_value_heads": 4,        // GQA の Key/Value ヘッド数
  "sliding_window": 256,           // Sliding Window Attention の範囲

  // --- 位置エンコーディング (RoPE) ---
  "initial_context_length": 1024,  // RoPE テーブルの初期長
  "max_context_length": 4096,      // モデルが扱える最大シーケンス長
  "rope_theta": 10000,             // RoPE 周波数スケールの基準値
  "rope_scaling_factor": 1.0,      // RoPE スケーリング倍率
  "rope_ntk_alpha": 1.0,           // RoPE NTK スケーリング α
  "rope_ntk_beta": 32.0            // RoPE NTK スケーリング β
}
```

- **モデル容量** は主に `num_hidden_layers`, `hidden_size`, `num_experts` で決まり、VRAM 使用量や計算時間に直結します。  
- **MoE** パラメータ (`num_experts`, `experts_per_token`) を増やすと推論時にも並列に複数専門家を呼び出すため、GPU メモリだけでなく帯域にも留意が必要です。  
- **GQA** を利用することで `num_key_value_heads` を `num_attention_heads` より少なく設定し、注意計算のキー/バリューバッファを削減しています。  
- **RoPE (Rotary Position Embedding)** に関わるパラメータは長文入力での安定性や精度に影響します。


### コンパクトモデル構造

- パラメーター数： 1,442,041,488 (1442.0M)
- データ型：bfloat16

[Train GPU Memory](./report/train-gpu-memory.png)

```txt
GPTOSSCompact(
  (embed_tokens): Embedding(32000, 1536)
  (layers): ModuleList(
    (0-5): 6 x TransformerBlock(
      (attention): GroupedQueryAttention(
        (norm): RMSNorm()
        (qkv_proj): Linear(in_features=1536, out_features=2048, bias=False)
        (out_proj): Linear(in_features=1536, out_features=1536, bias=False)
        (rope): RotaryEmbedding()
      )
      (moe): MoEBlock(
        (norm): RMSNorm()
        (gate): Linear(in_features=1536, out_features=16, bias=False)
        (activation): SwiGLU()
      )
    )
  )
  (norm): RMSNorm()
  (lm_head): Linear(in_features=1536, out_features=32000, bias=False)
)
```

### 学習結果

- Terminal Log

```txt
Epoch 0, Step 0: Loss=10.6603, PPL=42629.64, Acc=0.000, LR=1.00e-04
Epoch 0, Step 10: Loss=10.4275, PPL=33775.21, Acc=0.000, LR=1.00e-04
Epoch 0, Step 20: Loss=10.1225, PPL=24897.25, Acc=0.000, LR=9.99e-05
Epoch 0, Step 30: Loss=9.8977, PPL=19884.17, Acc=0.000, LR=9.98e-05
Epoch 0, Step 40: Loss=9.6210, PPL=15078.57, Acc=0.008, LR=9.96e-05
Epoch 0, Step 50: Loss=9.2712, PPL=10627.67, Acc=0.013, LR=9.94e-05
Epoch 0, Step 60: Loss=8.9161, PPL=7450.98, Acc=0.016, LR=9.91e-05
Epoch 0, Step 70: Loss=8.4182, PPL=4528.65, Acc=0.030, LR=9.88e-05
Epoch 0, Step 80: Loss=7.8544, PPL=2577.05, Acc=0.031, LR=9.84e-05
Epoch 0, Step 90: Loss=7.2264, PPL=1375.26, Acc=0.024, LR=9.80e-05
Epoch 0 completed. Average Loss: 9.0629
Epoch 1, Step 0: Loss=6.5901, PPL=727.83, Acc=0.024, LR=9.75e-05
Epoch 1, Step 10: Loss=6.0027, PPL=404.53, Acc=0.024, LR=9.70e-05
Epoch 1, Step 20: Loss=5.2260, PPL=186.05, Acc=0.043, LR=9.65e-05
Epoch 1, Step 30: Loss=4.4330, PPL=84.19, Acc=0.052, LR=9.59e-05
Epoch 1, Step 40: Loss=3.8558, PPL=47.27, Acc=0.065, LR=9.52e-05
Epoch 1, Step 50: Loss=3.6701, PPL=39.25, Acc=0.061, LR=9.45e-05
Epoch 1, Step 60: Loss=3.7437, PPL=42.25, Acc=0.052, LR=9.38e-05
Epoch 1, Step 70: Loss=3.5669, PPL=35.41, Acc=0.063, LR=9.30e-05
Epoch 1, Step 80: Loss=3.4046, PPL=30.10, Acc=0.054, LR=9.22e-05
Epoch 1, Step 90: Loss=3.2123, PPL=24.84, Acc=0.088, LR=9.14e-05
Epoch 1 completed. Average Loss: 4.6422
Epoch 2, Step 0: Loss=3.2183, PPL=24.98, Acc=0.076, LR=9.05e-05
Epoch 2, Step 10: Loss=3.1435, PPL=23.18, Acc=0.088, LR=8.95e-05
Epoch 2, Step 20: Loss=3.0749, PPL=21.65, Acc=0.079, LR=8.85e-05
Epoch 2, Step 30: Loss=3.0984, PPL=22.16, Acc=0.060, LR=8.75e-05
Epoch 2, Step 40: Loss=3.1621, PPL=23.62, Acc=0.088, LR=8.65e-05
Epoch 2, Step 50: Loss=3.0901, PPL=21.98, Acc=0.082, LR=8.54e-05
Epoch 2, Step 60: Loss=3.0433, PPL=20.97, Acc=0.085, LR=8.43e-05
Epoch 2, Step 70: Loss=3.0329, PPL=20.76, Acc=0.085, LR=8.31e-05
Epoch 2, Step 80: Loss=3.0596, PPL=21.32, Acc=0.077, LR=8.19e-05
Epoch 2, Step 90: Loss=2.9335, PPL=18.79, Acc=0.091, LR=8.07e-05
Epoch 2 completed. Average Loss: 3.0834
```

### 推論テスト

```bash
python add_documents/src/inference.py --model gpt_oss_compact.pt --prompt 'GPT models are' --max-tokens 20
```

[Inference GPU Memory](report/inference-gpu-memory.png)

