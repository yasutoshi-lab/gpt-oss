# GPT-OSS

## <u>概要</u>

openai/gpt-oss-20b, 120bのアーキテクチャをもとに、簡易的なモデルを再現するためのリポジトリです  

<img src=gpt-oss-20b.png width=500>

## <u>構成</u>

```bash
gpt-oss
├── README.md
├── gpt-oss-20b.png # gpt-oss-20bアーキテクチャ画像
├── pyproject.toml
└── src
    ├── inference.py  # 推論スクリプト
    ├── model_scratch.py # 訓練スクリプト
    └── sample_text.txt # 500件のサンプルデータセット
```

## <u>検証環境</u>

### ハードウェア要件

- **GPU**: rtx4090 16GB
- **RAM**: 62GB
- **DISK**: 2TB
- **CPU**: i9-14900HX

### システム要件

- **OS**: Ubuntu22.04.5
- **Architecture**: x86_64
- **UV**: 0.8.4
- **CUDA** 12.9

## <u>セットアップ</u>

- 仮想環境の作成と適用

```bash

# リポジトリのクローン
git clone https://github.com/yasutoshi-lab/gpt-oss.git

# ディレクトリ変更
cd gpt-oss/

# 仮想環境作成
uv venv --python 3.12.3

# 仮想環境適用
source .venv/bin/activate

# 依存関係インストール
uv sync
```

- 訓練の実行

```json
// デフォルトモデルパラメーター
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

```json
// デフォルト学習パラメーター
{
    // --- 最適化 ---
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "betas": [0.9, 0.95],
    "weight_decay": 0.1,
    "eps": 1e-8,

    // --- スケジューラ ---
    "scheduler": {
      "type": "CosineAnnealingLR",
      "T_max": 1000,
      "eta_min": 1e-6
    },

    // --- 学習 ---
    "epochs": 10,
    "batch_size": 5,
    "max_seq_length": 128,
    "gradient_clip_norm": 1.0,
    "precision": "bfloat16",
    "device": "cuda (if available) else cpu",
    "moe_aux_loss_weight": 0.01
}
```

```bash
uv run src/model_scratch.py
```

- 推論の実行

```bash
uv run src/inference.py --model gpt_oss_compact.pt --prompt 'GPT models are' --max-tokens 20
```

## 再現モデル構造

- パラメーター数： 1,442,041,488 (1442.0M)
- データ型：bfloat16

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

## <u>環境構築: UV</u>

```bash
# インストール
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# インストール確認
uv --version
```

## <u>環境構築: NVIDIA-Driver, CUDA-Toolkit</u>

※下記スクリプトは検証環境のシステム要件に従ったものです。実際の環境に合わせてインストールしてください  
[CUDA Toolkit 12.9 Downloads](https://developer.nvidia.com/cuda-12-9-0-download-archive)

```bash
# CUDA Toolkit Installer
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

# NVIDIA Driver Installer
sudo apt-get install -y nvidia-open
```

## <u>リファレンス</u>

[GitHub OpenAI/GPT-OSS](https://github.com/openai/gpt-oss)
[OpenAI GPT-OSSの紹介](https://openai.com/index/introducing-gpt-oss/)
[OpenAI GPT-OSSのモデルカード](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)
[HuggingFace OpenAI/GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
[HuggingFace OpenAI/GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)
