# GPT-OSS Compact Model - セットアップ・実行ガイド

RTX 4090 16GB向けに最適化されたGPT-OSS Compactモデルのセットアップから推論テストまでの完全ガイド

## 🖥️ システム要件

### ハードウェア要件
- **GPU**: NVIDIA RTX 4090 16GB以上推奨
- **RAM**: 32GB以上推奨（62GB使用確認済み）
- **ストレージ**: 10GB以上の空き容量
- **CPU**: Intel i9-14900HX相当以上推奨

### 検証済み環境
```bash
OS: Linux ubuntu 6.8.0-60-generic (Ubuntu 22.04)
Architecture: x86_64
GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16376 MiB)
Python: 3.10.12
PyTorch: 2.6.0+cu124
CUDA: 12.4
uv: 0.8.4
```

## 📦 事前準備

### 1. NVIDIA ドライバー・CUDAのインストール
```bash
# NVIDIA ドライバー確認
nvidia-smi

# CUDAが利用可能か確認
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. uvパッケージマネージャーのインストール
```bash
# uvが未インストールの場合
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# インストール確認
uv --version
```

## 🚀 セットアップ手順

### Step 1: プロジェクトディレクトリの準備
```bash
# gpt-ossリポジトリのクローン（または既存ディレクトリに移動）
cd /path/to/gpt-oss
```

### Step 2: uv仮想環境の作成
```bash
# 仮想環境作成（Python 3.10指定）
uv venv --python 3.10 gpt-oss-compact

# 仮想環境のアクティベート
source gpt-oss-compact/bin/activate

# 確認
which python
python --version
```

### Step 3: 必要なパッケージのインストール
```bash
# PyTorch (CUDA 12.4対応版)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 追加パッケージ
uv pip install numpy

# GPU確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 4: モデルファイルの確認
```bash
# model_scratch.pyの存在確認
ls -la add_documents/model_scratch.py

# ファイル権限確認
chmod +x add_documents/model_scratch.py
```

## 🔧 実行手順

### 1. 学習・推論の実行
```bash
# GPT-OSS Compactモデルの実行
cd /path/to/gpt-oss
python add_documents/model_scratch.py
```

### 2. 期待される出力
```
=== GPT-OSS Compact Demo ===
RTX 4090 16GB optimized implementation
Using device: cuda
GPU Memory: 16.6GB
Model config: ModelConfig(num_hidden_layers=6, num_experts=8, ...)
Model created with 202,934,088 parameters (202.9M)
Estimated parameter memory: 0.4GB
Dataset created with 50 examples

Starting training demo...
Starting training for 1 epochs...
Model parameters: 202,934,088
Epoch 0, Step 0: Loss=10.4826, PPL=35690.48, Acc=0.000, LR=1.00e-04
Epoch 0, Step 10: Loss=10.0376, PPL=22870.22, Acc=0.000, LR=1.00e-04
Epoch 0 completed. Average Loss: 10.2871

Inference demo...
Prompt: 'Hello'
Generated: 'Hello...(生成されたテキスト)'
Model saved to gpt_oss_compact.pt

Demo completed successfully!
```

### 3. 生成されるファイル
```bash
# モデルファイル
ls -la gpt_oss_compact.pt
```

## 🧪 推論テスト

### カスタム推論の実行
```python
#!/usr/bin/env python3
"""カスタム推論テスト"""
import torch
from model_scratch import GPTOSSCompact, ModelConfig, SimpleTokenizer

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデル読み込み
config = ModelConfig()
model = GPTOSSCompact(config, device=device)
model.load_state_dict(torch.load('gpt_oss_compact.pt', map_location=device))
model.eval()

# トークナイザー
tokenizer = SimpleTokenizer()

# 推論テスト
def generate_text(prompt, max_length=50):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        # シンプルなグリーディー生成
        for _ in range(max_length):
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            # 新しいトークンを追加
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            logits, _ = model(input_ids)
    
    return tokenizer.decode(input_ids[0].cpu().tolist())

# テスト実行
result = generate_text("Hello world")
print(f"Generated: {result}")
```

## ⚡ パフォーマンス最適化

### メモリ使用量の最適化
```python
# model_scratch.py内の設定調整例

# より小さなモデル（100M parameters）
@dataclass
class ModelConfig:
    num_hidden_layers: int = 4      # 6 → 4
    num_experts: int = 4           # 8 → 4  
    experts_per_token: int = 2     # 維持
    hidden_size: int = 512         # 768 → 512
    # ... その他設定
```

### 推論速度の最適化
```python
# KV Cacheを活用した高速推論
# model_scratch.py内で既に実装済み

# GPU最適化設定
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## 🔧 トラブルシューティング

### よくある問題と解決法

#### 1. CUDA Out of Memory
```bash
# エラー: RuntimeError: CUDA out of memory
```
**解決法**:
```python
# バッチサイズを削減
batch_size = 2  # 4 → 2

# モデルサイズを削減
hidden_size = 512  # 768 → 512
num_experts = 4    # 8 → 4
```

#### 2. PyTorchのバージョン互換性
```bash
# エラー: BFloat16 not supported
```
**解決法**:
```bash
# PyTorchの再インストール
uv pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### 3. 依存関係の問題
```bash
# エラー: Module not found
```
**解決法**:
```bash
# 仮想環境の再作成
uv venv --python 3.10 gpt-oss-compact-new
source gpt-oss-compact-new/bin/activate
uv pip install torch numpy
```

#### 4. GPU検出の問題
```bash
# CUDA is not available
```
**解決法**:
```bash
# CUDAドライバー確認
nvidia-smi

# PyTorch CUDA確認
python -c "import torch; print(torch.version.cuda)"

# 必要に応じてドライバー再インストール
```

## 📊 メトリクス・ベンチマーク

### 期待される性能指標
- **パラメータ数**: ~203M
- **メモリ使用量**: ~0.4GB (パラメータ) + 学習用オーバーヘッド
- **学習速度**: ~10秒/epoch (小規模データセット)
- **推論速度**: ~100 tokens/sec
- **最大コンテキスト長**: 4096 tokens

### メモリプロファイリング
```python
# GPUメモリ使用量確認
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")

print_gpu_memory()
```

## 🔬 カスタマイズ・拡張

### モデル構成の変更
```python
# add_documents/model_scratch.py内のModelConfig
@dataclass 
class ModelConfig:
    # レイヤー数調整
    num_hidden_layers: int = 8      # より深いモデル
    
    # 専門家数調整  
    num_experts: int = 16           # より多くの専門家
    experts_per_token: int = 4      # アクティブ専門家数
    
    # 隠れ層サイズ調整
    hidden_size: int = 1024         # より大きなモデル
```

### データセットの変更
```python
# カスタムデータセット作成
class CustomDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        # カスタム実装
        pass
```

## 📝 実行ログ例

### 成功例
```
=== GPT-OSS Compact Demo ===
RTX 4090 16GB optimized implementation
Using device: cuda
GPU Memory: 16.6GB
Model config: ModelConfig(num_hidden_layers=6, num_experts=8, experts_per_token=2, vocab_size=32000, hidden_size=768, intermediate_size=1536, swiglu_limit=7.0, head_dim=64, num_attention_heads=12, num_key_value_heads=2, sliding_window=128, initial_context_length=1024, max_context_length=4096, rope_theta=10000.0, rope_scaling_factor=1.0, rope_ntk_alpha=1.0, rope_ntk_beta=32.0)
Model created with 202,934,088 parameters (202.9M)
Estimated parameter memory: 0.4GB
Dataset created with 50 examples

Starting training demo...
Starting training for 1 epochs...
Model parameters: 202,934,088
Epoch 0, Step 0: Loss=10.4826, PPL=35690.48, Acc=0.000, LR=1.00e-04
Epoch 0, Step 10: Loss=10.0376, PPL=22870.22, Acc=0.000, LR=1.00e-04
Epoch 0 completed. Average Loss: 10.2871

Inference demo...
Prompt: 'Hello'
Generated: 'Hello<UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK>c<UNK><UNK><UNK><UNK><UNK><UNK><UNK>'
Model saved to gpt_oss_compact.pt

Demo completed successfully!
Model saved as: gpt_oss_compact.pt
```

## 🤝 コミュニティ・サポート

### 関連リンク
- [GPT-OSS GitHub Repository](https://github.com/gpt-oss/gpt-oss)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [uv Package Manager](https://github.com/astral-sh/uv)

### 貢献・改善
モデルの改善・最適化に関する提案や問題報告は、GitHubリポジトリのIssuesでお願いします。

---

**作成日**: 2025-01-06  
**テスト環境**: Ubuntu 22.04, RTX 4090 16GB, Python 3.10.12  
**モデルバージョン**: GPT-OSS Compact v1.0