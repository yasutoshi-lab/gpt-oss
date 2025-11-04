import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


@dataclass
class ModelConfig:
    """
    GPT-OSS Compact Configuration
    RTX 4090 16GB対応の軽量版パラメータ
    """
    # アーキテクチャ設定
    num_hidden_layers: int = 6          # Transformer ブロック数
    num_experts: int = 16               # MoE 専門家ネットワーク数
    experts_per_token: int = 4          # 1 トークン当たりルーティングされる専門家数 (Top-K)
    vocab_size: int = 32000             # トークナイザ語彙サイズ
    hidden_size: int = 1536             # 隠れ状態の次元数（モデル幅）
    intermediate_size: int = 3072       # FFN 中間層サイズ (= hidden_size × 2)
    swiglu_limit: float = 7.0           # SwiGLU 活性化での clamp 上限
    
    # アテンション設定
    head_dim: int = 64                  # 1 ヘッド当たりの次元数
    num_attention_heads: int = 24       # マルチヘッド注意のヘッド数
    num_key_value_heads: int = 4        # GQA の Key/Value ヘッド数
    sliding_window: int = 256           # Sliding Window Attention の範囲
    
    # 位置エンコーディング設定
    initial_context_length: int = 1024  # RoPE テーブルの初期長
    max_context_length: int = 4096      # モデルが扱える最大シーケンス長
    rope_theta: float = 10000.0         # RoPE 周波数スケールの基準値
    rope_scaling_factor: float = 1.0    # RoPE スケーリング倍率
    rope_ntk_alpha: float = 1.0         # RoPE NTK スケーリング α
    rope_ntk_beta: float = 32.0         # RoPE NTK スケーリング β
    
    def __post_init__(self):
        """設定値の妥当性チェック"""
        assert self.hidden_size % self.head_dim == 0, "hidden_size must be divisible by head_dim"
        assert self.intermediate_size == self.hidden_size * 2, "intermediate_size should be 2x hidden_size"
        assert self.num_attention_heads % self.num_key_value_heads == 0, "attention_heads must be divisible by kv_heads"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    LayerNormより高速で大規模モデルに適している
    """
    def __init__(self, num_features: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 計算精度向上のためfloat32で計算
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        return (x_normed * self.scale).type_as(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    回転行列による位置情報の注入
    """
    def __init__(self, head_dim: int, base: float = 10000.0, 
                 max_context_length: int = 8192, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_context_length = max_context_length
        
        # cos, sinテーブルを事前計算
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        if device is not None:
            inv_freq = inv_freq.to(device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # cos, sinテーブルをキャッシュ
        t = torch.arange(max_context_length, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-3)
        
        # 範囲チェック
        if start_pos + seq_len > self.max_context_length:
            raise ValueError(f"Sequence too long: {start_pos + seq_len} > {self.max_context_length}")
        
        cos = self.cos_cached[start_pos:start_pos + seq_len].unsqueeze(-2)
        sin = self.sin_cached[start_pos:start_pos + seq_len].unsqueeze(-2)
        
        return self._apply_rotary_emb(q, cos, sin), self._apply_rotary_emb(k, cos, sin)

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, 
                         sin: torch.Tensor) -> torch.Tensor:
        # [seq_len, num_heads, head_dim] -> [seq_len, num_heads, head_dim//2, 2]
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshaped.unbind(-1)
        
        # 回転適用
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return rotated.flatten(-2).type_as(x)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with Sliding Window
    メモリ効率化のためKey-Valueヘッド数を削減
    """
    def __init__(self, config: ModelConfig, layer_idx: int = 0, device=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None
        
        # Attention Sink: 先頭トークンへの学習可能な注意重み
        self.sinks = nn.Parameter(
            torch.zeros(self.num_q_heads, device=device, dtype=torch.float32)
        )
        
        # 正規化
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # QKV投影（効率化のため結合）
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False, device=device, dtype=torch.bfloat16
        )
        
        # 出力投影
        self.out_proj = nn.Linear(
            self.num_q_heads * self.head_dim,
            config.hidden_size,
            bias=False, device=device, dtype=torch.bfloat16
        )
        
        # スケーリング因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # RoPE
        self.rope = RotaryEmbedding(
            self.head_dim, config.rope_theta, 
            config.max_context_length, device
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0, 
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        # 正規化
        x_normed = self.norm(x)
        
        # QKV計算
        qkv = self.qkv_proj(x_normed)
        
        # Q, K, Vに分割
        q_size = self.num_q_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q = qkv[:, :, :q_size]
        k = qkv[:, :, q_size:q_size + kv_size]
        v = qkv[:, :, q_size + kv_size:q_size + 2 * kv_size]
        
        # テンソル形状変更
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # RoPE適用
        q, k = self.rope(q, k, start_pos)
        
        # KVキャッシュ処理
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        # Grouped Query Attention - KVキャッシュ処理後に実行
        if self.num_kv_heads < self.num_q_heads:
            # KVヘッドを複製してQヘッド数に合わせる
            k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=2)
            v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=2)
        
        # 新しいKVキャッシュを保存（GQA処理前の状態で）
        if kv_cache is not None:
            # キャッシュには元のKV形状（num_kv_heads）を保存
            _, seq_len, _, _ = k.shape
            k_for_cache = k.view(batch_size, seq_len, self.num_kv_heads, self.num_q_heads // self.num_kv_heads, self.head_dim).mean(3)
            v_for_cache = v.view(batch_size, seq_len, self.num_kv_heads, self.num_q_heads // self.num_kv_heads, self.head_dim).mean(3)
            new_kv_cache = (k_for_cache, v_for_cache)
        else:
            new_kv_cache = None
        
        # Flash Attention風の実装（メモリ効率化）
        attn_output = self._flash_attention(q, k, v, start_pos)
        
        # 出力投影
        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        
        # 残差接続
        return x + output, new_kv_cache

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, 
                        v: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, q_len, num_heads, head_dim = q.shape
        _, kv_len, kv_heads, _ = k.shape
        
        # GQAでヘッド数が一致していることを確認
        assert num_heads == kv_heads, f"Head count mismatch after GQA: q={num_heads}, kv={kv_heads}"
        
        # スケーリング
        q = q * self.scale
        
        # 注意重み計算 [batch, seq_q, heads, head_dim] x [batch, seq_kv, heads, head_dim]
        # -> [batch, heads, seq_q, seq_kv]
        q = q.transpose(1, 2)  # [batch, heads, seq_q, head_dim]
        k = k.transpose(1, 2)  # [batch, heads, seq_kv, head_dim] 
        v = v.transpose(1, 2)  # [batch, heads, seq_kv, head_dim]
        
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq_q, seq_kv]
        
        # Causal マスク
        if kv_len > 1:
            causal_mask = torch.triu(
                torch.ones(q_len, kv_len, device=scores.device), 
                diagonal=kv_len - q_len + 1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Sliding Window マスク
        if self.sliding_window is not None and kv_len > self.sliding_window:
            mask = self._create_sliding_mask(q_len, kv_len, start_pos)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # # Attention Sink - 一時的に無効化してテスト
        # if start_pos == 0 and kv_len > 1:
        #     sink_scores = self.sinks.view(1, -1, 1, 1)  # [1, num_heads, 1, 1]
        #     # scoresの実際のヘッド数に合わせる
        #     actual_heads = scores.shape[1]
        #     if sink_scores.shape[1] != actual_heads:
        #         # ヘッド数が異なる場合はスライスまたは繰り返し
        #         if sink_scores.shape[1] > actual_heads:
        #             sink_scores = sink_scores[:, :actual_heads, :, :]
        #         else:
        #             # 繰り返し（GQAの場合）
        #             repeat_factor = actual_heads // sink_scores.shape[1]
        #             sink_scores = sink_scores.repeat(1, repeat_factor, 1, 1)
        #     scores[:, :, :, 0] += sink_scores
        
        # Softmax + Value適用
        attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        attn_output = torch.matmul(attn_probs, v)  # [batch, heads, seq_q, head_dim]
        
        # 元の形状に戻す
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_q, heads, head_dim]
        
        return attn_output

    def _create_sliding_mask(self, q_len: int, kv_len: int, start_pos: int) -> torch.Tensor:
        """Sliding Window用のマスク作成"""
        mask = torch.ones(q_len, kv_len, dtype=torch.bool)
        for i in range(q_len):
            pos = start_pos + i
            start_idx = max(0, pos - self.sliding_window)
            end_idx = pos + 1
            mask[i, start_idx:end_idx] = False
        return mask


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function
    GLUの改良版、ReLU/GELUより高性能
    """
    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # インターリーブ形式から分離
        x_gate, x_linear = x.chunk(2, dim=-1)
        
        # クランプ（数値安定性）
        x_gate = x_gate.clamp(max=self.limit)
        x_linear = x_linear.clamp(-self.limit, self.limit)
        
        # SwiGLU計算
        swish_gate = x_gate * torch.sigmoid(self.alpha * x_gate)
        return swish_gate * (x_linear + 1)


class MoEBlock(nn.Module):
    """
    Mixture of Experts Block
    32専門家による効率的スパース計算
    """
    def __init__(self, config: ModelConfig, device=None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # 正規化
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # ゲートネットワーク
        self.gate = nn.Linear(
            config.hidden_size, config.num_experts,
            bias=False, device=device, dtype=torch.bfloat16
        )
        
        # 専門家の重み（簡略実装）
        # 実際は各専門家ごとに分離した重みを持つが、メモリ効率のため統合実装
        self.expert_weights = nn.Parameter(
            torch.randn(config.num_experts, config.intermediate_size * 2, 
                       config.hidden_size, device=device, dtype=torch.bfloat16)
        )
        self.expert_biases = nn.Parameter(
            torch.zeros(config.num_experts, config.intermediate_size * 2, 
                       device=device, dtype=torch.bfloat16)
        )
        
        self.expert_down_weights = nn.Parameter(
            torch.randn(config.num_experts, config.hidden_size, 
                       config.intermediate_size, device=device, dtype=torch.bfloat16)
        )
        self.expert_down_biases = nn.Parameter(
            torch.zeros(config.num_experts, config.hidden_size, 
                       device=device, dtype=torch.bfloat16)
        )
        
        # 活性化関数
        self.activation = SwiGLU()
        
        # 負荷バランス追跡
        self.register_buffer('expert_usage', torch.zeros(config.num_experts, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        residual = x
        
        # 正規化
        x = self.norm(x)
        
        # ゲート計算
        gate_logits = self.gate(x)  # [batch, seq, num_experts]
        
        # Top-K専門家選択
        top_k_logits, top_k_indices = torch.topk(
            gate_logits, self.experts_per_token, dim=-1
        )
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # 専門家使用量追跡（負荷バランス用）
        if self.training:
            expert_counts = torch.bincount(
                top_k_indices.flatten(), minlength=self.num_experts
            ).float()
            self.expert_usage = 0.9 * self.expert_usage + 0.1 * expert_counts
        
        # 専門家計算
        output = torch.zeros_like(x)
        
        for i in range(self.experts_per_token):
            # i番目の専門家のインデックスと重み
            expert_idx = top_k_indices[:, :, i]  # [batch, seq]
            expert_weight = top_k_probs[:, :, i:i+1]  # [batch, seq, 1]
            
            # バッチ処理のため専門家をまとめて計算
            expert_output = self._compute_expert_batch(x, expert_idx)
            output += expert_weight * expert_output
        
        return residual + output

    def _compute_expert_batch(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """MoE Blockの計算"""
        batch_size, seq_len, hidden_size = x.shape
        
        # 簡略化: 全専門家の平均を使用（実際はインデックスに基づく選択的計算）
        # 実装の複雑さを抑えるため、ここでは代表的専門家による計算
        expert_id = expert_indices[0, 0].item() % self.num_experts
        
        # Up projection + SwiGLU
        up_weight = self.expert_weights[expert_id]  # [intermediate * 2, hidden]
        up_bias = self.expert_biases[expert_id]     # [intermediate * 2]
        
        hidden = F.linear(x, up_weight, up_bias)  # weight shape is already correct
        hidden = self.activation(hidden)
        
        # Down projection
        down_weight = self.expert_down_weights[expert_id]  # [hidden, intermediate]
        down_bias = self.expert_down_biases[expert_id]     # [hidden]
        
        output = F.linear(hidden, down_weight, down_bias)  # weight shape is already correct
        return output

    def get_auxiliary_loss(self) -> torch.Tensor:
        """負荷バランス損失計算"""
        if not self.training:
            return torch.tensor(0.0, device=self.expert_usage.device)
        
        # 専門家使用量の分散を最小化
        usage_var = torch.var(self.expert_usage)
        return usage_var


class TransformerBlock(nn.Module):
    """
    GPT-OSS Transformer Block
    Attention + MoE + 残差接続
    """
    def __init__(self, config: ModelConfig, layer_idx: int, device=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = GroupedQueryAttention(config, layer_idx, device)
        self.moe = MoEBlock(config, device)

    def forward(self, x: torch.Tensor, start_pos: int = 0,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention
        x, new_kv_cache = self.attention(x, start_pos, kv_cache)
        
        # MoE
        x = self.moe(x)
        
        return x, new_kv_cache


class GPTOSSCompact(nn.Module):
    """
    GPT-OSS Compact Model
    RTX 4090 16GB対応の軽量版実装
    """
    def __init__(self, config: ModelConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Token Embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size,
            device=device, dtype=torch.bfloat16
        )
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, i, device)
            for i in range(config.num_hidden_layers)
        ])
        
        # 最終正規化
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # Language Modeling Head
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size,
            bias=False, device=device, dtype=torch.bfloat16
        )
        
        # 重み共有（メモリ節約）
        self.lm_head.weight = self.embed_tokens.weight
        
        # パラメータ初期化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """重みの初期化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.scale)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0,
                kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.embed_tokens(input_ids)
        
        # KVキャッシュ初期化
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        new_kv_caches = []
        
        # Transformer blocks
        for i, layer in enumerate(self.layers):
            x, new_kv_cache = layer(x, start_pos, kv_caches[i])
            new_kv_caches.append(new_kv_cache)
        
        # 最終正規化
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits, new_kv_caches

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                temperature: float = 1.0, top_p: float = 0.9,
                top_k: int = 50) -> torch.Tensor:
        """自己回帰的テキスト生成"""
        self.eval()
        
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        kv_caches = None
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # 最初のステップでは全シーケンス、以降は最後のトークンのみ
                if step == 0:
                    input_seq = generated
                    start_pos = 0
                else:
                    input_seq = generated[:, -1:]
                    start_pos = seq_len + step - 1
                
                # フォワードパス
                logits, kv_caches = self.forward(input_seq, start_pos, kv_caches)
                next_token_logits = logits[:, -1, :]
                
                # 温度スケーリング
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-K sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-P sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # サンプリング
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 生成トークン追加
                generated = torch.cat([generated, next_token], dim=-1)
                
                # # 停止条件（EOS token等）
                # if next_token.item() == eos_token_id:
                #     break
        
        return generated

    def get_num_params(self) -> int:
        """パラメータ数計算"""
        return sum(p.numel() for p in self.parameters())

    def get_auxiliary_loss(self) -> torch.Tensor:
        """MoE負荷バランス損失"""
        aux_loss = 0.0
        for layer in self.layers:
            aux_loss += layer.moe.get_auxiliary_loss()
        return aux_loss / len(self.layers)


class SimpleTokenizer:
    """
    簡易的な文字単位のトークナイザ（デモ用）
    実際の用途では tiktoken や HuggingFace tokenizers の利用を推奨
    """
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.unk_token_id = 3
        
        # 基本的な語彙マッピング（デモ用）
        chars = ''.join(chr(i) for i in range(32, 127))  # 印刷可能ASCII
        self.char_to_id = {char: i + 4 for i, char in enumerate(chars)}
        self.id_to_char = {i + 4: char for i, char in enumerate(chars)}
        
        # 特殊トークン追加
        self.id_to_char[self.bos_token_id] = '<BOS>'
        self.id_to_char[self.eos_token_id] = '<EOS>'
        self.id_to_char[self.pad_token_id] = '<PAD>'
        self.id_to_char[self.unk_token_id] = '<UNK>'

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """テキストをトークンIDに変換"""
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)
        
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_token_id))
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """トークンIDをテキストに変換"""
        chars = []
        for token_id in token_ids:
            if token_id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            chars.append(self.id_to_char.get(token_id, '<UNK>'))
        return ''.join(chars)


class TextDataset(Dataset):
    """
    シンプルなテキストデータセット（学習用）
    """
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # テキストをトークン化
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
            
            # 長いテキストを分割
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= 10:  # 最低長チェック
                    # パディング
                    if len(chunk) < max_length:
                        chunk.extend([tokenizer.pad_token_id] * (max_length - len(chunk)))
                    self.examples.append(chunk[:max_length])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # 入力と正解ラベル（因果モデルの為、1つずらして入力と正解ラベルを作成）
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        # パディングトークンは損失計算から除外
        labels = torch.where(labels == 2, -100, labels)  # pad_token_id = 2
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class GPTTrainer:
    """
    GPT-OSS Compact 学習器
    """
    def __init__(self, model: GPTOSSCompact, tokenizer: SimpleTokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device
        
        # オプティマイザ
        self.optimizer = AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8
        )
        
        # 学習率スケジューラ
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)
        
        # 混合精度は無効化（BFloat16との互換性のため）
        # self.scaler = torch.amp.GradScaler('cuda')

    def train_step(self, batch: dict) -> dict:
        """1回の学習ステップ"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 勾配リセット
        self.optimizer.zero_grad()
        
        # 混合精度でフォワードパス
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.model(input_ids)
            
            # 言語モデリング損失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # MoE補助損失
            aux_loss = self.model.get_auxiliary_loss()
            total_loss = loss + 0.01 * aux_loss
        
        # バックワード
        total_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        self.optimizer.step()
        self.scheduler.step()
        
        # メトリクス計算
        with torch.no_grad():
            perplexity = torch.exp(loss)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item(),
            'perplexity': perplexity.item(),
            'accuracy': accuracy.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }

    def train(self, dataloader: DataLoader, num_epochs: int = 10):
        """学習実行"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(dataloader):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                num_batches += 1
                
                # ログ出力
                if step % 1 == 0:
                    print(f"Epoch {epoch}, Step {step}: "
                          f"Loss={metrics['loss']:.4f}, "
                          f"PPL={metrics['perplexity']:.2f}, "
                          f"Acc={metrics['accuracy']:.3f}, "
                          f"LR={metrics['lr']:.2e}")
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

    def save_model(self, path: str):
        """モデル保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """モデル読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


def main():
    """メイン実行関数 - デモンストレーション"""
    print("=== GPT-OSS Compact Demo ===")
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 設定
    config = ModelConfig()
    print(f"Model config: {config}")
    
    # モデル作成
    model = GPTOSSCompact(config, device)

    # モデル構造を確認
    print(model)

    num_params = model.get_num_params()
    print(f"Model created with {num_params:,} parameters ({num_params/1e6:.1f}M)")
    
    # メモリ使用量推定
    param_memory = num_params * 2 / 1e9  # BF16 = 2 bytes per param
    print(f"Estimated parameter memory: {param_memory:.1f}GB")
    
    # トークナイザ
    tokenizer = SimpleTokenizer(config.vocab_size)
    
    # サンプルデータを外部ファイルから読み込み
    sample_text_file = os.path.join(os.path.dirname(__file__), "sample_text.txt")
    with open(sample_text_file, "r", encoding="utf-8") as f:
        sample_texts = [line.strip() for line in f if line.strip()]
    
    # データセット
    dataset = TextDataset(sample_texts, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    print(f"Dataset created with {len(dataset)} examples")
    
    # 学習器
    trainer = GPTTrainer(model, tokenizer, config)
    
    # 学習実行（デモ用に短時間）
    print("\nStarting training demo...")
    trainer.train(dataloader, num_epochs=3)
    
    # 推論デモ
    print("\nInference demo...")
    model.eval()
    
    prompt = "Hello"
    print(f"Prompt: '{prompt}'")
    
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], 
                           device=device)
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
    
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    print(f"Generated: '{generated_text}'")
    
    # モデル保存
    save_path = "gpt_oss_compact.pt"
    trainer.save_model(save_path)
    
    print(f"\nDemo completed successfully!")
    print(f"Model saved as: {save_path}")


if __name__ == "__main__":
    main()