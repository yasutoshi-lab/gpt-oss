import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# model_scratch.pyから必要なクラスをインポート
sys.path.append(os.path.dirname(__file__))
from model_scratch import (
    ModelConfig, GPTOSSCompact, SimpleTokenizer,
    RMSNorm, RotaryEmbedding, GroupedQueryAttention,
    SwiGLU, MoEBlock, TransformerBlock
)


class GPTOSSInference:
    """
    GPT-OSS Compact推論テスター
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス
            device: 使用デバイス ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        
        # デバイス設定
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # モデルとトークナイザーを読み込み
        self.model = None
        self.tokenizer = None
        self.config = None
        
        self._load_model()
    
    def _load_model(self):
        """モデルの読み込み"""
        print(f"Loading model from {self.model_path}...")
        
        try:
            # PyTorch 2.6以降のweights_only=True問題を回避
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=False  # 信頼できるファイルなのでFalseを使用
            )
            
            print("Checkpoint keys:", list(checkpoint.keys()))
            
            # 設定を取得
            self.config = checkpoint.get('config')
            if self.config is None:
                print("Warning: No config found in checkpoint, using default config")
                self.config = ModelConfig()
            
            print(f"Model config: {self.config}")
            
            # モデルを作成
            self.model = GPTOSSCompact(self.config, self.device)
            
            # 重みを読み込み
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Model weights loaded successfully")
            else:
                print("Warning: No model_state_dict found in checkpoint")
            
            # 評価モードに設定
            self.model.eval()
            
            # トークナイザーを作成
            self.tokenizer = SimpleTokenizer(self.config.vocab_size)
            
            # パラメータ数を表示
            num_params = self.model.get_num_params()
            print(f"Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")
            
            # メモリ使用量を推定
            param_memory = num_params * 2 / 1e9  # BF16 = 2 bytes per param
            print(f"Estimated parameter memory: {param_memory:.1f}GB")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_text(self, 
                     prompt: str,
                     max_new_tokens: int = 50,
                     temperature: float = 0.8,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     verbose: bool = True) -> str:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数
            temperature: 温度パラメータ
            top_p: Top-p sampling値
            top_k: Top-k sampling値
            verbose: 詳細表示
            
        Returns:
            生成されたテキスト
        """
        if verbose:
            print(f"\nGenerating text...")
            print(f"Prompt: '{prompt}'")
            print(f"Parameters: max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}")
        
        # プロンプトをトークン化
        input_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([input_tokens], device=self.device)
        
        if verbose:
            print(f"Input tokens: {len(input_tokens)} tokens")
            print(f"Tokenized: {input_tokens}")
        
        # 生成実行
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if start_time:
            start_time.record()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000  # seconds
        else:
            generation_time = None
        
        # デコード
        generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())
        
        if verbose:
            print(f"Generated: '{generated_text}'")
            if generation_time:
                tokens_per_sec = max_new_tokens / generation_time if generation_time > 0 else 0
                print(f"Generation time: {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
        
        return generated_text
    
    def test_basic_generation(self):
        """基本的な生成テスト"""
        print("\n" + "="*50)
        print("Basic Generation Test")
        print("="*50)
        
        test_prompts = [
            "Hello",
            "The quick brown fox",
            "AI is",
            "Python programming"
        ]
        
        for prompt in test_prompts:
            try:
                result = self.generate_text(
                    prompt, 
                    max_new_tokens=20,
                    temperature=0.8,
                    verbose=True
                )
                print(f"✓ Success: '{prompt}' -> '{result}'")
            except Exception as e:
                print(f"✗ Failed: '{prompt}' -> Error: {e}")
            print("-" * 30)
    
    def test_different_temperatures(self):
        """異なる温度での生成テスト"""
        print("\n" + "="*50)
        print("Temperature Test")
        print("="*50)
        
        prompt = "The future of artificial intelligence"
        temperatures = [0.1, 0.5, 0.8, 1.0, 1.2]
        
        for temp in temperatures:
            try:
                result = self.generate_text(
                    prompt,
                    max_new_tokens=25,
                    temperature=temp,
                    verbose=False
                )
                print(f"Temperature {temp}: '{result}'")
            except Exception as e:
                print(f"Temperature {temp}: Error: {e}")
    
    def test_long_generation(self):
        """長いテキスト生成テスト"""
        print("\n" + "="*50)
        print("Long Generation Test")
        print("="*50)
        
        prompt = "Once upon a time"
        
        try:
            result = self.generate_text(
                prompt,
                max_new_tokens=100,
                temperature=0.8,
                verbose=True
            )
            print(f"✓ Long generation successful")
            print(f"Generated length: {len(result)} characters")
        except Exception as e:
            print(f"✗ Long generation failed: {e}")
    
    def test_model_components(self):
        """モデル各コンポーネントのテスト"""
        print("\n" + "="*50)
        print("Model Components Test")
        print("="*50)
        
        try:
            # 基本的な入力データ
            batch_size, seq_len = 1, 10
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            print(f"Testing with input shape: {input_ids.shape}")
            
            # モデル全体のフォワードパス
            with torch.no_grad():
                logits, kv_caches = self.model(input_ids)
            
            print(f"✓ Model forward pass successful")
            print(f"Output logits shape: {logits.shape}")
            print(f"Expected shape: ({batch_size}, {seq_len}, {self.config.vocab_size})")
            
            # KVキャッシュのテスト
            if kv_caches is not None:
                print(f"✓ KV caches generated: {len(kv_caches)} layers")
                for i, cache in enumerate(kv_caches[:2]):  # 最初の2層だけ表示
                    if cache is not None:
                        k_cache, v_cache = cache
                        print(f"  Layer {i}: K={k_cache.shape}, V={v_cache.shape}")
            else:
                print("⚠ No KV caches generated")
            
            # 補助損失のテスト
            aux_loss = self.model.get_auxiliary_loss()
            print(f"✓ Auxiliary loss: {aux_loss.item():.6f}")
            
        except Exception as e:
            print(f"✗ Model components test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_memory_usage(self):
        """メモリ使用量テスト"""
        print("\n" + "="*50)
        print("Memory Usage Test")
        print("="*50)
        
        if self.device.type == 'cuda':
            # 初期メモリ
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(self.device) / 1e9
            print(f"Initial GPU memory: {initial_memory:.2f}GB")
            
            # 推論実行
            prompt = "Memory test prompt"
            result = self.generate_text(prompt, max_new_tokens=30, verbose=False)
            
            # 推論後メモリ
            final_memory = torch.cuda.memory_allocated(self.device) / 1e9
            print(f"After inference GPU memory: {final_memory:.2f}GB")
            print(f"Memory used for inference: {final_memory - initial_memory:.2f}GB")
            
            # 最大メモリ
            max_memory = torch.cuda.max_memory_allocated(self.device) / 1e9
            print(f"Peak GPU memory: {max_memory:.2f}GB")
            
        else:
            print("CPU mode - memory usage not tracked")
    
    def run_all_tests(self):
        """全てのテストを実行"""
        print("Starting GPT-OSS Compact Inference Tests...")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        
        try:
            # 基本テスト
            self.test_basic_generation()
            
            # 温度テスト
            self.test_different_temperatures()
            
            # 長い生成テスト
            self.test_long_generation()
            
            # コンポーネントテスト
            self.test_model_components()
            
            # メモリテスト
            self.test_memory_usage()
            
            print("\n" + "="*50)
            print("All Tests Completed!")
            print("="*50)
            
        except Exception as e:
            print(f"\nTest suite failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """メイン関数"""
    print("GPT-OSS Compact Inference Tester")
    print("="*50)
    
    # コマンドライン引数の処理
    import argparse
    parser = argparse.ArgumentParser(description="GPT-OSS Compact Inference Test")
    parser.add_argument("--model", default="gpt_oss_compact.pt", 
                       help="Path to the model file")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for inference")
    parser.add_argument("--prompt", default=None,
                       help="Single prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # モデルファイルの確認
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    try:
        # 推論を作成
        inference = GPTOSSInference(args.model, args.device)
        
        if args.prompt:
            # 単一プロンプトでの生成
            print(f"\nGenerating for single prompt...")
            result = inference.generate_text(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            print(f"\nFinal result: '{result}'")
        else:
            # 全テストを実行
            inference.run_all_tests()
            
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()