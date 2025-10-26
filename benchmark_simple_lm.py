"""
Benchmark Suite for Pruned SimpleAttentionLM
Measures: Latency, Sparsity, Quality, Memory, Throughput
"""

import torch
import torch.nn as nn
import time
import json
import numpy as np
from transformers import AutoTokenizer
import psutil
import os
from typing import Dict, List

# Import your model
from attention import SimpleAttentionLM
from pruner import Pruner


class SimpleAttentionBenchmark:
    """Benchmark unpruned vs pruned SimpleAttentionLM"""
    
    def __init__(self, device='auto'):
        """
        Initialize benchmark
        device: 'cuda', 'cpu', or 'auto'
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🔧 Using device: {self.device}")
        
        # Model config (match your train.py)
        self.config = {
            "vocab_size": 50257,
            "block_size": 4096,
            "n_layer": 6,
            "n_head": 8,
            "n_embd": 512,
            "ff_hidden": 512 * 4,
            "dropout": 0.1,
        }
        
        # Load tokenizer
        print("📦 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.baseline_model = None
        self.pruned_model = None
        
    def load_models(self, baseline_path: str, pruned_path: str = None):
        """Load baseline and optionally pruned model"""
        
        print(f"📦 Loading baseline model from {baseline_path}...")
        self.baseline_model = SimpleAttentionLM(**self.config).to(self.device)
        self.baseline_model.load_state_dict(torch.load(baseline_path, map_location=self.device))
        self.baseline_model.eval()
        
        if pruned_path and os.path.exists(pruned_path):
            print(f"📦 Loading pruned model from {pruned_path}...")
            self.pruned_model = SimpleAttentionLM(**self.config).to(self.device)
            self.pruned_model.load_state_dict(torch.load(pruned_path, map_location=self.device))
            self.pruned_model.eval()
        else:
            print("⚠️  No pruned model found - will only benchmark baseline")
            
    def create_pruned_model(self, sparsity: float = 0.5, pruning_method: str = 'magnitude'):
        """
        Create pruned version from baseline model
        sparsity: fraction to prune (0.5 = 50%)
        pruning_method: 'magnitude' or 'consensus'
        """
        print(f"🔪 Creating pruned model ({sparsity*100:.0f}% sparsity, method={pruning_method})...")
        
        # Clone the baseline model
        self.pruned_model = SimpleAttentionLM(**self.config).to(self.device)
        self.pruned_model.load_state_dict(self.baseline_model.state_dict())
        self.pruned_model.eval()
        
        # Apply pruning
        pruner = Pruner(self.pruned_model, device=self.device)
        
        if pruning_method == 'magnitude':
            pruner.prune_magnitude_global(sparsity)
        elif pruning_method == 'consensus':
            # Create dummy data for gradient scoring
            dummy_input = torch.randint(0, self.config['vocab_size'], (4, 128)).to(self.device)
            dummy_labels = torch.randint(0, self.config['vocab_size'], (4, 128)).to(self.device)
            loss_fn = nn.CrossEntropyLoss()
            
            pruner.prune_consensus(
                methods=['magnitude', 'gradient', 'random'],
                sparsity_per_method=sparsity,
                consensus_k=2,
                data_batch=(dummy_input, dummy_labels),
                loss_fn=loss_fn
            )
        
        print("✅ Pruned model created!")
        
    def count_parameters(self, model) -> Dict:
        """Count total and active (non-zero) parameters"""
        total = 0
        nonzero = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                total += param.numel()
                nonzero += (param.data != 0).sum().item()
        
        sparsity = (total - nonzero) / total * 100 if total > 0 else 0
        
        return {
            "total_params": total,
            "active_params": nonzero,
            "pruned_params": total - nonzero,
            "sparsity_pct": sparsity,
            "active_pct": 100 - sparsity
        }
    
    def measure_latency(self, model, prompts: List[str], max_new_tokens: int = 50) -> Dict:
        """Measure inference latency"""
        
        latencies = []
        
        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Clear cache
            if self.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Measure time
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=50
                )
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies)
        }
    
    def measure_memory(self, model) -> Dict:
        """Measure memory usage"""
        
        if self.device == 'cuda':
            # GPU memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Do a forward pass
            dummy_input = torch.randint(0, self.config['vocab_size'], (1, 100)).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            return {
                "allocated_mb": allocated,
                "peak_mb": peak,
                "device": "GPU"
            }
        else:
            # CPU memory
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)  # MB
            
            dummy_input = torch.randint(0, self.config['vocab_size'], (1, 100)).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            mem_after = process.memory_info().rss / (1024 ** 2)  # MB
            
            return {
                "allocated_mb": mem_after - mem_before,
                "peak_mb": mem_after,
                "device": "CPU"
            }
    
    def measure_throughput(self, model, num_tokens: int = 500) -> Dict:
        """Measure throughput (tokens/second)"""
        
        # Generate longer sequence for throughput measurement
        prompt = "Once upon a time"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=num_tokens,
                temperature=0.8,
                top_k=50
            )
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        total_time = end - start
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        throughput = tokens_generated / total_time
        
        return {
            "throughput_tokens_per_sec": throughput,
            "total_time_sec": total_time,
            "tokens_generated": tokens_generated
        }
    
    def measure_quality(self, model, prompts: List[str], max_new_tokens: int = 50) -> Dict:
        """Generate outputs for quality comparison"""
        
        outputs = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=50
                )
            
            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append(text)
        
        return {"generated_outputs": outputs}
    
    def compare_outputs(self, baseline_outputs: List[str], pruned_outputs: List[str]) -> Dict:
        """Compare quality between baseline and pruned outputs"""
        
        try:
            from torchmetrics.text.rouge import ROUGEScore
            rouge = ROUGEScore()
            
            rouge_scores = []
            for base, pruned in zip(baseline_outputs, pruned_outputs):
                score = rouge(pruned, base)
                rouge_scores.append({
                    'rouge1': score['rouge1_fmeasure'].item(),
                    'rouge2': score['rouge2_fmeasure'].item(),
                    'rougeL': score['rougeL_fmeasure'].item()
                })
            
            avg_rouge1 = np.mean([s['rouge1'] for s in rouge_scores])
            avg_rouge2 = np.mean([s['rouge2'] for s in rouge_scores])
            avg_rougeL = np.mean([s['rougeL'] for s in rouge_scores])
            
            return {
                "rouge1_score": avg_rouge1,
                "rouge2_score": avg_rouge2,
                "rougeL_score": avg_rougeL,
                "quality_retention_pct": avg_rougeL * 100
            }
        except ImportError:
            print("⚠️  torchmetrics not installed - skipping quality comparison")
            return {"quality_retention_pct": None}
    
    def run_full_benchmark(
        self, 
        test_prompts: List[str],
        output_file: str = "benchmark_results.json"
    ) -> Dict:
        """Run complete benchmark"""
        
        print("\n" + "="*60)
        print("🚀 STARTING BENCHMARK")
        print("="*60 + "\n")
        
        if self.baseline_model is None:
            raise ValueError("No models loaded! Call load_models() first.")
        
        results = {
            "device": self.device,
            "num_test_prompts": len(test_prompts),
            "baseline": {},
            "pruned": {},
            "improvements": {}
        }
        
        # --- Baseline Benchmarks ---
        print("📊 Benchmarking BASELINE model...")
        
        print("  • Counting parameters...")
        results["baseline"]["params"] = self.count_parameters(self.baseline_model)
        
        print("  • Measuring latency...")
        results["baseline"]["latency"] = self.measure_latency(self.baseline_model, test_prompts)
        
        print("  • Measuring memory...")
        results["baseline"]["memory"] = self.measure_memory(self.baseline_model)
        
        print("  • Measuring throughput...")
        results["baseline"]["throughput"] = self.measure_throughput(self.baseline_model)
        
        print("  • Generating outputs...")
        baseline_quality = self.measure_quality(self.baseline_model, test_prompts)
        results["baseline"]["sample_outputs"] = baseline_quality["generated_outputs"][:3]  # Save first 3
        
        # --- Pruned Benchmarks ---
        if self.pruned_model:
            print("\n📊 Benchmarking PRUNED model...")
            
            print("  • Counting parameters...")
            results["pruned"]["params"] = self.count_parameters(self.pruned_model)
            
            print("  • Measuring latency...")
            results["pruned"]["latency"] = self.measure_latency(self.pruned_model, test_prompts)
            
            print("  • Measuring memory...")
            results["pruned"]["memory"] = self.measure_memory(self.pruned_model)
            
            print("  • Measuring throughput...")
            results["pruned"]["throughput"] = self.measure_throughput(self.pruned_model)
            
            print("  • Generating outputs...")
            pruned_quality = self.measure_quality(self.pruned_model, test_prompts)
            results["pruned"]["sample_outputs"] = pruned_quality["generated_outputs"][:3]
            
            print("  • Comparing quality...")
            quality_comparison = self.compare_outputs(
                baseline_quality["generated_outputs"],
                pruned_quality["generated_outputs"]
            )
            results["pruned"]["quality"] = quality_comparison
            
            # --- Calculate Improvements ---
            print("\n📈 Calculating improvements...")
            
            baseline_lat = results["baseline"]["latency"]["avg_latency_ms"]
            pruned_lat = results["pruned"]["latency"]["avg_latency_ms"]
            speedup = baseline_lat / pruned_lat
            
            baseline_mem = results["baseline"]["memory"]["peak_mb"]
            pruned_mem = results["pruned"]["memory"]["peak_mb"]
            memory_reduction_pct = (baseline_mem - pruned_mem) / baseline_mem * 100
            
            baseline_throughput = results["baseline"]["throughput"]["throughput_tokens_per_sec"]
            pruned_throughput = results["pruned"]["throughput"]["throughput_tokens_per_sec"]
            throughput_improvement_pct = (pruned_throughput - baseline_throughput) / baseline_throughput * 100
            
            results["improvements"] = {
                "speedup_ratio": round(speedup, 2),
                "latency_reduction_ms": round(baseline_lat - pruned_lat, 2),
                "latency_reduction_pct": round((1 - pruned_lat / baseline_lat) * 100, 2),
                "memory_reduction_mb": round(baseline_mem - pruned_mem, 2),
                "memory_reduction_pct": round(memory_reduction_pct, 2),
                "throughput_improvement_pct": round(throughput_improvement_pct, 2),
                "sparsity_pct": results["pruned"]["params"]["sparsity_pct"],
                "neurons_active_pct": results["pruned"]["params"]["active_pct"],
                "quality_retention_pct": quality_comparison.get("quality_retention_pct", 0)
            }
        
        # Save results
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert)
        
        print(f"\n✅ Benchmark complete! Results saved to {output_file}")
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print nice summary"""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("="*60 + "\n")
        
        baseline = results["baseline"]
        print("BASELINE MODEL:")
        print(f"  • Parameters: {baseline['params']['total_params']:,}")
        print(f"  • Avg Latency: {baseline['latency']['avg_latency_ms']:.2f} ms")
        print(f"  • Memory: {baseline['memory']['peak_mb']:.2f} MB")
        print(f"  • Throughput: {baseline['throughput']['throughput_tokens_per_sec']:.2f} tokens/sec")
        
        if "pruned" in results and results["pruned"]:
            pruned = results["pruned"]
            improvements = results["improvements"]
            
            print("\nPRUNED MODEL:")
            print(f"  • Active Parameters: {pruned['params']['active_params']:,} ({pruned['params']['active_pct']:.1f}%)")
            print(f"  • Sparsity: {pruned['params']['sparsity_pct']:.1f}%")
            print(f"  • Avg Latency: {pruned['latency']['avg_latency_ms']:.2f} ms")
            print(f"  • Memory: {pruned['memory']['peak_mb']:.2f} MB")
            print(f"  • Throughput: {pruned['throughput']['throughput_tokens_per_sec']:.2f} tokens/sec")
            
            print("\n🎉 IMPROVEMENTS:")
            print(f"  • Speedup: {improvements['speedup_ratio']}x faster")
            print(f"  • Latency Reduction: {improvements['latency_reduction_pct']:.1f}%")
            print(f"  • Memory Reduction: {improvements['memory_reduction_pct']:.1f}%")
            print(f"  • Throughput Gain: {improvements['throughput_improvement_pct']:.1f}%")
            print(f"  • Neurons Active: {improvements['neurons_active_pct']:.1f}%")
            if improvements['quality_retention_pct']:
                print(f"  • Quality Retention: {improvements['quality_retention_pct']:.1f}%")
        
        print("\n" + "="*60 + "\n")


def get_test_prompts(num_prompts: int = 20) -> List[str]:
    """Get diverse test prompts"""
    
    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In the future, technology will",
        "The best way to learn programming is",
        "Artificial intelligence can help us",
        "Climate change is caused by",
        "The secret to happiness is",
        "To build a successful startup, you need",
        "The most important invention in history was",
        "If I could travel anywhere, I would",
        "Science has discovered that",
        "The key to good health is",
        "In a world without technology",
        "The greatest challenge facing humanity is",
        "My favorite book is about",
        "When I grow up, I want to",
        "The universe is so vast that",
        "Democracy works best when",
        "The internet has changed society by",
        "To solve world hunger, we should"
    ]
    
    return prompts[:num_prompts]


def main():
    """Main benchmark execution"""
    
    print("="*60)
    print("SimpleAttentionLM Pruning Benchmark")
    print("="*60 + "\n")
    
    # Initialize
    benchmark = SimpleAttentionBenchmark(device='auto')
    
    # Load baseline model
    baseline_path = "simple_lm_custom_dataset.pt"
    
    if not os.path.exists(baseline_path):
        print(f"❌ Error: Model file '{baseline_path}' not found!")
        print("Please train the model first using: python train.py")
        return
    
    # Check for pruned model
    pruned_path = "simple_lm_pruned.pt"
    
    if os.path.exists(pruned_path):
        print("Found both models - will benchmark both!")
        benchmark.load_models(baseline_path, pruned_path)
    else:
        print("Only baseline found - will create pruned version automatically")
        benchmark.load_models(baseline_path)
        
        # Create pruned model (50% sparsity with magnitude pruning)
        benchmark.create_pruned_model(sparsity=0.5, pruning_method='magnitude')
        
        # Save it for future use
        torch.save(benchmark.pruned_model.state_dict(), pruned_path)
        print(f"💾 Saved pruned model to {pruned_path}")
    
    # Get test prompts
    test_prompts = get_test_prompts(num_prompts=20)
    
    # Run benchmark
    results = benchmark.run_full_benchmark(
        test_prompts=test_prompts,
        output_file="benchmark_results.json"
    )
    
    print("\n🎊 Done! Run visualize_results.py to generate charts!")


if __name__ == "__main__":
    main()