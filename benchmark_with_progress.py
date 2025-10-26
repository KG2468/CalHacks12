"""
Enhanced Benchmark with Real-Time Progress Tracking
Captures metrics at each checkpoint and generates progress graphs
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
import matplotlib.pyplot as plt
from datetime import datetime

# Import your model
from attention import SimpleAttentionLM
from pruner import Pruner


class ProgressTracker:
    """Track metrics throughout benchmarking process"""
    
    def __init__(self):
        self.timeline = []
        self.checkpoints = {
            'latency': [],
            'memory': [],
            'throughput': [],
            'timestamps': []
        }
    
    def add_checkpoint(self, name: str, metric_type: str, value: float):
        """Add a checkpoint measurement"""
        timestamp = time.time()
        self.checkpoints[metric_type].append({
            'name': name,
            'value': value,
            'timestamp': timestamp
        })
        self.checkpoints['timestamps'].append(timestamp)
    
    def plot_progress(self, save_path: str = "benchmark_progress.png"):
        """Plot progress timeline"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('🔄 Benchmarking Progress Timeline', fontsize=16, fontweight='bold')
        
        # Normalize timestamps to start at 0
        if self.checkpoints['timestamps']:
            start_time = min(self.checkpoints['timestamps'])
            norm_times = [(t - start_time) for t in self.checkpoints['timestamps']]
        
        # Plot latency progress
        ax = axes[0]
        latency_data = self.checkpoints['latency']
        if latency_data:
            times = [(d['timestamp'] - start_time) for d in latency_data]
            values = [d['value'] for d in latency_data]
            names = [d['name'] for d in latency_data]
            
            ax.plot(times, values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            ax.fill_between(times, values, alpha=0.3, color='#2E86AB')
            
            # Add labels at each point
            for t, v, name in zip(times, values, names):
                ax.annotate(f'{v:.0f}ms', xy=(t, v), xytext=(0, 10),
                           textcoords='offset points', ha='center',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=11)
            ax.set_title('Inference Latency Progress', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_xlim(left=-0.5)
        
        # Plot memory progress
        ax = axes[1]
        memory_data = self.checkpoints['memory']
        if memory_data:
            times = [(d['timestamp'] - start_time) for d in memory_data]
            values = [d['value'] for d in memory_data]
            names = [d['name'] for d in memory_data]
            
            ax.plot(times, values, 's-', linewidth=2, markersize=8, color='#F18F01')
            ax.fill_between(times, values, alpha=0.3, color='#F18F01')
            
            for t, v, name in zip(times, values, names):
                ax.annotate(f'{v:.0f}MB', xy=(t, v), xytext=(0, 10),
                           textcoords='offset points', ha='center',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.set_ylabel('Memory (MB)', fontweight='bold', fontsize=11)
            ax.set_title('Memory Usage Progress', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_xlim(left=-0.5)
        
        # Plot throughput progress
        ax = axes[2]
        throughput_data = self.checkpoints['throughput']
        if throughput_data:
            times = [(d['timestamp'] - start_time) for d in throughput_data]
            values = [d['value'] for d in throughput_data]
            names = [d['name'] for d in throughput_data]
            
            ax.plot(times, values, '^-', linewidth=2, markersize=8, color='#06A77D')
            ax.fill_between(times, values, alpha=0.3, color='#06A77D')
            
            for t, v, name in zip(times, values, names):
                ax.annotate(f'{v:.0f} tok/s', xy=(t, v), xytext=(0, 10),
                           textcoords='offset points', ha='center',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.set_ylabel('Throughput (tokens/s)', fontweight='bold', fontsize=11)
            ax.set_title('Throughput Progress', fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_xlim(left=-0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved progress timeline to {save_path}")
        plt.close()


class EnhancedBenchmark:
    """Enhanced benchmark with progress tracking"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🔧 Using device: {self.device}")
        
        self.config = {
            "vocab_size": 50257,
            "block_size": 4096,
            "n_layer": 6,
            "n_head": 8,
            "n_embd": 512,
            "ff_hidden": 512 * 4,
            "dropout": 0.1,
        }
        
        print("📦 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.baseline_model = None
        self.pruned_model = None
        self.tracker = ProgressTracker()
        
    def load_models(self, baseline_path: str, pruned_path: str = None):
        """Load models"""
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
            print("⚠️  No pruned model found")
            
    def create_pruned_model(self, sparsity: float = 0.5, pruning_method: str = 'magnitude'):
        """Create pruned version"""
        print(f"🔪 Creating pruned model ({sparsity*100:.0f}% sparsity)...")
        
        self.pruned_model = SimpleAttentionLM(**self.config).to(self.device)
        self.pruned_model.load_state_dict(self.baseline_model.state_dict())
        self.pruned_model.eval()
        
        pruner = Pruner(self.pruned_model, device=self.device)
        
        if pruning_method == 'magnitude':
            pruner.prune_magnitude_global(sparsity)
        
        print("✅ Pruned model created!")
        
    def count_parameters(self, model) -> Dict:
        """Count parameters"""
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
    
    def measure_latency(self, model, prompts: List[str], model_name: str, max_new_tokens: int = 50) -> Dict:
        """Measure latency with progress tracking"""
        
        latencies = []
        
        print(f"  ⏱️  Testing {model_name} latency (prompt 0/{len(prompts)})", end='')
        
        for i, prompt in enumerate(prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
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
            
            # Track every 5 prompts
            if (i + 1) % 5 == 0 or (i + 1) == len(prompts):
                avg_so_far = np.mean(latencies)
                self.tracker.add_checkpoint(
                    f"{model_name} (n={i+1})",
                    'latency',
                    avg_so_far
                )
                print(f"\r  ⏱️  Testing {model_name} latency (prompt {i+1}/{len(prompts)}) - Avg: {avg_so_far:.1f}ms", end='')
        
        print()  # New line
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies)
        }
    
    def measure_memory(self, model, model_name: str) -> Dict:
        """Measure memory with tracking"""
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            dummy_input = torch.randint(0, self.config['vocab_size'], (1, 100)).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            self.tracker.add_checkpoint(model_name, 'memory', peak)
            
            return {
                "allocated_mb": allocated,
                "peak_mb": peak,
                "device": "GPU"
            }
        else:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            
            dummy_input = torch.randint(0, self.config['vocab_size'], (1, 100)).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            mem_after = process.memory_info().rss / (1024 ** 2)
            peak = mem_after
            
            self.tracker.add_checkpoint(model_name, 'memory', peak)
            
            return {
                "allocated_mb": mem_after - mem_before,
                "peak_mb": peak,
                "device": "CPU"
            }
    
    def measure_throughput(self, model, model_name: str, num_tokens: int = 500) -> Dict:
        """Measure throughput with tracking"""
        
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
        
        self.tracker.add_checkpoint(model_name, 'throughput', throughput)
        
        return {
            "throughput_tokens_per_sec": throughput,
            "total_time_sec": total_time,
            "tokens_generated": tokens_generated
        }
    
    def run_benchmark(self, test_prompts: List[str], output_file: str = "benchmark_results.json"):
        """Run benchmark with progress tracking"""
        
        print("\n" + "="*60)
        print("🚀 STARTING ENHANCED BENCHMARK WITH PROGRESS TRACKING")
        print("="*60 + "\n")
        
        if self.baseline_model is None:
            raise ValueError("No models loaded!")
        
        results = {
            "device": self.device,
            "num_test_prompts": len(test_prompts),
            "baseline": {},
            "pruned": {},
            "improvements": {}
        }
        
        # Baseline
        print("📊 Benchmarking BASELINE model...")
        results["baseline"]["params"] = self.count_parameters(self.baseline_model)
        results["baseline"]["latency"] = self.measure_latency(self.baseline_model, test_prompts, "Baseline")
        results["baseline"]["memory"] = self.measure_memory(self.baseline_model, "Baseline")
        results["baseline"]["throughput"] = self.measure_throughput(self.baseline_model, "Baseline")
        
        # Pruned
        if self.pruned_model:
            print("\n📊 Benchmarking PRUNED model...")
            results["pruned"]["params"] = self.count_parameters(self.pruned_model)
            results["pruned"]["latency"] = self.measure_latency(self.pruned_model, test_prompts, "Pruned")
            results["pruned"]["memory"] = self.measure_memory(self.pruned_model, "Pruned")
            results["pruned"]["throughput"] = self.measure_throughput(self.pruned_model, "Pruned")
            
            # Calculate improvements
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
                "quality_retention_pct": 0
            }
        
        # Save results
        with open(output_file, 'w') as f:
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
        
        # Generate progress visualization
        print("\n📈 Generating progress timeline...")
        self.tracker.plot_progress()
        
        return results


def get_test_prompts(num_prompts: int = 20) -> List[str]:
    """Get test prompts"""
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
    """Main execution"""
    print("="*60)
    print("Enhanced Benchmark with Progress Tracking")
    print("="*60 + "\n")
    
    benchmark = EnhancedBenchmark(device='auto')
    
    baseline_path = "simple_lm_custom_dataset.pt"
    
    if not os.path.exists(baseline_path):
        print(f"❌ Error: Model file '{baseline_path}' not found!")
        print("Please train the model first using: python train.py")
        return
    
    pruned_path = "simple_lm_pruned.pt"
    
    if os.path.exists(pruned_path):
        benchmark.load_models(baseline_path, pruned_path)
    else:
        benchmark.load_models(baseline_path)
        benchmark.create_pruned_model(sparsity=0.5, pruning_method='magnitude')
        torch.save(benchmark.pruned_model.state_dict(), pruned_path)
    
    test_prompts = get_test_prompts(num_prompts=20)
    
    results = benchmark.run_benchmark(
        test_prompts=test_prompts,
        output_file="benchmark_results.json"
    )
    
    print("\n🎊 Done! Check out benchmark_progress.png for the timeline!")
    print("Run visualize_results_pro.py to generate professional charts!")


if __name__ == "__main__":
    main()