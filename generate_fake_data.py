"""
Generate Fake Benchmark Data for Testing Visualizations
Creates realistic benchmark_results.json with plausible metrics
"""

import json

def generate_fake_benchmark_data():
    """Generate realistic fake benchmark data"""
    
    # Baseline model (unpruned) - slower, more memory
    baseline_latency = 850  # ms
    baseline_memory = 450   # MB
    baseline_throughput = 58.5  # tokens/sec
    baseline_params = 15_500_000  # ~15.5M parameters
    
    # Pruned model (optimized) - 50% pruned, 1.67x speedup
    pruned_latency = baseline_latency / 1.67  # ~509ms
    pruned_memory = baseline_memory * 0.65    # 35% reduction
    pruned_throughput = baseline_throughput * 1.38  # 38% improvement
    pruned_params = int(baseline_params * 0.50)  # 50% active
    
    fake_results = {
        "device": "cpu",
        "num_test_prompts": 20,
        "baseline": {
            "params": {
                "total_params": baseline_params,
                "active_params": baseline_params,
                "pruned_params": 0,
                "sparsity_pct": 0.0,
                "active_pct": 100.0
            },
            "latency": {
                "avg_latency_ms": baseline_latency,
                "std_latency_ms": 45.2,
                "min_latency_ms": baseline_latency * 0.9,
                "max_latency_ms": baseline_latency * 1.2
            },
            "memory": {
                "allocated_mb": baseline_memory * 0.85,
                "peak_mb": baseline_memory,
                "device": "CPU"
            },
            "throughput": {
                "throughput_tokens_per_sec": baseline_throughput,
                "total_time_sec": 8.5,
                "tokens_generated": 500
            },
            "sample_outputs": [
                "Once upon a time there was a brave knight who traveled across distant lands.",
                "The meaning of life is to find purpose through connection, growth, and contribution to others.",
                "In the future, technology will enable humans to solve complex problems more efficiently."
            ]
        },
        "pruned": {
            "params": {
                "total_params": baseline_params,
                "active_params": pruned_params,
                "pruned_params": baseline_params - pruned_params,
                "sparsity_pct": 50.0,
                "active_pct": 50.0
            },
            "latency": {
                "avg_latency_ms": pruned_latency,
                "std_latency_ms": 28.5,
                "min_latency_ms": pruned_latency * 0.92,
                "max_latency_ms": pruned_latency * 1.15
            },
            "memory": {
                "allocated_mb": pruned_memory * 0.88,
                "peak_mb": pruned_memory,
                "device": "CPU"
            },
            "throughput": {
                "throughput_tokens_per_sec": pruned_throughput,
                "total_time_sec": 6.2,
                "tokens_generated": 500
            },
            "sample_outputs": [
                "Once upon a time there was a brave knight who journeyed through mysterious lands.",
                "The meaning of life is discovering purpose through meaningful connections and personal growth.",
                "In the future, technology will help humanity address challenges with greater efficiency."
            ],
            "quality": {
                "rouge1_score": 0.92,
                "rouge2_score": 0.88,
                "rougeL_score": 0.90,
                "quality_retention_pct": 90.0
            }
        },
        "improvements": {
            "speedup_ratio": round(baseline_latency / pruned_latency, 2),
            "latency_reduction_ms": round(baseline_latency - pruned_latency, 2),
            "latency_reduction_pct": round((1 - pruned_latency / baseline_latency) * 100, 2),
            "memory_reduction_mb": round(baseline_memory - pruned_memory, 2),
            "memory_reduction_pct": round((baseline_memory - pruned_memory) / baseline_memory * 100, 2),
            "throughput_improvement_pct": round((pruned_throughput - baseline_throughput) / baseline_throughput * 100, 2),
            "sparsity_pct": 50.0,
            "neurons_active_pct": 50.0,
            "quality_retention_pct": 90.0
        }
    }
    
    return fake_results


def main():
    """Generate and save fake benchmark data"""
    
    print("🎲 Generating fake benchmark data...")
    
    fake_data = generate_fake_benchmark_data()
    
    # Save to JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(fake_data, f, indent=2)
    
    print("✅ Created benchmark_results.json with fake data!")
    print("\nFake Results Summary:")
    print(f"  • Speedup: {fake_data['improvements']['speedup_ratio']}x")
    print(f"  • Latency Reduction: {fake_data['improvements']['latency_reduction_pct']:.1f}%")
    print(f"  • Memory Reduction: {fake_data['improvements']['memory_reduction_pct']:.1f}%")
    print(f"  • Throughput Gain: +{fake_data['improvements']['throughput_improvement_pct']:.1f}%")
    print(f"  • Sparsity: {fake_data['improvements']['sparsity_pct']:.1f}%")
    print("\n📊 Now run: python visualize_results_pro.py")
    print("   (or python visualize_results.py for basic version)")


if __name__ == "__main__":
    main()