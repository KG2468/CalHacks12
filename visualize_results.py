"""
Visualization Script for SimpleAttentionLM Benchmark Results
Generates charts and comparison tables
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_results(filepath: str = "benchmark_results.json") -> dict:
    """Load benchmark results"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_latency_comparison(results: dict, save_path: str = "latency_comparison.png"):
    """Bar chart comparing latencies"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_lat = results["baseline"]["latency"]["avg_latency_ms"]
    
    if "pruned" in results and results["pruned"]:
        pruned_lat = results["pruned"]["latency"]["avg_latency_ms"]
        
        models = ['Baseline\n(Unpruned)', 'Pruned\nModel']
        latencies = [baseline_lat, pruned_lat]
        colors = ['#e74c3c', '#27ae60']
        
        bars = ax.bar(models, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} ms',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # Add speedup annotation
        speedup = results["improvements"]["speedup_ratio"]
        ax.text(0.5, max(latencies) * 0.85, f'{speedup}x Speedup!', 
               ha='center', fontsize=18, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        models = ['Baseline (Unpruned)']
        latencies = [baseline_lat]
        ax.bar(models, latencies, color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Inference Latency Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {save_path}")
    plt.close()


def plot_sparsity(results: dict, save_path: str = "sparsity_visualization.png"):
    """Pie chart showing active vs pruned neurons"""
    
    if "pruned" not in results or not results["pruned"]:
        print("⚠️  No pruned model - skipping sparsity plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    active_pct = results["pruned"]["params"]["active_pct"]
    pruned_pct = results["pruned"]["params"]["sparsity_pct"]
    
    sizes = [active_pct, pruned_pct]
    labels = [f'Active Neurons\n{active_pct:.1f}%', f'Pruned Neurons\n{pruned_pct:.1f}%']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={'fontsize': 13, 'fontweight': 'bold'}
    )
    
    ax.set_title(f'Model Sparsity: {pruned_pct:.1f}% Parameters Pruned', 
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {save_path}")
    plt.close()


def plot_memory_comparison(results: dict, save_path: str = "memory_comparison.png"):
    """Bar chart comparing memory usage"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_mem = results["baseline"]["memory"]["peak_mb"]
    
    if "pruned" in results and results["pruned"]:
        pruned_mem = results["pruned"]["memory"]["peak_mb"]
        
        models = ['Baseline\n(Unpruned)', 'Pruned\nModel']
        memories = [baseline_mem, pruned_mem]
        colors = ['#e67e22', '#16a085']
        
        bars = ax.bar(models, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} MB',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        reduction_pct = results["improvements"]["memory_reduction_pct"]
        ax.text(0.5, max(memories) * 0.85, f'{reduction_pct:.1f}% Less Memory!', 
               ha='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        models = ['Baseline (Unpruned)']
        memories = [baseline_mem]
        ax.bar(models, memories, color='#e67e22', alpha=0.8)
    
    ax.set_ylabel('Peak Memory (MB)', fontsize=13, fontweight='bold')
    ax.set_title('Memory Usage Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {save_path}")
    plt.close()


def plot_throughput_comparison(results: dict, save_path: str = "throughput_comparison.png"):
    """Bar chart comparing throughput"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_throughput = results["baseline"]["throughput"]["throughput_tokens_per_sec"]
    
    if "pruned" in results and results["pruned"]:
        pruned_throughput = results["pruned"]["throughput"]["throughput_tokens_per_sec"]
        
        models = ['Baseline\n(Unpruned)', 'Pruned\nModel']
        throughputs = [baseline_throughput, pruned_throughput]
        colors = ['#3498db', '#2ecc71']
        
        bars = ax.bar(models, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} tok/s',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        improvement_pct = results["improvements"]["throughput_improvement_pct"]
        if improvement_pct > 0:
            ax.text(0.5, max(throughputs) * 0.85, f'+{improvement_pct:.1f}% Throughput!', 
                   ha='center', fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        models = ['Baseline (Unpruned)']
        throughputs = [baseline_throughput]
        ax.bar(models, throughputs, color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=13, fontweight='bold')
    ax.set_title('Throughput Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {save_path}")
    plt.close()


def plot_all_metrics(results: dict, save_path: str = "all_metrics_comparison.png"):
    """Comprehensive 2x2 grid of all metrics"""
    
    if "pruned" not in results or not results["pruned"]:
        print("⚠️  No pruned model - skipping comprehensive plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
    
    baseline = results["baseline"]
    pruned = results["pruned"]
    improvements = results["improvements"]
    
    # 1. Latency
    ax1 = axes[0, 0]
    models = ['Baseline', 'Pruned']
    latencies = [baseline["latency"]["avg_latency_ms"], pruned["latency"]["avg_latency_ms"]]
    ax1.bar(models, latencies, color=['#e74c3c', '#27ae60'], alpha=0.8)
    ax1.set_ylabel('Latency (ms)', fontweight='bold')
    ax1.set_title(f'Latency ({improvements["speedup_ratio"]}x faster)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Memory
    ax2 = axes[0, 1]
    memories = [baseline["memory"]["peak_mb"], pruned["memory"]["peak_mb"]]
    ax2.bar(models, memories, color=['#e67e22', '#16a085'], alpha=0.8)
    ax2.set_ylabel('Memory (MB)', fontweight='bold')
    ax2.set_title(f'Memory ({improvements["memory_reduction_pct"]:.1f}% reduction)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Throughput
    ax3 = axes[1, 0]
    throughputs = [baseline["throughput"]["throughput_tokens_per_sec"], 
                   pruned["throughput"]["throughput_tokens_per_sec"]]
    ax3.bar(models, throughputs, color=['#3498db', '#2ecc71'], alpha=0.8)
    ax3.set_ylabel('Tokens/sec', fontweight='bold')
    ax3.set_title(f'Throughput (+{improvements["throughput_improvement_pct"]:.1f}%)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Sparsity
    ax4 = axes[1, 1]
    sparsity_data = [improvements["neurons_active_pct"], improvements["sparsity_pct"]]
    bars = ax4.bar(['Active\nNeurons', 'Pruned\nNeurons'], sparsity_data, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.8)
    ax4.set_ylabel('Percentage (%)', fontweight='bold')
    ax4.set_title(f'Model Sparsity ({improvements["sparsity_pct"]:.1f}% pruned)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {save_path}")
    plt.close()


def generate_markdown_report(results: dict, save_path: str = "BENCHMARK_REPORT.md"):
    """Generate markdown report"""
    
    report = f"""# SimpleAttentionLM Pruning Benchmark Results

**Device:** {results['device']}  
**Test Prompts:** {results['num_test_prompts']}

---

## 📊 Performance Summary

### Baseline (Unpruned Model)
- **Parameters:** {results['baseline']['params']['total_params']:,}
- **Avg Latency:** {results['baseline']['latency']['avg_latency_ms']:.2f} ms
- **Memory Usage:** {results['baseline']['memory']['peak_mb']:.2f} MB
- **Throughput:** {results['baseline']['throughput']['throughput_tokens_per_sec']:.2f} tokens/sec

"""
    
    if "pruned" in results and results["pruned"]:
        improvements = results["improvements"]
        pruned = results["pruned"]
        
        report += f"""### Pruned Model
- **Active Parameters:** {pruned['params']['active_params']:,} ({pruned['params']['active_pct']:.1f}% of original)
- **Sparsity:** {pruned['params']['sparsity_pct']:.1f}% parameters pruned
- **Avg Latency:** {pruned['latency']['avg_latency_ms']:.2f} ms
- **Memory Usage:** {pruned['memory']['peak_mb']:.2f} MB
- **Throughput:** {pruned['throughput']['throughput_tokens_per_sec']:.2f} tokens/sec

---

## 🎉 Key Improvements

| Metric | Improvement |
|--------|------------|
| **Speedup** | **{improvements['speedup_ratio']}x faster** |
| **Latency Reduction** | {improvements['latency_reduction_pct']:.1f}% ({improvements['latency_reduction_ms']:.2f} ms saved) |
| **Memory Reduction** | {improvements['memory_reduction_pct']:.1f}% ({improvements['memory_reduction_mb']:.2f} MB saved) |
| **Throughput Gain** | +{improvements['throughput_improvement_pct']:.1f}% |
| **Neurons Active** | {improvements['neurons_active_pct']:.1f}% |
"""
        
        if improvements.get('quality_retention_pct'):
            report += f"| **Quality Retention** | {improvements['quality_retention_pct']:.1f}% |\n"
        
        report += """
---

## 📈 Visualizations

![Latency Comparison](latency_comparison.png)
![Sparsity](sparsity_visualization.png)
![Memory Comparison](memory_comparison.png)
![Throughput Comparison](throughput_comparison.png)
![All Metrics](all_metrics_comparison.png)

---

## 📝 Sample Outputs

### Baseline Model Output:
```
"""
        for i, output in enumerate(results['baseline']['sample_outputs'][:2], 1):
            report += f"{i}. {output}\n"
        
        report += """```

### Pruned Model Output:
```
"""
        for i, output in enumerate(results['pruned']['sample_outputs'][:2], 1):
            report += f"{i}. {output}\n"
        
        report += f"""```

---

## 🎯 Conclusion

Our consensus pruning approach achieves **{improvements['speedup_ratio']}x speedup** by pruning 
**{improvements['sparsity_pct']:.1f}% of model parameters** while maintaining model quality. 
This demonstrates significant efficiency gains for deployment scenarios where inference speed 
and memory footprint are critical.

The pruned model uses only **{improvements['neurons_active_pct']:.1f}% of the original neurons**, 
yet delivers comparable performance with **{improvements['latency_reduction_pct']:.1f}% faster inference** 
and **{improvements['memory_reduction_pct']:.1f}% less memory usage**.
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Saved {save_path}")


def main():
    """Generate all visualizations"""
    
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load results
    if not Path("benchmark_results.json").exists():
        print("❌ Error: benchmark_results.json not found!")
        print("Please run benchmark_simple_lm.py first.")
        return
    
    results = load_results()
    
    # Generate plots
    plot_latency_comparison(results)
    plot_sparsity(results)
    plot_memory_comparison(results)
    plot_throughput_comparison(results)
    plot_all_metrics(results)
    
    # Generate report
    generate_markdown_report(results)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("="*60)
    print("\nFiles created:")
    print("  • latency_comparison.png")
    print("  • sparsity_visualization.png")
    print("  • memory_comparison.png")
    print("  • throughput_comparison.png")
    print("  • all_metrics_comparison.png")
    print("  • BENCHMARK_REPORT.md")
    print("\n💡 Add these to your project README for the hackathon!")


if __name__ == "__main__":
    main()