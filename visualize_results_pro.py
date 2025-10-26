"""
Professional Dashboard-Style Visualizations
Grafana/Prometheus inspired design with polished styling
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color scheme (professional)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'accent': '#6A4C93',       # Purple
    'neutral': '#5E6472',      # Gray
    'bg_dark': '#1E1E1E',      # Dark background
    'bg_light': '#F5F5F5',     # Light background
}

def load_results(filepath: str = "benchmark_results.json") -> dict:
    """Load benchmark results"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_metric_card(ax, title, value, unit, color, improvement=None):
    """Create a professional metric card"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background card with rounded corners
    card = FancyBboxPatch(
        (0.5, 1), 9, 8,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor=color,
        facecolor='white',
        alpha=0.9
    )
    ax.add_patch(card)
    
    # Title
    ax.text(5, 8, title, 
           ha='center', va='top',
           fontsize=14, fontweight='bold',
           color=COLORS['neutral'])
    
    # Main value
    ax.text(5, 5.5, f"{value}",
           ha='center', va='center',
           fontsize=36, fontweight='bold',
           color=color)
    
    # Unit
    ax.text(5, 3.5, unit,
           ha='center', va='center',
           fontsize=12,
           color=COLORS['neutral'])
    
    # Improvement badge
    if improvement:
        badge_color = COLORS['success'] if improvement > 0 else COLORS['danger']
        symbol = '↑' if improvement > 0 else '↓'
        ax.text(5, 2, f"{symbol} {abs(improvement):.1f}%",
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               color=badge_color,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor=badge_color, 
                        alpha=0.2,
                        edgecolor=badge_color))


def plot_professional_dashboard(results: dict, save_path: str = "dashboard_overview.png"):
    """Create professional dashboard with metric cards"""
    
    if "pruned" not in results or not results["pruned"]:
        print("⚠️  No pruned model - skipping dashboard")
        return
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['bg_light'])
    
    # Title
    fig.suptitle('🚀 Model Performance Dashboard', 
                fontsize=24, fontweight='bold',
                y=0.98, color=COLORS['neutral'])
    
    # Create grid for metric cards
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3,
                         left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    improvements = results["improvements"]
    baseline = results["baseline"]
    pruned = results["pruned"]
    
    # Row 1: Key metrics
    ax1 = fig.add_subplot(gs[0, 0])
    create_metric_card(
        ax1, "Speedup", 
        f"{improvements['speedup_ratio']}x",
        "times faster",
        COLORS['success'],
        improvements['latency_reduction_pct']
    )
    
    ax2 = fig.add_subplot(gs[0, 1])
    create_metric_card(
        ax2, "Latency", 
        f"{pruned['latency']['avg_latency_ms']:.0f}",
        "milliseconds",
        COLORS['primary'],
        -improvements['latency_reduction_pct']
    )
    
    ax3 = fig.add_subplot(gs[0, 2])
    create_metric_card(
        ax3, "Memory", 
        f"{pruned['memory']['peak_mb']:.0f}",
        "MB",
        COLORS['warning'],
        -improvements['memory_reduction_pct']
    )
    
    ax4 = fig.add_subplot(gs[0, 3])
    create_metric_card(
        ax4, "Throughput", 
        f"{pruned['throughput']['throughput_tokens_per_sec']:.0f}",
        "tokens/sec",
        COLORS['accent'],
        improvements['throughput_improvement_pct']
    )
    
    # Row 2: Comparison bars
    ax5 = fig.add_subplot(gs[1, :2])
    plot_comparison_bars(ax5, results, 'latency')
    
    ax6 = fig.add_subplot(gs[1, 2:])
    plot_comparison_bars(ax6, results, 'memory')
    
    # Row 3: Sparsity and detailed metrics
    ax7 = fig.add_subplot(gs[2, :2])
    plot_sparsity_gauge(ax7, improvements['sparsity_pct'])
    
    ax8 = fig.add_subplot(gs[2, 2:])
    plot_metrics_table(ax8, results)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg_light'])
    print(f"✅ Saved {save_path}")
    plt.close()


def plot_comparison_bars(ax, results, metric_type):
    """Plot professional comparison bars with rounded edges"""
    
    baseline = results["baseline"]
    pruned = results["pruned"]
    
    if metric_type == 'latency':
        title = "Inference Latency"
        baseline_val = baseline['latency']['avg_latency_ms']
        pruned_val = pruned['latency']['avg_latency_ms']
        unit = "ms"
        color_base = COLORS['danger']
        color_opt = COLORS['success']
    else:  # memory
        title = "Memory Usage"
        baseline_val = baseline['memory']['peak_mb']
        pruned_val = pruned['memory']['peak_mb']
        unit = "MB"
        color_base = COLORS['warning']
        color_opt = COLORS['primary']
    
    categories = ['Baseline', 'Optimized']
    values = [baseline_val, pruned_val]
    colors = [color_base, color_opt]
    
    # Create bars with rounded tops
    bars = ax.barh(categories, values, color=colors, alpha=0.8, height=0.6)
    
    # Add rounded corners effect
    for bar, color in zip(bars, colors):
        bar.set_edgecolor(color)
        bar.set_linewidth(2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + max(values)*0.02, i, f'{val:.1f} {unit}',
               va='center', fontsize=12, fontweight='bold',
               color=colors[i])
    
    ax.set_xlabel(unit, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def plot_sparsity_gauge(ax, sparsity_pct):
    """Plot a gauge-style sparsity visualization"""
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 1.5)
    ax.axis('off')
    
    # Draw gauge arc
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Background arc
    ax.plot(r * np.cos(theta), r * np.sin(theta), 
           linewidth=20, color='lightgray', alpha=0.3,
           solid_capstyle='round')
    
    # Progress arc
    progress_theta = np.linspace(0, np.pi * (sparsity_pct / 100), 100)
    color = COLORS['success'] if sparsity_pct > 40 else COLORS['warning']
    ax.plot(r * np.cos(progress_theta), r * np.sin(progress_theta),
           linewidth=20, color=color, alpha=0.8,
           solid_capstyle='round')
    
    # Center text
    ax.text(0, 0.3, f"{sparsity_pct:.1f}%",
           ha='center', va='center',
           fontsize=42, fontweight='bold',
           color=color)
    
    ax.text(0, -0.1, "Parameters Pruned",
           ha='center', va='center',
           fontsize=13, color=COLORS['neutral'])
    
    # Markers
    for angle, label in [(0, '0%'), (np.pi/2, '50%'), (np.pi, '100%')]:
        x, y = r * np.cos(angle) * 1.15, r * np.sin(angle) * 1.15
        ax.text(x, y, label, ha='center', va='center',
               fontsize=10, color=COLORS['neutral'])
    
    ax.set_title("Model Sparsity", fontsize=14, fontweight='bold', pad=20)


def plot_metrics_table(ax, results):
    """Plot professional metrics table"""
    
    ax.axis('off')
    
    improvements = results["improvements"]
    
    # Data for table
    data = [
        ['Speedup', f"{improvements['speedup_ratio']}x", '🚀'],
        ['Latency ↓', f"{improvements['latency_reduction_pct']:.1f}%", '⚡'],
        ['Memory ↓', f"{improvements['memory_reduction_pct']:.1f}%", '💾'],
        ['Throughput ↑', f"{improvements['throughput_improvement_pct']:.1f}%", '📈'],
        ['Active Neurons', f"{improvements['neurons_active_pct']:.1f}%", '🧠'],
    ]
    
    if improvements.get('quality_retention_pct'):
        data.append(['Quality', f"{improvements['quality_retention_pct']:.1f}%", '✨'])
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=['Metric', 'Value', ''],
        cellLoc='left',
        loc='center',
        colWidths=[0.5, 0.3, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Style rows
    for i in range(1, len(data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor('white' if i % 2 == 0 else COLORS['bg_light'])
            cell.set_edgecolor(COLORS['neutral'])
            cell.set_linewidth(0.5)
            if j == 1:  # Value column
                cell.set_text_props(weight='bold', color=COLORS['success'])
            if j == 2:  # Emoji column
                cell.set_text_props(fontsize=14)
    
    ax.set_title("Performance Improvements", fontsize=14, fontweight='bold', pad=20)


def plot_detailed_comparison(results: dict, save_path: str = "detailed_comparison.png"):
    """Create detailed side-by-side comparison chart"""
    
    if "pruned" not in results or not results["pruned"]:
        print("⚠️  No pruned model - skipping detailed comparison")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(COLORS['bg_light'])
    fig.suptitle('📊 Detailed Performance Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    baseline = results["baseline"]
    pruned = results["pruned"]
    improvements = results["improvements"]
    
    # 1. Latency Distribution
    ax = axes[0, 0]
    models = ['Baseline', 'Optimized']
    latencies = [
        baseline['latency']['avg_latency_ms'],
        pruned['latency']['avg_latency_ms']
    ]
    bars = ax.bar(models, latencies, color=[COLORS['danger'], COLORS['success']], 
                  alpha=0.7, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}ms', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('Average Latency', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Memory Usage
    ax = axes[0, 1]
    memories = [
        baseline['memory']['peak_mb'],
        pruned['memory']['peak_mb']
    ]
    bars = ax.bar(models, memories, color=[COLORS['warning'], COLORS['primary']], 
                  alpha=0.7, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, memories):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}MB', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    ax.set_ylabel('Memory (MB)', fontweight='bold')
    ax.set_title('Peak Memory Usage', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Throughput
    ax = axes[0, 2]
    throughputs = [
        baseline['throughput']['throughput_tokens_per_sec'],
        pruned['throughput']['throughput_tokens_per_sec']
    ]
    bars = ax.bar(models, throughputs, color=[COLORS['accent'], COLORS['success']], 
                  alpha=0.7, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    ax.set_ylabel('Tokens/Second', fontweight='bold')
    ax.set_title('Throughput', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Parameter Count
    ax = axes[1, 0]
    param_data = [
        baseline['params']['total_params'],
        pruned['params']['active_params']
    ]
    colors_params = [COLORS['neutral'], COLORS['success']]
    bars = ax.bar(models, param_data, color=colors_params, 
                  alpha=0.7, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, param_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val/1e6:.1f}M', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    ax.set_ylabel('Parameters', fontweight='bold')
    ax.set_title('Active Parameters', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Improvement Percentages
    ax = axes[1, 1]
    metrics = ['Latency\nReduction', 'Memory\nSavings', 'Throughput\nGain']
    values = [
        improvements['latency_reduction_pct'],
        improvements['memory_reduction_pct'],
        improvements['throughput_improvement_pct']
    ]
    colors_imp = [COLORS['success'], COLORS['primary'], COLORS['accent']]
    bars = ax.barh(metrics, values, color=colors_imp, 
                   alpha=0.7, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
               f'{val:.1f}%', ha='left', va='center',
               fontsize=11, fontweight='bold')
    ax.set_xlabel('Improvement (%)', fontweight='bold')
    ax.set_title('Percentage Improvements', fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    
    # 6. Sparsity Breakdown
    ax = axes[1, 2]
    sparsity_data = [
        improvements['neurons_active_pct'],
        improvements['sparsity_pct']
    ]
    labels = ['Active', 'Pruned']
    colors_sparse = [COLORS['success'], COLORS['danger']]
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax.pie(
        sparsity_data, labels=labels, autopct='%1.1f%%',
        colors=colors_sparse, explode=explode,
        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax.set_title('Parameter Distribution', fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg_light'])
    print(f"✅ Saved {save_path}")
    plt.close()


def generate_professional_report(results: dict, save_path: str = "PROFESSIONAL_REPORT.md"):
    """Generate polished markdown report with tables"""
    
    report = f"""# 🚀 Model Optimization Performance Report

<div align="center">

## Executive Summary

**{results['improvements']['speedup_ratio']}x Faster** | **{results['improvements']['sparsity_pct']:.1f}% Pruned** | **{results['improvements']['neurons_active_pct']:.1f}% Active**

</div>

---

## 📊 Key Performance Indicators

<table>
<tr>
<td align="center"><b>⚡ Speedup</b><br/><h2>{results['improvements']['speedup_ratio']}x</h2></td>
<td align="center"><b>📉 Latency Reduction</b><br/><h2>{results['improvements']['latency_reduction_pct']:.1f}%</h2></td>
<td align="center"><b>💾 Memory Savings</b><br/><h2>{results['improvements']['memory_reduction_pct']:.1f}%</h2></td>
<td align="center"><b>📈 Throughput Gain</b><br/><h2>+{results['improvements']['throughput_improvement_pct']:.1f}%</h2></td>
</tr>
</table>

---

## 🎯 Detailed Metrics

### Baseline Model (Unpruned)
| Metric | Value |
|--------|-------|
| 📦 **Total Parameters** | {results['baseline']['params']['total_params']:,} |
| ⏱️ **Avg Latency** | {results['baseline']['latency']['avg_latency_ms']:.2f} ms |
| 💾 **Memory Usage** | {results['baseline']['memory']['peak_mb']:.2f} MB |
| 🚀 **Throughput** | {results['baseline']['throughput']['throughput_tokens_per_sec']:.2f} tokens/sec |

### Optimized Model (Pruned)
| Metric | Value |
|--------|-------|
| ✅ **Active Parameters** | {results['pruned']['params']['active_params']:,} ({results['pruned']['params']['active_pct']:.1f}%) |
| 🔪 **Pruned Parameters** | {results['pruned']['params']['pruned_params']:,} ({results['pruned']['params']['sparsity_pct']:.1f}%) |
| ⚡ **Avg Latency** | {results['pruned']['latency']['avg_latency_ms']:.2f} ms |
| 💾 **Memory Usage** | {results['pruned']['memory']['peak_mb']:.2f} MB |
| 🚀 **Throughput** | {results['pruned']['throughput']['throughput_tokens_per_sec']:.2f} tokens/sec |

---

## 📈 Performance Improvements

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Latency** | {results['baseline']['latency']['avg_latency_ms']:.1f} ms | {results['pruned']['latency']['avg_latency_ms']:.1f} ms | 🟢 **{results['improvements']['speedup_ratio']}x faster** ({results['improvements']['latency_reduction_pct']:.1f}% reduction) |
| **Memory** | {results['baseline']['memory']['peak_mb']:.1f} MB | {results['pruned']['memory']['peak_mb']:.1f} MB | 🟢 **{results['improvements']['memory_reduction_pct']:.1f}% less** |
| **Throughput** | {results['baseline']['throughput']['throughput_tokens_per_sec']:.1f} tok/s | {results['pruned']['throughput']['throughput_tokens_per_sec']:.1f} tok/s | 🟢 **+{results['improvements']['throughput_improvement_pct']:.1f}%** |
| **Parameters** | {results['baseline']['params']['total_params']:,} | {results['pruned']['params']['active_params']:,} | 🟢 **{results['improvements']['sparsity_pct']:.1f}% pruned** |

---

## 📊 Visual Dashboard

![Dashboard Overview](dashboard_overview.png)
![Detailed Comparison](detailed_comparison.png)

---

## 💡 Key Insights

- ✅ **Achieved {results['improvements']['speedup_ratio']}x speedup** through strategic pruning
- ✅ **Reduced inference latency by {results['improvements']['latency_reduction_pct']:.1f}%** from {results['baseline']['latency']['avg_latency_ms']:.1f}ms to {results['pruned']['latency']['avg_latency_ms']:.1f}ms
- ✅ **Decreased memory footprint by {results['improvements']['memory_reduction_pct']:.1f}%** making deployment more efficient
- ✅ **Improved throughput by {results['improvements']['throughput_improvement_pct']:.1f}%** enabling higher request rates
- ✅ **Pruned {results['improvements']['sparsity_pct']:.1f}% of parameters** while maintaining performance

---

## 🎯 Conclusion

Our consensus pruning approach successfully optimized the model by removing **{results['improvements']['sparsity_pct']:.1f}% of parameters** while achieving:
- **{results['improvements']['speedup_ratio']}x faster inference**
- **{results['improvements']['memory_reduction_pct']:.1f}% memory reduction**
- **{results['improvements']['throughput_improvement_pct']:.1f}% throughput improvement**

This demonstrates significant efficiency gains suitable for production deployment scenarios where inference speed and resource utilization are critical.

---

<div align="center">

**Device:** {results['device'].upper()} | **Test Prompts:** {results['num_test_prompts']}

*Generated using professional benchmarking suite*

</div>
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Saved {save_path}")


def main():
    """Generate all professional visualizations"""
    
    print("\n" + "="*60)
    print("🎨 GENERATING PROFESSIONAL VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load results
    if not Path("benchmark_results.json").exists():
        print("❌ Error: benchmark_results.json not found!")
        print("Please run benchmark_simple_lm.py first.")
        return
    
    results = load_results()
    
    # Generate professional visualizations
    print("Creating dashboard overview...")
    plot_professional_dashboard(results)
    
    print("Creating detailed comparison...")
    plot_detailed_comparison(results)
    
    print("Generating professional report...")
    generate_professional_report(results)
    
    print("\n" + "="*60)
    print("✅ ALL PROFESSIONAL VISUALIZATIONS GENERATED!")
    print("="*60)
    print("\nFiles created:")
    print("  • dashboard_overview.png - Professional dashboard")
    print("  • detailed_comparison.png - Detailed analysis charts")
    print("  • PROFESSIONAL_REPORT.md - Polished markdown report")
    print("\n💡 These look amazing for presentations!")


if __name__ == "__main__":
    main()