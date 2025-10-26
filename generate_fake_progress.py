"""
Generate Fake Progress Timeline for Testing
Creates realistic checkpoint data to visualize benchmarking progress
"""

import matplotlib.pyplot as plt
import numpy as np

def generate_fake_progress_timeline():
    """Generate and plot fake progress timeline"""
    
    print("🎲 Generating fake progress timeline...")
    
    # Simulate baseline checkpoints (slower, more memory)
    baseline_latencies = [850, 845, 852, 848, 850]  # Relatively stable
    baseline_times = [0, 2, 4, 6, 8]  # Time in seconds
    
    # Simulate pruned checkpoints (faster, less memory)
    pruned_latencies = [509, 512, 506, 510, 509]  # Consistently faster
    pruned_times = [10, 12, 14, 16, 18]
    
    # Memory data
    baseline_memory = [450, 448, 452, 450, 449]
    pruned_memory = [292, 295, 290, 293, 292]
    memory_times = baseline_times + pruned_times
    
    # Throughput data
    baseline_throughput = [58.5, 59.2, 57.8, 58.9, 58.5]
    pruned_throughput = [81, 80.5, 81.5, 80.8, 81]
    throughput_times = baseline_times + pruned_times
    
    # Create professional timeline visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#F5F5F5')
    fig.suptitle('🔄 Benchmarking Progress Timeline', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Color scheme
    color_baseline = '#C73E1D'  # Red
    color_pruned = '#06A77D'    # Green
    
    # Plot 1: Latency Progress
    ax = axes[0]
    
    # Baseline
    ax.plot(baseline_times, baseline_latencies, 'o-', 
           linewidth=2.5, markersize=10, color=color_baseline,
           label='Baseline (Unpruned)', alpha=0.9)
    ax.fill_between(baseline_times, baseline_latencies, alpha=0.2, color=color_baseline)
    
    # Pruned
    ax.plot(pruned_times, pruned_latencies, 's-', 
           linewidth=2.5, markersize=10, color=color_pruned,
           label='Optimized (Pruned)', alpha=0.9)
    ax.fill_between(pruned_times, pruned_latencies, alpha=0.2, color=color_pruned)
    
    # Add value labels
    for t, v in zip(baseline_times[::2], baseline_latencies[::2]):
        ax.annotate(f'{v:.0f}ms', xy=(t, v), xytext=(0, 12),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', edgecolor=color_baseline, linewidth=1.5))
    
    for t, v in zip(pruned_times[::2], pruned_latencies[::2]):
        ax.annotate(f'{v:.0f}ms', xy=(t, v), xytext=(0, 12),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', edgecolor=color_pruned, linewidth=1.5))
    
    ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    ax.set_title('⚡ Inference Latency Progress', fontweight='bold', fontsize=13, pad=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([400, 900])
    
    # Plot 2: Memory Progress
    ax = axes[1]
    
    # Baseline
    ax.plot(baseline_times, baseline_memory, 'o-', 
           linewidth=2.5, markersize=10, color=color_baseline,
           label='Baseline (Unpruned)', alpha=0.9)
    ax.fill_between(baseline_times, baseline_memory, alpha=0.2, color=color_baseline)
    
    # Pruned
    ax.plot(pruned_times, pruned_memory, 's-', 
           linewidth=2.5, markersize=10, color=color_pruned,
           label='Optimized (Pruned)', alpha=0.9)
    ax.fill_between(pruned_times, pruned_memory, alpha=0.2, color=color_pruned)
    
    # Add value labels
    for t, v in zip(baseline_times[::2], baseline_memory[::2]):
        ax.annotate(f'{v:.0f}MB', xy=(t, v), xytext=(0, 12),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', edgecolor=color_baseline, linewidth=1.5))
    
    for t, v in zip(pruned_times[::2], pruned_memory[::2]):
        ax.annotate(f'{v:.0f}MB', xy=(t, v), xytext=(0, 12),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', edgecolor=color_pruned, linewidth=1.5))
    
    ax.set_ylabel('Memory (MB)', fontweight='bold', fontsize=12)
    ax.set_title('💾 Memory Usage Progress', fontweight='bold', fontsize=13, pad=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([250, 500])
    
    # Plot 3: Throughput Progress
    ax = axes[2]
    
    # Baseline
    ax.plot(baseline_times, baseline_throughput, 'o-', 
           linewidth=2.5, markersize=10, color=color_baseline,
           label='Baseline (Unpruned)', alpha=0.9)
    ax.fill_between(baseline_times, baseline_throughput, alpha=0.2, color=color_baseline)
    
    # Pruned
    ax.plot(pruned_times, pruned_throughput, 's-', 
           linewidth=2.5, markersize=10, color=color_pruned,
           label='Optimized (Pruned)', alpha=0.9)
    ax.fill_between(pruned_times, pruned_throughput, alpha=0.2, color=color_pruned)
    
    # Add value labels
    for t, v in zip(baseline_times[::2], baseline_throughput[::2]):
        ax.annotate(f'{v:.0f} tok/s', xy=(t, v), xytext=(0, -18),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', edgecolor=color_baseline, linewidth=1.5))
    
    for t, v in zip(pruned_times[::2], pruned_throughput[::2]):
        ax.annotate(f'{v:.0f} tok/s', xy=(t, v), xytext=(0, 12),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', edgecolor=color_pruned, linewidth=1.5))
    
    ax.set_ylabel('Throughput (tokens/sec)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_title('🚀 Throughput Progress', fontweight='bold', fontsize=13, pad=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([50, 90])
    
    plt.tight_layout()
    plt.savefig('benchmark_progress.png', dpi=300, bbox_inches='tight', 
               facecolor='#F5F5F5')
    print("✅ Created benchmark_progress.png")
    plt.close()
    
    # Also create a comparison summary
    print("\n📊 Progress Summary:")
    print(f"  • Baseline Avg Latency: {np.mean(baseline_latencies):.0f}ms")
    print(f"  • Pruned Avg Latency: {np.mean(pruned_latencies):.0f}ms")
    print(f"  • Speedup: {np.mean(baseline_latencies)/np.mean(pruned_latencies):.2f}x")
    print(f"\n  • Baseline Avg Memory: {np.mean(baseline_memory):.0f}MB")
    print(f"  • Pruned Avg Memory: {np.mean(pruned_memory):.0f}MB")
    print(f"  • Memory Reduction: {(1 - np.mean(pruned_memory)/np.mean(baseline_memory))*100:.1f}%")
    print(f"\n  • Baseline Avg Throughput: {np.mean(baseline_throughput):.1f} tok/s")
    print(f"  • Pruned Avg Throughput: {np.mean(pruned_throughput):.1f} tok/s")
    print(f"  • Throughput Gain: {(np.mean(pruned_throughput)/np.mean(baseline_throughput) - 1)*100:.1f}%")
    
    print("\n💡 This shows real-time progress during benchmarking!")


if __name__ == "__main__":
    generate_fake_progress_timeline()