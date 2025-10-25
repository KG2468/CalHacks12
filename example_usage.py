"""
Example usage of the model pruning and compression library.

This script demonstrates various pruning and compression techniques
on a simple neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model_pruning import (
    ModelPruner,
    ModelCompressor,
    iterative_pruning,
    compare_pruning_methods
)


class ExampleNet(nn.Module):
    """Example neural network for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class ExampleCNN(nn.Module):
    """Example CNN for demonstration."""
    
    def __init__(self):
        super(ExampleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def print_separator(title=""):
    """Print a formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)


def demo_magnitude_pruning():
    """Demonstrate magnitude-based pruning."""
    print_separator("Magnitude-Based Pruning Demo")
    
    model = ExampleNet()
    pruner = ModelPruner(model)
    
    # Get initial statistics
    print("\nInitial Model Statistics:")
    initial_summary = pruner.get_sparsity_summary()
    print(f"  Total parameters: {initial_summary['total_params']:,}")
    print(f"  Active parameters: {initial_summary['active_params']:,}")
    print(f"  Overall sparsity: {initial_summary['overall_sparsity']:.2%}")
    
    # Apply magnitude pruning
    print("\nApplying 30% magnitude-based pruning...")
    stats = pruner.magnitude_pruning(amount=0.3)
    
    print("\nPer-layer statistics:")
    for layer_name, layer_stats in stats.items():
        print(f"  {layer_name}:")
        print(f"    Sparsity: {layer_stats['sparsity']:.2%}")
        print(f"    Pruned: {layer_stats['pruned_params']:,} / {layer_stats['total_params']:,}")
    
    # Get final statistics
    print("\nFinal Model Statistics:")
    final_summary = pruner.get_sparsity_summary()
    print(f"  Total parameters: {final_summary['total_params']:,}")
    print(f"  Active parameters: {final_summary['active_params']:,}")
    print(f"  Overall sparsity: {final_summary['overall_sparsity']:.2%}")
    print(f"  Compression ratio: {1 / (1 - final_summary['overall_sparsity']):.2f}x")


def demo_structured_pruning():
    """Demonstrate structured pruning."""
    print_separator("Structured Pruning Demo")
    
    model = ExampleCNN()
    pruner = ModelPruner(model)
    
    print("\nInitial Model Statistics:")
    initial_summary = pruner.get_sparsity_summary()
    print(f"  Total parameters: {initial_summary['total_params']:,}")
    
    # Apply structured pruning
    print("\nApplying 25% structured pruning (output channels)...")
    stats = pruner.structured_pruning(amount=0.25, dim=0)
    
    print("\nPer-layer statistics:")
    for layer_name, layer_stats in stats.items():
        print(f"  {layer_name}:")
        print(f"    Sparsity: {layer_stats['sparsity']:.2%}")
        print(f"    Dimension: {layer_stats['dim']}")
    
    final_summary = pruner.get_sparsity_summary()
    print(f"\nFinal overall sparsity: {final_summary['overall_sparsity']:.2%}")


def demo_global_pruning():
    """Demonstrate global magnitude pruning."""
    print_separator("Global Magnitude Pruning Demo")
    
    model = ExampleNet()
    pruner = ModelPruner(model)
    
    print("\nInitial Model Statistics:")
    initial_summary = pruner.get_sparsity_summary()
    print(f"  Total parameters: {initial_summary['total_params']:,}")
    
    # Apply global pruning
    print("\nApplying 40% global magnitude pruning...")
    stats = pruner.global_magnitude_pruning(amount=0.4)
    
    print("\nPer-layer statistics:")
    for layer_name, layer_stats in stats.items():
        print(f"  {layer_name}: {layer_stats['sparsity']:.2%} sparsity")
    
    final_summary = pruner.get_sparsity_summary()
    print(f"\nFinal overall sparsity: {final_summary['overall_sparsity']:.2%}")
    print(f"Note: Global pruning distributes pruning across all layers based on magnitude")


def demo_iterative_pruning():
    """Demonstrate iterative pruning."""
    print_separator("Iterative Pruning Demo")
    
    model = ExampleNet()
    
    # Define pruning schedule
    schedule = [0.1, 0.15, 0.2]
    print(f"\nPruning schedule: {schedule}")
    
    print("\nApplying iterative pruning...")
    pruned_model, stats = iterative_pruning(
        model,
        pruning_schedule=schedule,
        pruning_method='magnitude'
    )
    
    print("\nIteration results:")
    for iteration_stats in stats:
        iteration = iteration_stats['iteration']
        amount = iteration_stats['amount']
        summary = iteration_stats['summary']
        print(f"\n  Iteration {iteration + 1} (amount={amount}):")
        print(f"    Overall sparsity: {summary['overall_sparsity']:.2%}")
        print(f"    Active parameters: {summary['active_params']:,}")


def demo_comparison():
    """Compare different pruning methods."""
    print_separator("Pruning Methods Comparison")
    
    def model_factory():
        return ExampleNet()
    
    methods = ['magnitude', 'global', 'random']
    amount = 0.3
    
    print(f"\nComparing methods with {amount:.0%} pruning:")
    
    results = compare_pruning_methods(
        model_factory,
        methods=methods,
        amount=amount
    )
    
    print("\nComparison results:")
    for method, result in results.items():
        summary = result['overall_summary']
        print(f"\n  {method.upper()}:")
        print(f"    Overall sparsity: {summary['overall_sparsity']:.2%}")
        print(f"    Active parameters: {summary['active_params']:,}")
        print(f"    Zero parameters: {summary['total_zero_params']:,}")


def demo_quantization():
    """Demonstrate model quantization."""
    print_separator("Model Quantization Demo")
    
    model = ExampleNet()
    compressor = ModelCompressor(model)
    
    # Get original model size
    print("\nOriginal model:")
    original_size = compressor.get_model_size()
    print(f"  Size: {original_size['total_size_mb']:.2f} MB")
    print(f"  Parameters size: {original_size['param_size_mb']:.2f} MB")
    
    # Apply quantization
    print("\nApplying dynamic quantization...")
    quantized_model = compressor.dynamic_quantization()
    
    # Get quantized model size
    quantized_size = compressor.get_model_size(quantized_model)
    print("\nQuantized model:")
    print(f"  Size: {quantized_size['total_size_mb']:.2f} MB")
    print(f"  Parameters size: {quantized_size['param_size_mb']:.2f} MB")
    
    # Calculate compression ratio
    if quantized_size['total_size_mb'] > 0:
        compression_ratio = original_size['total_size_mb'] / quantized_size['total_size_mb']
        print(f"\nCompression ratio: {compression_ratio:.2f}x")
    else:
        print("\nNote: Quantization size calculation varies by PyTorch version")
        print("Quantized models typically achieve 2-4x compression")


def demo_combined_pruning_and_quantization():
    """Demonstrate combining pruning and quantization."""
    print_separator("Combined Pruning + Quantization Demo")
    
    model = ExampleNet()
    
    # Get original size
    compressor = ModelCompressor(model)
    original_size = compressor.get_model_size()
    print(f"\nOriginal model size: {original_size['total_size_mb']:.2f} MB")
    
    # Apply pruning
    print("\nApplying 50% magnitude pruning...")
    pruner = ModelPruner(model)
    pruner.magnitude_pruning(amount=0.5)
    pruner.make_pruning_permanent()
    
    pruned_summary = pruner.get_sparsity_summary()
    print(f"After pruning - Sparsity: {pruned_summary['overall_sparsity']:.2%}")
    
    # Apply quantization
    print("\nApplying quantization...")
    quantized_model = compressor.dynamic_quantization()
    
    # Get final size
    final_size = compressor.get_model_size(quantized_model)
    print(f"\nFinal model size: {final_size['total_size_mb']:.2f} MB")
    
    # Calculate total compression
    if final_size['total_size_mb'] > 0:
        total_compression = original_size['total_size_mb'] / final_size['total_size_mb']
        print(f"Total compression ratio: {total_compression:.2f}x")
    else:
        # Estimate based on pruning alone
        effective_compression = 1 / (1 - pruned_summary['overall_sparsity'])
        print(f"Estimated compression from pruning: {effective_compression:.2f}x")
        print("Note: Quantization would provide additional 2-4x compression")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" PyTorch Model Pruning and Compression Demo")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run demonstrations
    demo_magnitude_pruning()
    demo_structured_pruning()
    demo_global_pruning()
    demo_iterative_pruning()
    demo_comparison()
    demo_quantization()
    demo_combined_pruning_and_quantization()
    
    print("\n" + "=" * 80)
    print(" Demo Complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
