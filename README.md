# CalHacks12 - PyTorch Model Pruning & Compression

A comprehensive library for neural network pruning and compression in PyTorch. This library extends PyTorch's built-in pruning capabilities with additional features and utilities for model optimization.

## Features

### Pruning Techniques
- **Magnitude-based Pruning**: Remove weights with smallest absolute values
- **Structured Pruning**: Remove entire channels/filters for actual speedup
- **Global Pruning**: Prune across all layers based on global magnitude threshold
- **Gradient-based Pruning**: Prune weights with smallest gradient magnitudes
- **Random Pruning**: Baseline pruning method for comparison
- **Iterative Pruning**: Apply pruning gradually with a schedule

### Compression Techniques
- **Dynamic Quantization**: Reduce model size by quantizing weights
- **Model Size Analysis**: Measure and compare model sizes

### Utilities
- **State Management**: Save and restore model states
- **Pruning History**: Track all pruning operations
- **Sparsity Analysis**: Detailed statistics on model sparsity
- **Method Comparison**: Compare different pruning methods side-by-side

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Magnitude Pruning

```python
import torch
import torch.nn as nn
from model_pruning import ModelPruner

# Create your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Initialize pruner
pruner = ModelPruner(model)

# Apply 30% magnitude-based pruning
stats = pruner.magnitude_pruning(amount=0.3)

# Get sparsity summary
summary = pruner.get_sparsity_summary()
print(f"Overall sparsity: {summary['overall_sparsity']:.2%}")
```

### Structured Pruning for CNNs

```python
from model_pruning import ModelPruner

# CNN model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),
    nn.ReLU()
)

pruner = ModelPruner(model)

# Prune 25% of output channels
stats = pruner.structured_pruning(amount=0.25, dim=0)
```

### Iterative Pruning

```python
from model_pruning import iterative_pruning

# Define pruning schedule
schedule = [0.1, 0.2, 0.3, 0.4]

# Apply iterative pruning
pruned_model, stats = iterative_pruning(
    model,
    pruning_schedule=schedule,
    pruning_method='magnitude'
)
```

### Model Quantization

```python
from model_pruning import ModelCompressor

compressor = ModelCompressor(model)

# Get original size
original_size = compressor.get_model_size()
print(f"Original: {original_size['total_size_mb']:.2f} MB")

# Apply quantization
quantized_model = compressor.dynamic_quantization()

# Get quantized size
quantized_size = compressor.get_model_size(quantized_model)
print(f"Quantized: {quantized_size['total_size_mb']:.2f} MB")
```

### Combine Pruning and Quantization

```python
from model_pruning import ModelPruner, ModelCompressor

# Prune
pruner = ModelPruner(model)
pruner.magnitude_pruning(amount=0.5)
pruner.make_pruning_permanent()

# Quantize
compressor = ModelCompressor(model)
final_model = compressor.dynamic_quantization()

# Get compression ratio
original_size = compressor.get_model_size(original_model)
final_size = compressor.get_model_size(final_model)
ratio = original_size['total_size_mb'] / final_size['total_size_mb']
print(f"Total compression: {ratio:.2f}x")
```

## API Reference

### ModelPruner

Main class for applying various pruning techniques.

#### Methods

- `magnitude_pruning(amount, layer_types, name_filter)`: Apply L1 magnitude-based pruning
- `structured_pruning(amount, dim, layer_types)`: Apply structured pruning to channels/filters
- `global_magnitude_pruning(amount, layer_types)`: Apply global magnitude-based pruning
- `gradient_based_pruning(gradients, amount, layer_types)`: Apply gradient-based pruning
- `random_pruning(amount, layer_types)`: Apply random pruning (baseline)
- `get_sparsity_summary()`: Get overall sparsity statistics
- `save_state()`: Save current model state
- `restore_state()`: Restore saved model state
- `make_pruning_permanent()`: Remove pruning masks permanently

### ModelCompressor

Class for model compression techniques.

#### Methods

- `dynamic_quantization(layer_types, dtype)`: Apply dynamic quantization
- `get_model_size(model)`: Calculate model size in MB

### Utility Functions

- `iterative_pruning(model, pruning_schedule, pruning_method, **kwargs)`: Apply pruning iteratively
- `prune_and_fine_tune(model, amount, fine_tune_fn, pruning_method, **kwargs)`: Prune and fine-tune
- `compare_pruning_methods(model_factory, methods, amount, **kwargs)`: Compare pruning methods

## Examples

Run the example script to see all features in action:

```bash
python example_usage.py
```

This will demonstrate:
- Magnitude-based pruning
- Structured pruning
- Global pruning
- Iterative pruning
- Method comparison
- Quantization
- Combined pruning and quantization

## Testing

Run the test suite:

```bash
pytest test_model_pruning.py -v
```

## Performance Tips

1. **Magnitude Pruning**: Best for unstructured sparsity, works well with sparse inference libraries
2. **Structured Pruning**: Provides actual speedup without specialized hardware
3. **Global Pruning**: Better distribution of pruning across layers
4. **Iterative Pruning**: Better accuracy preservation with gradual pruning
5. **Quantization**: Can be combined with pruning for maximum compression

## Use Cases

- **Edge Deployment**: Reduce model size for mobile/IoT devices
- **Inference Optimization**: Speed up inference with structured pruning
- **Memory Constraints**: Compress models to fit in limited memory
- **Research**: Compare pruning methods and study neural network redundancy

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.21+
- pytest 7.0+ (for testing)

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{calhacks12_pruning,
  title = {PyTorch Model Pruning and Compression Library},
  author = {CalHacks12 Team},
  year = {2025},
  url = {https://github.com/KG2468/CalHacks12}
}
```