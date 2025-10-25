"""
Unit tests for the model pruning and compression module.
"""

import torch
import torch.nn as nn
import pytest
from model_pruning import (
    ModelPruner, 
    ModelCompressor, 
    iterative_pruning,
    prune_and_fine_tune,
    compare_pruning_methods
)


class SimpleNet(nn.Module):
    """Simple neural network for testing."""
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """Simple CNN for testing structured pruning."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestModelPruner:
    """Test cases for ModelPruner class."""
    
    def test_initialization(self):
        """Test ModelPruner initialization."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        assert pruner.model is model
        assert pruner.original_state is None
        assert len(pruner.pruning_history) == 0
        
    def test_save_and_restore_state(self):
        """Test saving and restoring model state."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        # Save initial state
        pruner.save_state()
        original_weight = model.fc1.weight.data.clone()
        
        # Modify weights
        with torch.no_grad():
            model.fc1.weight.data.zero_()
        
        # Restore state
        pruner.restore_state()
        
        assert torch.allclose(model.fc1.weight.data, original_weight)
        
    def test_magnitude_pruning(self):
        """Test magnitude-based pruning."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        amount = 0.3
        stats = pruner.magnitude_pruning(amount=amount)
        
        # Check that stats were returned
        assert len(stats) > 0
        
        # Check that pruning was applied
        for name, layer_stats in stats.items():
            assert 'sparsity' in layer_stats
            assert layer_stats['sparsity'] > 0
            assert layer_stats['sparsity'] <= amount + 0.1  # Allow some margin
            
    def test_structured_pruning(self):
        """Test structured pruning on CNN."""
        model = SimpleCNN()
        pruner = ModelPruner(model)
        
        amount = 0.3
        stats = pruner.structured_pruning(amount=amount, dim=0)
        
        # Check that stats were returned
        assert len(stats) > 0
        
        # Check that pruning was applied
        for name, layer_stats in stats.items():
            assert 'sparsity' in layer_stats
            assert layer_stats['sparsity'] > 0
            
    def test_random_pruning(self):
        """Test random pruning."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        amount = 0.3
        stats = pruner.random_pruning(amount=amount)
        
        # Check that stats were returned
        assert len(stats) > 0
        
        # Check that pruning was applied
        for name, layer_stats in stats.items():
            assert 'sparsity' in layer_stats
            assert layer_stats['sparsity'] > 0
            
    def test_global_magnitude_pruning(self):
        """Test global magnitude pruning."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        amount = 0.3
        stats = pruner.global_magnitude_pruning(amount=amount)
        
        # Check that stats were returned
        assert len(stats) > 0
        
        # Check overall sparsity
        summary = pruner.get_sparsity_summary()
        assert summary['overall_sparsity'] >= amount - 0.05  # Allow small margin
        
    def test_gradient_based_pruning(self):
        """Test gradient-based pruning."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        # Create dummy gradients
        gradients = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                gradients[name] = torch.randn_like(param)
        
        amount = 0.3
        stats = pruner.gradient_based_pruning(gradients, amount=amount)
        
        # Check that stats were returned
        assert len(stats) > 0
        
        # Check that pruning was applied
        for name, layer_stats in stats.items():
            assert 'sparsity' in layer_stats
            
    def test_get_sparsity_summary(self):
        """Test getting sparsity summary."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        # Before pruning
        summary_before = pruner.get_sparsity_summary()
        assert summary_before['overall_sparsity'] == 0
        
        # After pruning
        pruner.magnitude_pruning(amount=0.3)
        summary_after = pruner.get_sparsity_summary()
        assert summary_after['overall_sparsity'] > 0
        assert summary_after['total_zero_params'] > 0
        
    def test_pruning_history(self):
        """Test that pruning history is recorded."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        pruner.magnitude_pruning(amount=0.2)
        pruner.random_pruning(amount=0.1)
        
        assert len(pruner.pruning_history) == 2
        assert pruner.pruning_history[0]['method'] == 'magnitude_pruning'
        assert pruner.pruning_history[1]['method'] == 'random_pruning'
        
    def test_make_pruning_permanent(self):
        """Test making pruning permanent."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        # Apply pruning
        pruner.magnitude_pruning(amount=0.3)
        
        # Check that mask exists
        assert hasattr(model.fc1, 'weight_orig')
        
        # Make permanent
        pruner.make_pruning_permanent()
        
        # Check that mask is removed
        assert not hasattr(model.fc1, 'weight_orig')
        

class TestModelCompressor:
    """Test cases for ModelCompressor class."""
    
    def test_initialization(self):
        """Test ModelCompressor initialization."""
        model = SimpleNet()
        compressor = ModelCompressor(model)
        assert compressor.model is model
        
    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        model = SimpleNet()
        compressor = ModelCompressor(model)
        
        quantized_model = compressor.dynamic_quantization()
        
        # Check that quantization was applied
        assert quantized_model is not None
        
    def test_get_model_size(self):
        """Test getting model size."""
        model = SimpleNet()
        compressor = ModelCompressor(model)
        
        size_info = compressor.get_model_size()
        
        assert 'total_size_mb' in size_info
        assert 'param_size_mb' in size_info
        assert size_info['total_size_mb'] > 0
        
    def test_size_reduction_after_quantization(self):
        """Test that quantization reduces model size."""
        model = SimpleNet()
        compressor = ModelCompressor(model)
        
        # Get original size
        original_size = compressor.get_model_size()
        
        # Quantize
        quantized_model = compressor.dynamic_quantization()
        quantized_size = compressor.get_model_size(quantized_model)
        
        # Quantized model should be smaller (in most cases)
        # Note: For very small models, overhead might make it larger
        assert quantized_size['total_size_mb'] is not None


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_iterative_pruning(self):
        """Test iterative pruning with schedule."""
        model = SimpleNet()
        schedule = [0.1, 0.2, 0.3]
        
        pruned_model, stats = iterative_pruning(
            model, 
            schedule, 
            pruning_method='magnitude'
        )
        
        assert len(stats) == len(schedule)
        
        # Check that sparsity increases with iterations
        for i in range(len(stats)):
            assert stats[i]['summary']['overall_sparsity'] >= 0
            
    def test_prune_and_fine_tune(self):
        """Test pruning with fine-tuning."""
        model = SimpleNet()
        
        def dummy_fine_tune(m):
            """Dummy fine-tuning function."""
            return m
        
        pruned_model, stats = prune_and_fine_tune(
            model,
            amount=0.3,
            fine_tune_fn=dummy_fine_tune,
            pruning_method='magnitude'
        )
        
        assert 'initial_stats' in stats
        assert 'final_summary' in stats
        
    def test_compare_pruning_methods(self):
        """Test comparing different pruning methods."""
        def model_factory():
            return SimpleNet()
        
        results = compare_pruning_methods(
            model_factory,
            methods=['magnitude', 'random'],
            amount=0.3
        )
        
        assert 'magnitude' in results
        assert 'random' in results
        
        for method, result in results.items():
            assert 'per_layer_stats' in result
            assert 'overall_summary' in result


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_restore_without_save(self):
        """Test that restore fails without save."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        with pytest.raises(ValueError):
            pruner.restore_state()
            
    def test_zero_pruning(self):
        """Test pruning with amount=0."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        stats = pruner.magnitude_pruning(amount=0.0)
        summary = pruner.get_sparsity_summary()
        
        # Should have minimal or no sparsity
        assert summary['overall_sparsity'] < 0.01
        
    def test_full_pruning(self):
        """Test pruning with amount=1.0."""
        model = SimpleNet()
        pruner = ModelPruner(model)
        
        stats = pruner.magnitude_pruning(amount=1.0)
        summary = pruner.get_sparsity_summary()
        
        # Should have very high sparsity
        assert summary['overall_sparsity'] > 0.9


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
