"""
Custom Machine Learning Model Pruning and Compression for PyTorch

This module provides various pruning and compression techniques for PyTorch neural networks,
including magnitude-based pruning, structured pruning, gradient-based pruning, and quantization.

Author: CalHacks12 Team
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from collections import OrderedDict


class ModelPruner:
    """
    A comprehensive model pruning class that supports multiple pruning strategies.
    
    Supported strategies:
    - Magnitude-based pruning (unstructured)
    - Structured pruning (channel/filter pruning)
    - Gradient-based pruning
    - Random pruning
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the ModelPruner.
        
        Args:
            model: PyTorch model to be pruned
        """
        self.model = model
        self.original_state = None
        self.pruning_history = []
        
    def save_state(self):
        """Save the current state of the model for potential restoration."""
        self.original_state = OrderedDict()
        for name, param in self.model.named_parameters():
            self.original_state[name] = param.data.clone()
            
    def restore_state(self):
        """Restore the model to its saved state."""
        if self.original_state is None:
            raise ValueError("No saved state to restore. Call save_state() first.")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_state:
                    param.data.copy_(self.original_state[name])
                    
    def magnitude_pruning(
        self, 
        amount: float = 0.3, 
        layer_types: Tuple = (nn.Linear, nn.Conv2d),
        name_filter: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Apply magnitude-based unstructured pruning to the model.
        
        Args:
            amount: Fraction of connections to prune (0.0 to 1.0)
            layer_types: Tuple of layer types to prune
            name_filter: Optional string to filter layer names
            
        Returns:
            Dictionary with pruning statistics per layer
        """
        stats = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                if name_filter and name_filter not in name:
                    continue
                    
                # Apply L1 unstructured pruning based on weight magnitude
                prune.l1_unstructured(module, name='weight', amount=amount)
                
                # Calculate sparsity
                weight_tensor = module.weight
                zeros = float(torch.sum(weight_tensor == 0))
                total = float(weight_tensor.nelement())
                sparsity = zeros / total
                
                stats[name] = {
                    'sparsity': sparsity,
                    'pruned_params': int(zeros),
                    'total_params': int(total)
                }
                
        self.pruning_history.append({
            'method': 'magnitude_pruning',
            'amount': amount,
            'stats': stats
        })
        
        return stats
    
    def structured_pruning(
        self,
        amount: float = 0.3,
        dim: int = 0,
        layer_types: Tuple = (nn.Conv2d,)
    ) -> Dict[str, float]:
        """
        Apply structured pruning (removes entire channels/filters).
        
        Args:
            amount: Fraction of structures to prune (0.0 to 1.0)
            dim: Dimension along which to prune (0 for output channels, 1 for input channels)
            layer_types: Tuple of layer types to prune
            
        Returns:
            Dictionary with pruning statistics per layer
        """
        stats = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                # Apply structured pruning using L_n norm
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=amount, 
                    n=2, 
                    dim=dim
                )
                
                # Calculate sparsity
                weight_tensor = module.weight
                zeros = float(torch.sum(weight_tensor == 0))
                total = float(weight_tensor.nelement())
                sparsity = zeros / total
                
                stats[name] = {
                    'sparsity': sparsity,
                    'pruned_params': int(zeros),
                    'total_params': int(total),
                    'dim': dim
                }
                
        self.pruning_history.append({
            'method': 'structured_pruning',
            'amount': amount,
            'dim': dim,
            'stats': stats
        })
        
        return stats
    
    def gradient_based_pruning(
        self,
        gradients: Dict[str, torch.Tensor],
        amount: float = 0.3,
        layer_types: Tuple = (nn.Linear, nn.Conv2d)
    ) -> Dict[str, float]:
        """
        Apply gradient-based pruning (prune weights with smallest gradient magnitudes).
        
        Args:
            gradients: Dictionary mapping parameter names to their gradients
            amount: Fraction of connections to prune (0.0 to 1.0)
            layer_types: Tuple of layer types to prune
            
        Returns:
            Dictionary with pruning statistics per layer
        """
        stats = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                param_name = f"{name}.weight"
                
                if param_name not in gradients:
                    continue
                
                # Get gradient magnitudes
                grad = gradients[param_name]
                grad_magnitude = torch.abs(grad)
                
                # Calculate threshold - prune weights with smallest gradient magnitudes
                # (keep weights with larger gradients as they're more important)
                threshold = torch.quantile(grad_magnitude.flatten(), amount)
                
                # Create mask - keep weights with gradient magnitude above threshold
                mask = (grad_magnitude > threshold).float()
                
                # Apply mask
                with torch.no_grad():
                    module.weight.data *= mask
                    
                # Calculate sparsity
                weight_tensor = module.weight
                zeros = float(torch.sum(weight_tensor == 0))
                total = float(weight_tensor.nelement())
                sparsity = zeros / total
                
                stats[name] = {
                    'sparsity': sparsity,
                    'pruned_params': int(zeros),
                    'total_params': int(total)
                }
                
        self.pruning_history.append({
            'method': 'gradient_based_pruning',
            'amount': amount,
            'stats': stats
        })
        
        return stats
    
    def random_pruning(
        self,
        amount: float = 0.3,
        layer_types: Tuple = (nn.Linear, nn.Conv2d)
    ) -> Dict[str, float]:
        """
        Apply random unstructured pruning to the model (useful as baseline).
        
        Args:
            amount: Fraction of connections to prune (0.0 to 1.0)
            layer_types: Tuple of layer types to prune
            
        Returns:
            Dictionary with pruning statistics per layer
        """
        stats = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                # Apply random unstructured pruning
                prune.random_unstructured(module, name='weight', amount=amount)
                
                # Calculate sparsity
                weight_tensor = module.weight
                zeros = float(torch.sum(weight_tensor == 0))
                total = float(weight_tensor.nelement())
                sparsity = zeros / total
                
                stats[name] = {
                    'sparsity': sparsity,
                    'pruned_params': int(zeros),
                    'total_params': int(total)
                }
                
        self.pruning_history.append({
            'method': 'random_pruning',
            'amount': amount,
            'stats': stats
        })
        
        return stats
    
    def global_magnitude_pruning(
        self,
        amount: float = 0.3,
        layer_types: Tuple = (nn.Linear, nn.Conv2d)
    ) -> Dict[str, float]:
        """
        Apply global magnitude-based pruning across all layers.
        
        Args:
            amount: Fraction of connections to prune globally (0.0 to 1.0)
            layer_types: Tuple of layer types to prune
            
        Returns:
            Dictionary with pruning statistics per layer
        """
        # Collect all parameters to prune
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        # Calculate statistics
        stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                weight_tensor = module.weight
                zeros = float(torch.sum(weight_tensor == 0))
                total = float(weight_tensor.nelement())
                sparsity = zeros / total
                
                stats[name] = {
                    'sparsity': sparsity,
                    'pruned_params': int(zeros),
                    'total_params': int(total)
                }
                
        self.pruning_history.append({
            'method': 'global_magnitude_pruning',
            'amount': amount,
            'stats': stats
        })
        
        return stats
    
    def make_pruning_permanent(self):
        """
        Make all pruning masks permanent by removing the mask and modifying the weights.
        This reduces the model size permanently.
        """
        for module in self.model.modules():
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_orig'):
                prune.remove(module, 'bias')
                
    def get_sparsity_summary(self) -> Dict[str, Union[float, int]]:
        """
        Get a summary of the current model sparsity.
        
        Returns:
            Dictionary with overall sparsity statistics
        """
        total_zeros = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            # Check for pruned weights
            if hasattr(module, 'weight'):
                weight = module.weight
                zeros = float(torch.sum(weight == 0))
                total = float(weight.nelement())
                total_zeros += zeros
                total_params += total
                
        overall_sparsity = total_zeros / total_params if total_params > 0 else 0
        
        return {
            'overall_sparsity': overall_sparsity,
            'total_zero_params': int(total_zeros),
            'total_params': int(total_params),
            'active_params': int(total_params - total_zeros)
        }


class ModelCompressor:
    """
    A class for model compression using quantization techniques.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the ModelCompressor.
        
        Args:
            model: PyTorch model to be compressed
        """
        self.model = model
        self.original_dtype = next(model.parameters()).dtype
        
    def dynamic_quantization(
        self,
        layer_types: Tuple = (nn.Linear, nn.LSTM, nn.GRU),
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Args:
            layer_types: Tuple of layer types to quantize
            dtype: Target data type for quantization
            
        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            qconfig_spec=set(layer_types),
            dtype=dtype
        )
        
        return quantized_model
    
    def get_model_size(self, model: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Calculate the size of the model in MB.
        
        Args:
            model: Model to measure (uses self.model if None)
            
        Returns:
            Dictionary with size information
        """
        if model is None:
            model = self.model
            
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        total_size = param_size + buffer_size
        size_mb = total_size / (1024 ** 2)
        
        return {
            'total_size_mb': size_mb,
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_size_mb': buffer_size / (1024 ** 2),
            'total_size_bytes': total_size
        }


def iterative_pruning(
    model: nn.Module,
    pruning_schedule: List[float],
    pruning_method: str = 'magnitude',
    **kwargs
) -> Tuple[nn.Module, List[Dict]]:
    """
    Apply iterative pruning with a schedule of pruning amounts.
    
    Args:
        model: PyTorch model to prune
        pruning_schedule: List of pruning amounts for each iteration
        pruning_method: Method to use ('magnitude', 'structured', 'global', 'random')
        **kwargs: Additional arguments for the pruning method
        
    Returns:
        Tuple of (pruned model, list of statistics for each iteration)
    """
    pruner = ModelPruner(model)
    all_stats = []
    
    for iteration, amount in enumerate(pruning_schedule):
        if pruning_method == 'magnitude':
            stats = pruner.magnitude_pruning(amount=amount, **kwargs)
        elif pruning_method == 'structured':
            stats = pruner.structured_pruning(amount=amount, **kwargs)
        elif pruning_method == 'global':
            stats = pruner.global_magnitude_pruning(amount=amount, **kwargs)
        elif pruning_method == 'random':
            stats = pruner.random_pruning(amount=amount, **kwargs)
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")
            
        all_stats.append({
            'iteration': iteration,
            'amount': amount,
            'stats': stats,
            'summary': pruner.get_sparsity_summary()
        })
        
    return model, all_stats


def prune_and_fine_tune(
    model: nn.Module,
    amount: float,
    fine_tune_fn: callable,
    pruning_method: str = 'magnitude',
    **kwargs
) -> Tuple[nn.Module, Dict]:
    """
    Prune a model and apply a fine-tuning function.
    
    Args:
        model: PyTorch model to prune
        amount: Amount of pruning to apply
        fine_tune_fn: Function to fine-tune the model (takes model as argument)
        pruning_method: Method to use for pruning
        **kwargs: Additional arguments for the pruning method
        
    Returns:
        Tuple of (pruned and fine-tuned model, pruning statistics)
    """
    pruner = ModelPruner(model)
    
    # Apply pruning
    if pruning_method == 'magnitude':
        stats = pruner.magnitude_pruning(amount=amount, **kwargs)
    elif pruning_method == 'structured':
        stats = pruner.structured_pruning(amount=amount, **kwargs)
    elif pruning_method == 'global':
        stats = pruner.global_magnitude_pruning(amount=amount, **kwargs)
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")
    
    # Fine-tune the pruned model
    fine_tuned_model = fine_tune_fn(model)
    
    # Get final statistics
    final_stats = {
        'initial_stats': stats,
        'final_summary': pruner.get_sparsity_summary()
    }
    
    return fine_tuned_model, final_stats


def compare_pruning_methods(
    model_factory: callable,
    methods: List[str] = ['magnitude', 'structured', 'global', 'random'],
    amount: float = 0.3,
    **kwargs
) -> Dict[str, Dict]:
    """
    Compare different pruning methods on the same model architecture.
    
    Args:
        model_factory: Function that returns a fresh model instance
        methods: List of pruning methods to compare
        amount: Amount of pruning to apply
        **kwargs: Additional arguments for pruning methods
        
    Returns:
        Dictionary mapping method names to their statistics
    """
    results = {}
    
    for method in methods:
        # Create a fresh model for each method
        model = model_factory()
        pruner = ModelPruner(model)
        
        # Apply pruning
        if method == 'magnitude':
            stats = pruner.magnitude_pruning(amount=amount, **kwargs)
        elif method == 'structured':
            stats = pruner.structured_pruning(amount=amount, **kwargs)
        elif method == 'global':
            stats = pruner.global_magnitude_pruning(amount=amount, **kwargs)
        elif method == 'random':
            stats = pruner.random_pruning(amount=amount, **kwargs)
        else:
            continue
            
        # Get summary
        summary = pruner.get_sparsity_summary()
        
        results[method] = {
            'per_layer_stats': stats,
            'overall_summary': summary
        }
        
    return results
