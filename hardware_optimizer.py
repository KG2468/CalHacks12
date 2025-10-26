import torch
import torch.nn as nn
import torch.quantization
from torch.ao.quantization import quantize_dynamic
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping
from torch.quantization.observer import MinMaxObserver, HistogramObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Union, Callable
import numpy as np

class HardwareOptimizer:
    """
    A hardware-aware optimizer applying compression techniques (pruning, quantization)
    and preparing models for diverse hardware backends (PyTorch, ONNX, TensorFlow).
    """
    def __init__(self, model: nn.Module, example_input):
        self.model = model
        self.example_input = example_input
        # Default device is CUDA if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move model and input to the chosen device
        self.model = self.model.to(self.device)
        if isinstance(self.example_input, torch.Tensor):
            self.example_input = self.example_input.to(self.device)
        self.original_state = model.state_dict()
        
    def _get_device_capabilities(self) -> Dict:
        """Detect hardware capabilities, focusing on precision support."""
        caps = {
            'is_cuda': torch.cuda.is_available(),
            'cuda_major': torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0,
            'is_mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        return caps
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Determine the optimal floating-point precision (FP16/BF16/FP32)."""
        caps = self._get_device_capabilities()
        
        if caps['is_cuda']:
            if caps['cuda_major'] >= 8:  # Ampere/Hopper (VNNI, BF16 support)
                return torch.bfloat16
            return torch.float16 # Older GPUs
        elif caps['is_mps']:
            return torch.float16
        return torch.float32
    
    # --------------------------------------------------------------------------
    # 1. Mixed Precision (FP16/BF16 for GPU)
    # --------------------------------------------------------------------------
    def apply_mixed_precision(self) -> nn.Module:
        """Apply optimal floating-point precision based on hardware."""
        dtype = self.get_optimal_dtype()
        print(f"Applying Mixed Precision: Targeting {dtype} on {self.device.type}.")
        
        if dtype == next(self.model.parameters()).dtype:
            return self.model
        
        # Convert model and input
        self.model = self.model.to(dtype)
        if isinstance(self.example_input, torch.Tensor):
            self.example_input = self.example_input.to(dtype)
        
        return self.model

    # --------------------------------------------------------------------------
    # 2. Hardware-Aware Quantization (INT8)
    # --------------------------------------------------------------------------
    def quantize_model(self, qscheme: str = 'static', calibration_data_loader=None) -> nn.Module:
        """
        Quantizes the model. PTQ-Static is preferred for most accelerators.
        Note: PyTorch native quantization is primarily designed for CPU backends.
        """
        qscheme = qscheme.lower()
        self.model.eval()

        # PTQ needs to happen on the CPU to leverage FBGEMM/QNNPACK kernels
        if self.device.type != 'cpu':
            print("Warning: Quantization requires moving model to CPU for native PTQ kernels.")
            self.model = self.model.to('cpu')
            if isinstance(self.example_input, torch.Tensor):
                self.example_input = self.example_input.to('cpu')

        # Use the official, stable default QConfig for x86 CPUs (FBGEMM)
        # This configuration is validated for static quantization and includes 
        # per-channel weight quantization for higher accuracy.
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('fbgemm')
        
        if qscheme == 'dynamic':
            # Dynamic quantization is simpler but less performant than static/QAT
            print("-> Applying Dynamic Post-Training Quantization (INT8).")
            qconfig_spec = {nn.Linear: torch.quantization.default_dynamic_qconfig}
            return quantize_dynamic(self.model, qconfig_spec, dtype=torch.qint8)

        elif qscheme == 'static':
            print("-> Applying Static Post-Training Quantization (INT8) via FX Graph Mode.")
            
            # Prepare: Fuse modules and insert observers based on qconfig_mapping
            prepared_model = prepare_fx(self.model, qconfig_mapping, (self.example_input,))
            
            # Calibrate: Run data through the prepared model to collect statistics
            with torch.no_grad():
                if calibration_data_loader:
                    for inputs, _ in calibration_data_loader:
                        prepared_model(inputs)
                else:
                    prepared_model(self.example_input)
            
            # Convert: Final quantization
            return convert_fx(prepared_model)
            
        elif qscheme == 'qat':
            print("-> Preparing model for Quantization Aware Training (QAT).")
            self.model.train()
            # Prepare for QAT (inserts fake-quant modules)
            return prepare_qat_fx(self.model, qconfig_mapping, (self.example_input,))
            
        return self.model
    
    # --------------------------------------------------------------------------
    # 3. Memory-Aware Pruning (Structural)
    # --------------------------------------------------------------------------
    def prune_model(self, amount: float = 0.2, granularity: str = 'channel') -> nn.Module:
        """
        Prune the model using structural (channel) pruning to optimize memory hierarchy.
        Unstructured pruning is generally avoided as it doesn't guarantee a speedup.
        """
        if granularity != 'channel':
            print(f"Warning: Only structural 'channel' pruning is hardware-aware. Using it.")
            granularity = 'channel'
            
        print(f"Applying Structural Channel Pruning ({amount*100:.0f}% sparsity).")
        
        # Identify all Conv2d and Linear layers for pruning
        parameters_to_prune = [
            (module, 'weight') 
            for module in self.model.modules() 
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        
        # L2-norm structured pruning on output channels (dim=0)
        # This removes entire neurons/filters, resulting in a smaller, DENSE model
        # which is much more efficient for memory access (cache locality) and computation.
        for module, name in parameters_to_prune:
            try:
                # dim=0 is typically the output channel dimension (filters)
                prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)
            except Exception as e:
                # Skip if module dimensions are incompatible (e.g., small bias layer)
                continue
        
        # Make pruning permanent: remove pruning reparameterization
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
            
        return self.model
    
    # --------------------------------------------------------------------------
    # 4. Multi-Backend Deployment (ONNX/TensorFlow Stubs)
    # --------------------------------------------------------------------------
    
    def optimize_for_onnx(self, output_path: str = 'model_optimized.onnx') -> None:
        """
        Exports the PyTorch model to ONNX format for cross-platform deployment.
        The ONNX Runtime (ORT) can then apply further hardware-specific optimizations.
        """
        if next(self.model.parameters()).dtype != torch.float32:
            print("Warning: Exporting non-FP32 model. Use ONNX Runtime's tools for INT8/FP16 conversion.")
            # For ONNX export compatibility, use the original FP32 example input
            export_input = self.example_input.to(torch.float32)
        else:
            export_input = self.example_input

        # Handle quantized model state dict if present (must dequantize before export)
        # However, for simplicity, we rely on a clean FP32/FP16/BF16 model here.
        
        torch.onnx.export(
            self.model.cpu(),
            (export_input.cpu(),) if isinstance(export_input, torch.Tensor) else export_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        )
        print(f"Model exported to {output_path}. Ready for ONNX Runtime optimization passes.")

    def export_to_tensorflow_format(self, output_dir: str = 'tf_model') -> None:
        """
        Placeholder method for exporting a model compatible with TensorFlow/TFLite.
        Requires an intermediate conversion tool (e.g., ONNX-TFLite converter).
        """
        print("-" * 50)
        print("TensorFlow/TFLite Export Placeholder")
        print("To complete this step, convert the ONNX model using TensorFlow's TFLite Converter.")
        print("This allows leveraging TFLite's quantization/compilation for mobile/edge devices.")
        print("-" * 50)

    def get_model_stats(self) -> Dict:
        """Get statistics about the model size and performance."""
        total_params = sum(p.numel() for p in self.model.parameters())
        dtype = next(self.model.parameters()).dtype
        
        # Simple size estimate
        bytes_per_param = 4 
        if dtype in [torch.float16, torch.bfloat16]: bytes_per_param = 2
        elif dtype in [torch.qint8, torch.quint8]: bytes_per_param = 1
            
        model_size_mb = total_params * bytes_per_param / (1024 ** 2)
        
        return {
            'total_parameters': total_params,
            'model_size_mb': f"{model_size_mb:.2f} MB",
            'device': str(self.device),
            'dtype': str(dtype),
        }