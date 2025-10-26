import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.benchmark as benchmark
from typing import Dict, Union, Tuple
import os
import random
import time
import torch.nn.utils.prune # Need this for pruning utility functions

# --- Hugging Face Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# --- External Optimizer Import ---
try:
    from hardware_optimizer import HardwareOptimizer 
    from pruner import Pruner
except ImportError:
    print("FATAL ERROR: Could not import HardwareOptimizer.")
    print("Please ensure 'hardware_optimizer.py' is in the same directory.")
    exit()

# In test_hardware_optimizer.py (near the top)

# -----------------------------

# ==============================================================================
# I. LLM Model and Data Setup
# ==============================================================================

MODEL_NAME = "google/gemma-3-270m" 
MAX_LENGTH = 128
LLM_BENCHMARK_PROMPT = "The capital of France is"

# Global Tokenizer (required for all data operations)
GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
# Set padding token to EOS token ID for generation compatibility
GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token

# --- LLM Data Setup ---
class MockLLMDataset(Dataset):
    """Generates mock data for Perplexity evaluation."""
    def __init__(self, tokenizer, size=500, max_len=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.texts = ["This is a sample sentence for testing perplexity evaluation."] * size
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenizer returns a dict of tensors (input_ids, attention_mask)
        encoding = self.tokenizer(
            self.texts[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        # Squeeze to remove the leading '1' dimension added by return_tensors='pt'
        return {k: v.squeeze(0) for k, v in encoding.items()}, torch.tensor(0) 

def load_gemma_model(precision: torch.dtype, device: torch.device) -> AutoModelForCausalLM:
    """Loads the Gemma model in the specified precision and maps it to the device."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=precision, 
        device_map=device,      
        low_cpu_mem_usage=True
    )
    # Ensure model is in evaluation mode for inference testing
    model.eval()
    return model

# ==============================================================================
# II. Testing and Benchmarking Utilities (LLM Metrics)
# ==============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """Calculates model size by saving state_dict to a temp file."""
    # Ensure all parameters are on CPU before saving for consistent size measurement
    model.cpu() 
    torch.save(model.state_dict(), "temp_model_size.pth")
    size_mb = os.path.getsize("temp_model_size.pth") / (1024**2)
    os.remove("temp_model_size.pth")
    return size_mb

def evaluate_perplexity(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Calculates Perplexity (PPL) for a generative model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    model.to(device) # Ensure model is on the target device

    with torch.no_grad():
        for inputs_dict, _ in data_loader:
            # Move dict inputs (input_ids, attention_mask) to device
            input_ids = inputs_dict['input_ids'].to(device)
            attention_mask = inputs_dict['attention_mask'].to(device)

            # Pass inputs AND targets (labels=input_ids) to get internal loss calculation
            # Use labels=input_ids for Causal LMs to compute perplexity loss
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            
            # Loss is cross-entropy per token
            loss = outputs.loss.item() 
            
            # Count the number of tokens (excluding padding) for weighted average
            num_tokens = torch.sum(attention_mask).item()
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
    avg_loss = total_loss / total_tokens
    # Perplexity is exp(average loss)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

# In test_hardware_optimizer.py, replace the existing benchmark_generation_speed function

# In test_hardware_optimizer.py, replace the existing benchmark_generation_speed function

def benchmark_generation_speed(model: nn.Module, prompt_text: str, tokenizer, device: torch.device) -> float:
    """
    Measures generation speed in Tokens Per Second (TPS). 
    Handles both CUDA and CPU devices for timing.
    """
    model.eval()
    model.to(device)
    
    # Encode the prompt
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    
    num_return_sequences = 1
    max_new_tokens = 50
    
    # --- WARMUP (CRUCIAL) ---
    # Warm-up run doesn't need timing, but needs synchronization if on CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=10, num_return_sequences=num_return_sequences, do_sample=False)
    
    # --- MAIN BENCHMARK RUN ---
    
    if device.type == 'cuda':
        # GPU timing using CUDA Events
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            start_event.record()
            output_sequences = model.generate(
                input_ids, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
            end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0 # Time in seconds
        
    else:
        # CPU timing using standard time library
        start_time = time.time()
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        elapsed_time = time.time() - start_time
        
    # --- RESULTS CALCULATION ---
    
    # Calculate tokens generated (output length minus input length)
    total_generated_tokens = (output_sequences.size(1) - input_ids.size(1)) * num_return_sequences
    
    # Speed is tokens per second (TPS)
    speed_tps = total_generated_tokens / elapsed_time
    
    return speed_tps
# ==============================================================================
# III. Execution and Reporting
# ==============================================================================

if __name__ == "__main__":
    
    # --- Setup ---
    # Target CUDA if available for LLM acceleration
    TARGET_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine optimal dtype for GPU
    if TARGET_DEVICE.type == 'cuda':
        major, _ = torch.cuda.get_device_capability()
        TARGET_DTYPE = torch.bfloat16 if major >= 8 else torch.float16
    else:
        TARGET_DTYPE = torch.float32

    
    print("\n" + "="*70)
    print(f"TESTING GEMMA-3-270M OPTIMIZATION | TARGET DEVICE: {TARGET_DEVICE.type.upper()} ({TARGET_DTYPE})")
    print("="*70)

    # --- 1. Load Baseline Model ---
    print("\n[1/4] Loading FP32 Baseline Model...")
    # Load FP32 model, then cast to the optimal mixed precision dtype (BF16/FP16)
    # This is the effective baseline for modern GPU testing.
    fp32_model = load_gemma_model(torch.float32, TARGET_DEVICE).to(TARGET_DTYPE) 
    
    # Create DataLoaders
    CALIB_LOADER = DataLoader(MockLLMDataset(GLOBAL_TOKENIZER, size=10), batch_size=2)
    TEST_LOADER = DataLoader(MockLLMDataset(GLOBAL_TOKENIZER, size=100), batch_size=4) 

    # Example input dictionary needed for optimizer's FX trace (uses input_ids)
    EXAMPLE_INPUT_DICT, _ = next(iter(CALIB_LOADER))
    
    # Use the input_ids tensor for the optimizer's example_input
    optimizer_example_input = EXAMPLE_INPUT_DICT['input_ids'][0].unsqueeze(0).to(TARGET_DTYPE) 


    # --- 2. Optimization Pipeline ---
    print("\n[2/4] Applying Hardware Optimization (Mixed Precision + Pruning)...")
    
    # A. Create a clean copy of the model for optimization
    optimized_model = load_gemma_model(torch.float32, TARGET_DEVICE)
    
    # B. Initialize optimizer and apply the pipeline
    optimizer = HardwareOptimizer(optimized_model, optimizer_example_input)

    # 1. Apply Mixed Precision (Handles FP16/BF16 conversion)
    optimized_model = optimizer.apply_mixed_precision()
    
    # 2. Apply Structural Pruning
    #optimized_model = optimizer.prune_model(amount=0.1, granularity='channel')
    
    # 3. No INT8 quantization is applied here, as it's typically CPU-only or requires TensorRT.

    
    # --- 3. Measurement ---
    print("\n[3/4] Measuring Performance...")

    # Baseline Measurements
    fp32_size = get_model_size_mb(fp32_model)
    fp32_ppl = evaluate_perplexity(fp32_model, TEST_LOADER, TARGET_DEVICE)
    fp32_tps = benchmark_generation_speed(fp32_model, LLM_BENCHMARK_PROMPT, GLOBAL_TOKENIZER, TARGET_DEVICE)
    
    # Optimized Measurements
    optimized_size = get_model_size_mb(optimized_model)
    optimized_ppl = evaluate_perplexity(optimized_model, TEST_LOADER, TARGET_DEVICE)
    optimized_tps = benchmark_generation_speed(optimized_model, LLM_BENCHMARK_PROMPT, GLOBAL_TOKENIZER, TARGET_DEVICE)


    # --- 4. Final Comparison Report ---
    print("\n" + "="*70)
    print("FINAL GEMMA OPTIMIZATION REPORT (Pruning + Mixed Precision)")
    print("="*70)
    
    # Calculate metrics
    size_reduction = 100 * (fp32_size - optimized_size) / fp32_size
    ppl_change = optimized_ppl - fp32_ppl
    speedup = optimized_tps / fp32_tps
    
    print(f"Target Device: {TARGET_DEVICE.type.upper()} ({TARGET_DTYPE})")
    print(f"Optimization Scheme: Mixed Precision ({TARGET_DTYPE}) + Pruning (10%)")
    print("-" * 70)
    
    print(f"| Metric | Baseline ({TARGET_DTYPE}) | Optimized Model | Change |")
    print(f"|:---|:---|:---|:---|")
    print(f"| Model Size | {fp32_size:.2f} MB | {optimized_size:.2f} MB | {size_reduction:.1f}% Reduction |")
    print(f"| Perplexity (PPL) | {fp32_ppl:.3f} | {optimized_ppl:.3f} | {ppl_change:.3f} (Lower is better) |")
    print(f"| Tokens/Second (TPS) | {fp32_tps:.2f} | {optimized_tps:.2f} | {speedup:.2f}x Speedup |")
    print("-" * 70)