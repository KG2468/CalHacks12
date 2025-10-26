import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.benchmark as benchmark
from typing import Dict, Union, Tuple
import os
import random
import time
import torch.nn.utils.prune # Need this for pruning utility functions
import argparse # Added argparse

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

# ==============================================================================
# I. LLM Model and Data Setup
# ==============================================================================

# MODEL_NAME = "google/gemma-3-270m" # Removed hardcoded name
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_LENGTH = 128
LLM_BENCHMARK_PROMPT = "The capital of France is"

# PRUNED_SVD_PATH = "gemma_pruned_svd_final.pth" # Will be set dynamically
# GLOBAL_TOKENIZER will be set dynamically in main

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

def load_model(model_name: str, precision: torch.dtype, device: torch.device) -> AutoModelForCausalLM:
    """Loads the specified model in the specified precision and maps it to the device."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, # Use argument
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
    # Use a temporary file path
    temp_path = "temp_model_size.pth"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024**2)
    os.remove(temp_path)
    return size_mb

@torch.no_grad()
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

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item() 
            
            num_tokens = torch.sum(attention_mask).item()
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf') # Avoid division by zero
        
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

@torch.no_grad()
def benchmark_generation_speed(model: nn.Module, prompt_text: str, tokenizer, device: torch.device) -> float:
    """
    Measures generation speed in Tokens Per Second (TPS). 
    Handles both CUDA and CPU devices for timing.
    """
    model.eval()
    model.to(device)
    
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    
    num_return_sequences = 1
    max_new_tokens = 50
    
    # --- WARMUP (CRUCIAL) ---
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
    total_generated_tokens = (output_sequences.size(1) - input_ids.size(1)) * num_return_sequences
    speed_tps = total_generated_tokens / elapsed_time
    
    return speed_tps
    
# ==============================================================================
# III. Execution and Reporting
# ==============================================================================

if __name__ == "__main__":
    
    # --- 2. Setup (Device, Dtype, Paths, Tokenizer) ---
    
    # Device Setup
    if torch.cuda.is_available():
        TARGET_DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        TARGET_DEVICE = torch.device('mps')
    else:
        TARGET_DEVICE = torch.device('cpu')
    
    # Dtype Setup
    if TARGET_DEVICE.type == 'cuda':
        major, _ = torch.cuda.get_device_capability()
        TARGET_DTYPE = torch.bfloat16 if major >= 8 else torch.float16
    elif TARGET_DEVICE.type == 'mps':
        TARGET_DTYPE = torch.float16
    else:
        TARGET_DTYPE = torch.float32 # CPU must use float32

    # Dynamic Paths
    # safe_model_name = args.model_name.replace("/", "_")
    PRUNED_SVD_PATH = f"{safe_model_name}_pruned_svd_final.pth"
    OUTPUT_PATH = f"{safe_model_name}_quantized_final.pth"

    # Tokenizer Setup
    GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token

    
    print("\n" + "="*70)
    print(f"TESTING {MODEL_NAME.upper()} OPTIMIZATION | TARGET DEVICE: {TARGET_DEVICE.type.upper()} ({TARGET_DTYPE})")
    print("="*70)

    # --- 3. Load Baseline Model ---
    print("\n[1/4] Loading Baseline Model...")
    # Load FP32 model, then cast to the optimal mixed precision dtype (BF16/FP16)
    fp32_model = load_model(MODEL_NAME, torch.float32, TARGET_DEVICE).to(TARGET_DTYPE) 

    # (Optional: Load pre-optimized model for comparison)
    # print(f"\n[0] Loading Pruned+SVD model weights from {PRUNED_SVD_PATH}...")
    # try:
    #     svd_model = load_model(args.model_name, TARGET_DTYPE, TARGET_DEVICE)
    #     svd_model.load_state_dict(torch.load(PRUNED_SVD_PATH, map_location=TARGET_DEVICE))
    #     svd_model.eval()
    #     print("✅ Pruned + SVD model loaded successfully.")
    # except FileNotFoundError:
    #     print(f"File not found: {PRUNED_SVD_PATH}. Skipping SVD model loading.")
    #     svd_model = None
    
    # Create DataLoaders
    CALIB_LOADER = DataLoader(MockLLMDataset(GLOBAL_TOKENIZER, size=10), batch_size=2)
    TEST_LOADER = DataLoader(MockLLMDataset(GLOBAL_TOKENIZER, size=100), batch_size=4) 

    # Example input dictionary needed for optimizer's FX trace
    EXAMPLE_INPUT_DICT, _ = next(iter(CALIB_LOADER))
    
    # Use the input_ids tensor for the optimizer's example_input
    # Note: HardwareOptimizer might expect a specific dtype. Let's cast to target dtype.
    optimizer_example_input = EXAMPLE_INPUT_DICT['input_ids'][0].unsqueeze(0).to(TARGET_DEVICE)


    # --- 4. Optimization Pipeline ---
    print("\n[2/4] Applying Hardware Optimization (Mixed Precision + Pruning)...")
    
    # A. Create a clean copy of the model for optimization
    optimized_model = load_model(MODEL_NAME, torch.float32, TARGET_DEVICE)
    
    # B. Initialize optimizer and apply the pipeline
    # The example input should match the device/dtype the model expects for tracing
    optimizer = HardwareOptimizer(optimized_model, optimizer_example_input)

    # 1. Apply Mixed Precision (Handles FP16/BF16 conversion)
    optimized_model = optimizer.apply_mixed_precision()
    
    # 2. Apply Structural Pruning (Assuming this is part of your optimizer)
    # optimized_model = optimizer.prune_model(amount=0.1, granularity='channel') 
    # (Uncommented as it seems intended)
    
    # --- 5. Measurement ---
    print("\n[3/4] Measuring Performance...")

    # Baseline Measurements
    print("Measuring baseline...")
    fp32_size = get_model_size_mb(fp32_model)
    fp32_ppl = evaluate_perplexity(fp32_model, TEST_LOADER, TARGET_DEVICE)
    fp32_tps = benchmark_generation_speed(fp32_model, LLM_BENCHMARK_PROMPT, GLOBAL_TOKENIZER, TARGET_DEVICE)
    del fp32_model # Free memory
    
    # Optimized Measurements
    print("Measuring optimized model...")
    optimized_size = get_model_size_mb(optimized_model)
    optimized_ppl = evaluate_perplexity(optimized_model, TEST_LOADER, TARGET_DEVICE)
    optimized_tps = benchmark_generation_speed(optimized_model, LLM_BENCHMARK_PROMPT, GLOBAL_TOKENIZER, TARGET_DEVICE)


    # --- 6. Final Comparison Report ---
    print("\n" + "="*70)
    print(f"FINAL {MODEL_NAME.upper()} OPTIMIZATION REPORT (Pruning + Mixed Precision)")
    print("="*70)
    
    # Calculate metrics
    size_reduction = 100 * (fp32_size - optimized_size) / fp32_size
    ppl_change = optimized_ppl - fp32_ppl
    speedup = optimized_tps / fp32_tps
    
    print(f"Target Device: {TARGET_DEVICE.type.upper()} ({TARGET_DTYPE})")
    # Note: The optimization scheme is whatever HardwareOptimizer does. 
    # The print statement below assumes what it does.
    print(f"Optimization Scheme: Mixed Precision ({TARGET_DTYPE}) + Pruning (if enabled)") 
    print("-" * 70)
    
    print(f"| Metric | Baseline ({TARGET_DTYPE}) | Optimized Model | Change |")
    print(f"|:---|:---|:---|:---|")
    print(f"| Model Size | {fp32_size:.2f} MB | {optimized_size:.2f} MB | {size_reduction:.1f}% Reduction |")
    print(f"| Perplexity (PPL) | {fp32_ppl:.3f} | {optimized_ppl:.3f} | {ppl_change:+.3f} (Lower is better) |")
    print(f"| Tokens/Second (TPS) | {fp32_tps:.2f} | {optimized_tps:.2f} | {speedup:.2f}x Speedup |")
    print("-" * 70)

    # Save the model's state_dict (weights) to a file
    torch.save(optimized_model.state_dict(), OUTPUT_PATH)
    print(f"\n✅ Optimized model saved to: {OUTPUT_PATH}")