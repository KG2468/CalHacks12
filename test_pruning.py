"""
test_pruner_svd.py
Evaluates pruning and pruning+SVD compression on Gemma-3-270M.
Follows the same measurement structure as test_hardware_optimizer.py.
"""

import os, time, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from pruner import Pruner, PrunerWithSVD  # your existing file

# ================================================================
# I.  Model and Data Setup
# ================================================================
MODEL_NAME = "google/gemma-3-270m"
MAX_LEN = 128
PROMPT = "The capital of France is"

PRUNED_SVD_PATH = "gemma_quantized_final.pth"

# Tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
TOKENIZER.pad_token = TOKENIZER.eos_token


class MockLLMDataset(Dataset):
    def __init__(self, tokenizer, size=100, max_len=MAX_LEN):
        self.samples = ["This is a test sentence for evaluating pruning effects."] * size
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True, padding="max_length", max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(0)


def load_gemma_model(dtype, device):
    LARGE_MODEL_NAME = "Qwen/Qwen3-8B"
    model = AutoModelForCausalLM.from_pretrained(
        LARGE_MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    return model


# ================================================================
# II.  Metrics
# ================================================================
import tempfile
import os

def get_model_size_mb(model):
    model.cpu()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 ** 2)
    os.remove(tmp.name)
    return size_mb


@torch.no_grad()
def evaluate_perplexity(model, loader, device):
    model.to(device)
    model.eval()
    total_loss, total_tokens = 0, 0
    for batch, _ in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        loss = model(input_ids, attention_mask=attn, labels=input_ids).loss.item()
        n_tokens = attn.sum().item()
        total_loss += loss * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(avg_loss)))


@torch.no_grad()
def benchmark_generation_speed(model, prompt, tokenizer, device):
    model.to(device)
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    max_new_tokens = 50

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.time()
        model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.time() - t0

    gen_tokens = max_new_tokens
    return gen_tokens / elapsed  # tokens/sec


# ================================================================
# III.  Experiment
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hardware optimizations on a Hugging Face model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m",
        help="The Hugging Face model to evaluate (e.g., 'google/gemma-3-270m', 'mistralai/Mistral-7B-v0.1')."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"\nRunning Gemma Pruning + SVD Test on {device} ({dtype})")

    # Data
    test_loader = DataLoader(MockLLMDataset(TOKENIZER, size=80), batch_size=4)

    # 1️⃣ Baseline
    print("\n[1/3] Loading baseline model...")
    base_model = load_gemma_model(dtype, device)
    base_size = get_model_size_mb(base_model)
    base_ppl = evaluate_perplexity(base_model, test_loader, device)
    base_tps = benchmark_generation_speed(base_model, PROMPT, TOKENIZER, device)

    # 2️⃣ Pruned Model
    print("\n[2/3] Applying pruning...")
    pruned_model = load_gemma_model(dtype, device)
    pruner = Pruner(pruned_model)
    pruner.prune_magnitude_global(0.3)  # 30% global prune
    pruned_size = get_model_size_mb(pruned_model)
    pruned_ppl = evaluate_perplexity(pruned_model, test_loader, device)
    pruned_tps = benchmark_generation_speed(pruned_model, PROMPT, TOKENIZER, device)

    # 3️⃣ Pruned + SVD
    print("\n[3/3] Applying pruning + SVD decomposition...")
    svd_model = load_gemma_model(dtype, device)
    svd_pruner = PrunerWithSVD(svd_model)
    svd_pruner.prune_magnitude_global(0.3)
    svd_pruner.apply_svd_on_masks(mode="reconstruct_dense", min_rank=4, inplace=True)
    svd_size = get_model_size_mb(svd_model)
    svd_ppl = evaluate_perplexity(svd_model, test_loader, device)
    svd_tps = benchmark_generation_speed(svd_model, PROMPT, TOKENIZER, device)

    # ================================================================
    # 4️⃣ Save final Pruned + SVD model state
    # ================================================================
    OUTPUT_PATH = "gemma_pruned_svd_final.pth"

    # Save the model's state_dict (weights) to a file
    torch.save(svd_model.state_dict(), OUTPUT_PATH)
    print(f"\n✅ Pruned + SVD model saved to: {OUTPUT_PATH}")


    # ================================================================
    # IV.  Report
    # ================================================================
    print("\n" + "=" * 70)
    print("GEMMA PRUNING + SVD COMPARISON REPORT")
    print("=" * 70)
    print(f"| Metric | Baseline | Pruned (30%) | Pruned+SVD | Change vs Baseline |")
    print(f"|:--|--:|--:|--:|--:|")
    print(f"| Model Size (MB) | {base_size:.2f} | {pruned_size:.2f} | {svd_size:.2f} | "
          f"{100*(svd_size-base_size)/base_size:+.1f}% |")
    print(f"| Perplexity | {base_ppl:.3f} | {pruned_ppl:.3f} | {svd_ppl:.3f} | "
          f"{svd_ppl-base_ppl:+.3f} |")
    print(f"| Tokens/sec | {base_tps:.2f} | {pruned_tps:.2f} | {svd_tps:.2f} | "
          f"{svd_tps/base_tps:.2f}× speedup |")
    print("-" * 70)
