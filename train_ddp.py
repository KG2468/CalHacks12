import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
import time
import numpy as np
import os
import wandb
import argparse # <-- Add argparse for potential command-line args

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# --------------------

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_MODE"] = "online"       # ensure online sync
os.environ["WANDB_CONSOLE"] = "off"       # prevent stdout blocking

# --- Import your model ---
try:
    from attention import SimpleAttentionLM
except ImportError:
    print("Error: Could not import SimpleAttentionLM from attention.py")
    exit()

# --- 1. Config ---
# No changes here, keep your CONFIG dict as is
CONFIG = {
    "vocab_size": 50257,
    "block_size": 4096,
    "n_layer": 6,
    "n_head": 10,
    "n_embd": 640,
    "ff_hidden": 512 * 4,
    "dropout": 0.1,
    "batch_size": 32, # This will be batch size PER GPU
    "learning_rate": 2e-4,
    "num_epochs": 5,
    "eval_interval": 100,
    "log_interval": 10,
    "device": 'cuda', # We'll determine the specific GPU later
    "tokenizer_name": 'EleutherAI/gpt-neo-125M'
}

# --- 2. Data Pipeline (MmapDataset) ---
# Use the separate Train/Val datasets from the previous fix
# (MmapTrainDataset and MmapValDataset classes are unchanged)
class MmapTrainDataset(Dataset):
    def __init__(self, bin_file_path, block_size, train_split=0.9, dtype=np.uint16):
        # ... (implementation as before) ...
        super().__init__()
        self.block_size = block_size
        self.dtype = dtype
        file_size_bytes = os.path.getsize(bin_file_path)
        item_size = np.dtype(self.dtype).itemsize
        self.num_tokens = file_size_bytes // item_size
        self.split_idx = int(self.num_tokens * train_split)
        print(f"Memory-mapping {bin_file_path} ({self.num_tokens:,} tokens)")
        self.mmap = np.memmap(bin_file_path, dtype=self.dtype, mode='r')

    def __len__(self):
        return self.split_idx - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.mmap[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

class MmapValDataset(Dataset):
    def __init__(self, bin_file_path, block_size, train_split=0.9, dtype=np.uint16):
        # ... (implementation as before) ...
        super().__init__()
        self.block_size = block_size
        self.dtype = dtype
        file_size_bytes = os.path.getsize(bin_file_path)
        item_size = np.dtype(self.dtype).itemsize
        self.num_tokens = file_size_bytes // item_size
        self.split_idx = int(self.num_tokens * train_split)
        # Re-map only if needed, or share the map if possible (advanced)
        # For simplicity, we re-map here
        # print(f"Memory-mapping {bin_file_path} for validation")
        self.mmap = np.memmap(bin_file_path, dtype=self.dtype, mode='r')

    def __len__(self):
        return (self.num_tokens - self.split_idx) - self.block_size - 1

    def __getitem__(self, idx):
        actual_idx = idx + self.split_idx
        chunk = self.mmap[actual_idx : actual_idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# --- DDP Setup Function ---
def setup_ddp():
    """Initializes the distributed environment."""
    if dist.is_available() and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        dist.init_process_group(backend='nccl') # NCCL is recommended for NVIDIA GPUs
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        print(f"Rank {dist.get_rank()} initialized on GPU {local_rank}")
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    else:
        print("DDP not available or no GPUs found. Running in single-process mode.")
        return False, 0, 1, 0 # rank, world_size, local_rank for non-DDP

def cleanup_ddp():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
# -------------------------


# --- 3. Training & Evaluation Functions ---
@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, is_ddp): # <-- Added device, is_ddp
    """Calculates average validation loss."""
    # If using DDP, model is already wrapped. Access original model via model.module
    eval_model = model.module if is_ddp else model
    eval_model.eval()
    losses = []
    val_iter = iter(val_loader)
    num_eval_batches = 100 # Evaluate on 100 batches per process

    for k in range(num_eval_batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device) # <-- Use assigned device
        logits, _ = eval_model(x, past_kv_caches=None) # <-- Use eval_model
        loss = loss_fn(logits.view(-
                    1, eval_model.vocab_size), y.view(-1)) # <-- Use eval_model
        losses.append(loss.item())

    eval_model.train() # Set model back to training mode

    if not losses:
        return 0.0

    # --- Aggregate losses across all GPUs ---
    avg_loss = torch.tensor(losses).mean().item()
    if is_ddp:
        # Need to gather losses from all ranks
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    # ------------------------------------------

    return avg_loss

# --- 4. Main Training Script ---

if __name__ == "__main__":
    # --- Setup DDP ---
    is_ddp, rank, world_size, local_rank = setup_ddp()
    device = f'cuda:{local_rank}' if is_ddp else CONFIG['device'] # Assign GPU
    # Update CONFIG for clarity, though device is passed explicitly now
    CONFIG['device'] = device
    print(f"Rank {rank} using device: {device}")
    # Adjust batch size for DDP: total_batch_size = batch_size_per_gpu * world_size
    # CONFIG['total_batch_size'] = CONFIG['batch_size'] * world_size

    # --- W&B Init (only on rank 0) ---
    if rank == 0:
        wandb.init(
            project="simple-attention-lm-hackathon-ddp", # <-- New project name
            config=CONFIG
        )
        print(f"W&B Run: {wandb.run.name}")
    
    if rank == 0:
        print("Testing W&B log right after init...", flush=True)
        wandb.log({"sanity_loss": 123})
        time.sleep(5)
    # ----------------------------------

    DATA_FILE = "train_dataset.bin"
    if not os.path.exists(DATA_FILE):
        if rank == 0:
            print(f"Error: {DATA_FILE} not found.")
            print("Please run build_dataset.py first to create the dataset.")
            wandb.finish()
        cleanup_ddp() # Clean up DDP even on error
        exit()

    # --- Init Tokenizer (all ranks need this) ---
    if rank == 0: print(f"Loading '{CONFIG['tokenizer_name']}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Init Data (all ranks need dataset objects) ---
    if rank == 0: print("Loading data...")
    train_data = MmapTrainDataset(DATA_FILE, CONFIG['block_size'], train_split=0.9)
    val_data = MmapValDataset(DATA_FILE, CONFIG['block_size'], train_split=0.9)

    if rank == 0:
        print(f"Total sequences (approx): {(train_data.num_tokens - CONFIG['block_size'] - 1):,}")
        print(f"Train sequences: {len(train_data):,}, Val sequences: {len(val_data):,}")

    # --- Create Sampler for DDP ---
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    # Validation sampler often not needed unless validation is huge/slow
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    # -----------------------------

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'], # Batch size per GPU
        # shuffle=True, <-- Sampler handles shuffling in DDP
        sampler=train_sampler, # <-- Use sampler
        shuffle=(train_sampler is None), # Shuffle only if not using sampler
        pin_memory=True,
        num_workers=4 if device.startswith('cuda') else 0
    )
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['batch_size'], # Batch size per GPU
        sampler=val_sampler, # <-- Use sampler for validation too (optional but safer)
        shuffle=False,
        pin_memory=True,
        num_workers=4 if device.startswith('cuda') else 0
    )

    # --- Init Model ---
    if rank == 0: print("Initializing model...")
    model_config = { # Recreate config dict for clarity
        "vocab_size": CONFIG['vocab_size'],
        "block_size": CONFIG['block_size'],
        "n_layer": CONFIG['n_layer'],
        "n_head": CONFIG['n_head'],
        "n_embd": CONFIG['n_embd'],
        "ff_hidden": CONFIG['ff_hidden'],
        "dropout": CONFIG['dropout']
    }
    model = SimpleAttentionLM(**model_config).to(device) # <-- Move to assigned device

    # --- Wrap Model with DDP ---
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        if rank == 0: print("Model wrapped with DDP.")
    # ---------------------------

    if rank == 0: # Only rank 0 logs W&B watch and param count
        wandb.watch(model, log='all', log_freq=CONFIG['log_interval'])
        # Access original model parameters if wrapped
        param_model = model.module if is_ddp else model
        param_count = sum(p.numel() for p in param_model.parameters() if p.requires_grad) / 1e6
        print(f"Model parameters: {param_count:.2f}M")
        wandb.run.summary["model_parameters_M"] = param_count
        wandb.run.summary["world_size"] = world_size

    # --- Init Optimizer & Loss ---
    # Give DDP model parameters to optimizer
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    if rank == 0: print("Starting training...")
    model.train()
    step = 0 # Global step counter
    for epoch in range(CONFIG['num_epochs']):
        if rank == 0: print(f"--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        epoch_start_time = time.time()

        # --- Set epoch for sampler (important for shuffling) ---
        if is_ddp and train_sampler:
            train_sampler.set_epoch(epoch)
        # --------------------------------------------------------

        for i, (x, y) in enumerate(train_loader):
            batch_start_time = time.time()
            x, y = x.to(device), y.to(device) # <-- Use assigned device

            # DDP handles forward/backward sync automatically
            logits, _ = model(x, past_kv_caches=None)
            # Access vocab_size via module if DDP wrapped
            vocab_size = model.module.vocab_size if is_ddp else model.vocab_size
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1 # Note: This step counter increments on EACH process

            # Use global step based on rank 0 or average across ranks if needed
            # For logging, using rank 0's step count is usually sufficient
            global_step = step * world_size # Approximate global step

            if step % CONFIG['log_interval'] == 0:
                batch_time = (time.time() - batch_start_time) * 1000 # in ms
                if rank == 0: # Only rank 0 prints logs
                    print(f"Rank {rank} Step {step} (Global ~{global_step}) | Loss: {loss.item():.4f} | Time: {batch_time:.2f}ms")

                # --- W&B Log (only rank 0) ---
                if rank == 0:
                    wandb.log({
                        "step": global_step, # Log estimated global step
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        "batch_time_ms": batch_time,
                        "learning_rate": optimizer.param_groups[0]['lr'] # Log LR
                    })
                # ------------------------------

            if step % CONFIG['eval_interval'] == 0 and step > 0:
                # --- Validation (all ranks participate, rank 0 logs) ---
                val_loss = evaluate(model, val_loader, loss_fn, device, is_ddp)
                if rank == 0:
                    print(f"--- Validation ---")
                    print(f"Rank {rank} Step {step} (Global ~{global_step}) | Val Loss: {val_loss:.4f}")
                    print(f"--------------------")
                    wandb.log({
                        "step": global_step,
                        "epoch": epoch,
                        "val_loss": val_loss
                    })
                # ----------------------------------------------------
                model.train() # Ensure model is back in training mode

        # --- Sync at end of epoch ---
        if is_ddp:
            dist.barrier() # Wait for all processes to finish epoch

        epoch_time = time.time() - epoch_start_time
        if rank == 0:
            print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")

            # --- Save model checkpoint every epoch ---
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")

            save_model = model.module if is_ddp else model
            torch.save(save_model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model checkpoint -> {ckpt_path}")

            # Optionally sync with W&B
            wandb.save(ckpt_path)

            # Optional: limit number of checkpoints to keep (e.g., last 3)
            # MAX_CHECKPOINTS = 3
            # ckpts = sorted(
            #     [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch_")]
            # )
            # if len(ckpts) > MAX_CHECKPOINTS:
            #     old_ckpt = os.path.join(ckpt_dir, ckpts[0])
            #     os.remove(old_ckpt)
            #     print(f"[Checkpoint] Deleted old checkpoint -> {old_ckpt}")

    if rank == 0: print("Training finished.")

    # --- Save the model (only rank 0) ---
    if rank == 0:
        model_save_path = "simple_lm_custom_dataset_ddp.pt"
        # Save the underlying model state dict
        save_model = model.module if is_ddp else model
        torch.save(save_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # --- Test Generation (only rank 0) ---
        print("\n--- Testing Generation ---")
        # Load state dict into the *original* model structure, not the DDP wrapper
        gen_model = SimpleAttentionLM(**model_config).to(device)
        gen_model.load_state_dict(torch.load(model_save_path))
        gen_model.eval() # Set to eval mode for generation

        prompt = "Once upon a time there was"
        print(f"Prompt: '{prompt}'")
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad(): # Ensure no gradients are computed
             generated_ids = gen_model.generate(
                prompt_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n--- Generated Text ---")
        print(generated_text)

        # --- W&B Log Generation ---
        generation_table = wandb.Table(columns=["step", "prompt", "generated_text"])
        generation_table.add_data(global_step, prompt, generated_text)
        wandb.log({"generation_samples": generation_table})

        # --- W&B Finish ---
        wandb.finish()
        print("W&B run finished.")
    # ----------------------------------------

    # --- Cleanup DDP ---
    cleanup_ddp()
    # ------------------