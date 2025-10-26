import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
import time
import numpy as np
import os
import wandb
import argparse

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_CONSOLE"] = "off"

# --- Import your model ---
try:
    from attention import SimpleAttentionLM
except ImportError:
    print("Error: Could not import SimpleAttentionLM from attention.py")
    exit()

# --- 1. Config ---
CONFIG = {
    "vocab_size": 50257,
    "block_size": 4096,
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "ff_hidden": 512 * 4,
    "dropout": 0.1,
    "batch_size": 32,  # This will be batch size PER GPU
    "learning_rate": 2e-4,
    "num_epochs": 5,
    "eval_interval": 100,
    "log_interval": 10,
    "device": 'cuda',
    "tokenizer_name": 'EleutherAI/gpt-neo-125M'
}

# --- 2. Data Pipeline (MmapDataset) ---
class MmapTrainDataset(Dataset):
    def __init__(self, bin_file_path, block_size, train_split=0.9, dtype=np.uint16):
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
        super().__init__()
        self.block_size = block_size
        self.dtype = dtype
        file_size_bytes = os.path.getsize(bin_file_path)
        item_size = np.dtype(self.dtype).itemsize
        self.num_tokens = file_size_bytes // item_size
        self.split_idx = int(self.num_tokens * train_split)
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
        try:
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")
            return True, rank, world_size, local_rank
        except Exception as e:
            print(f"Error initializing DDP: {e}")
            return False, 0, 1, 0
    else:
        print("DDP not available or no GPUs found. Running in single-process mode.")
        return False, 0, 1, 0

def cleanup_ddp():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# --- 3. Training & Evaluation Functions ---
@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, is_ddp):
    """Calculates average validation loss."""
    eval_model = model.module if is_ddp else model
    eval_model.eval()
    
    losses = []
    val_iter = iter(val_loader)
    num_eval_batches = 100
    
    for k in range(num_eval_batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        
        x, y = x.to(device), y.to(device)
        logits, _ = eval_model(x, past_kv_caches=None)
        loss = loss_fn(logits.view(-1, eval_model.vocab_size), y.view(-1))
        losses.append(loss.item())
    
    if not losses:
        avg_loss = 0.0
    else:
        avg_loss = sum(losses) / len(losses)
    
    # Aggregate losses across all GPUs
    if is_ddp:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    eval_model.train()
    return avg_loss


# --- 4. Main Training Script ---
if __name__ == "__main__":
    # --- Setup DDP ---
    is_ddp, rank, world_size, local_rank = setup_ddp()
    
    # Set different seed per rank for data loading randomness
    seed = 42 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    device = f'cuda:{local_rank}' if is_ddp else CONFIG['device']
    CONFIG['device'] = device
    
    if rank == 0:
        print(f"World size: {world_size} GPUs")
        print(f"Effective batch size: {CONFIG['batch_size'] * world_size}")
    
    # --- W&B Init (only on rank 0) ---
    if rank == 0:
        wandb.init(
            project="simple-attention-lm-hackathon-ddp",
            config={**CONFIG, "world_size": world_size, "effective_batch_size": CONFIG['batch_size'] * world_size}
        )
        print(f"W&B Run: {wandb.run.name}")
    
    # Synchronize all processes before continuing
    if is_ddp:
        dist.barrier()
    
    DATA_FILE = "train_dataset.bin"
    if not os.path.exists(DATA_FILE):
        if rank == 0:
            print(f"Error: {DATA_FILE} not found.")
            print("Please run build_dataset.py first to create the dataset.")
            if rank == 0:
                wandb.finish()
        cleanup_ddp()
        exit()
    
    # --- Init Tokenizer (all ranks need this) ---
    if rank == 0:
        print(f"Loading '{CONFIG['tokenizer_name']}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Init Data (all ranks need dataset objects) ---
    if rank == 0:
        print("Loading data...")
    
    train_data = MmapTrainDataset(DATA_FILE, CONFIG['block_size'], train_split=0.9)
    val_data = MmapValDataset(DATA_FILE, CONFIG['block_size'], train_split=0.9)
    
    if rank == 0:
        print(f"Total sequences (approx): {(train_data.num_tokens - CONFIG['block_size'] - 1):,}")
        print(f"Train sequences: {len(train_data):,}, Val sequences: {len(val_data):,}")
        total_tokens = 1_400_000_000  # Your dataset size
        effective_batch_size = CONFIG['batch_size'] * world_size
        steps_per_epoch = total_tokens // effective_batch_size
        print(f"Steps per epoch: {steps_per_epoch:,}")
    
    # --- Create Sampler for DDP ---
    train_sampler = DistributedSampler(
        train_data, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True,
        drop_last=True
    ) if is_ddp else None
    
    val_sampler = DistributedSampler(
        val_data, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    ) if is_ddp else None
    
    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=4 if device.startswith('cuda') else 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        pin_memory=True,
        num_workers=4 if device.startswith('cuda') else 0
    )
    
    # --- Init Model ---
    if rank == 0:
        print("Initializing model...")
    
    model_config = {
        "vocab_size": CONFIG['vocab_size'],
        "block_size": CONFIG['block_size'],
        "n_layer": CONFIG['n_layer'],
        "n_head": CONFIG['n_head'],
        "n_embd": CONFIG['n_embd'],
        "ff_hidden": CONFIG['ff_hidden'],
        "dropout": CONFIG['dropout']
    }
    
    model = SimpleAttentionLM(**model_config).to(device)
    
    # --- Wrap Model with DDP ---
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if rank == 0:
            print("Model wrapped with DDP.")
    
    if rank == 0:
        param_model = model.module if is_ddp else model
        param_count = sum(p.numel() for p in param_model.parameters() if p.requires_grad) / 1e6
        print(f"Model parameters: {param_count:.2f}M")
        wandb.watch(model, log='all', log_freq=CONFIG['log_interval'])
        wandb.run.summary["model_parameters_M"] = param_count
        wandb.run.summary["world_size"] = world_size
    
    # --- Init Optimizer & Loss ---
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Synchronize before training
    if is_ddp:
        dist.barrier()
    
    # --- Training Loop ---
    if rank == 0:
        print("Starting training...")
    
    model.train()
    global_step = 0
    
    try:
        for epoch in range(CONFIG['num_epochs']):
            if rank == 0:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
                print(f"{'='*50}")
            
            epoch_start_time = time.time()
            
            # Set epoch for sampler (important for shuffling)
            if is_ddp and train_sampler:
                train_sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                batch_start_time = time.time()
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits, _ = model(x, past_kv_caches=None)
                vocab_size = model.module.vocab_size if is_ddp else model.vocab_size
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Gradient clipping (optional but recommended)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                global_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                
                # Logging
                if global_step % CONFIG['log_interval'] == 0:
                    batch_time = (time.time() - batch_start_time) * 1000
                    
                    if rank == 0:
                        print(f"Step {global_step:6d} | Epoch {epoch+1} | "
                              f"Loss: {loss.item():.4f} | Time: {batch_time:.1f}ms")
                        
                        wandb.log({
                            "step": global_step,
                            "epoch": epoch + 1,
                            "train_loss": loss.item(),
                            "batch_time_ms": batch_time,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })
                
                # Evaluation
                if global_step % CONFIG['eval_interval'] == 0 and global_step > 0:
                    # Synchronize before evaluation
                    if is_ddp:
                        dist.barrier()
                    
                    val_loss = evaluate(model, val_loader, loss_fn, device, is_ddp)
                    
                    if rank == 0:
                        print(f"\n{'='*50}")
                        print(f"Validation at Step {global_step}")
                        print(f"Val Loss: {val_loss:.4f}")
                        print(f"{'='*50}\n")
                        
                        wandb.log({
                            "step": global_step,
                            "epoch": epoch + 1,
                            "val_loss": val_loss
                        })
                    
                    model.train()
                    
                    # Synchronize after evaluation
                    if is_ddp:
                        dist.barrier()
            
            # End of epoch synchronization
            if is_ddp:
                dist.barrier()
            
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # Aggregate epoch loss across ranks
            if is_ddp:
                loss_tensor = torch.tensor(avg_epoch_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_epoch_loss = loss_tensor.item()
            
            if rank == 0:
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Time: {epoch_time:.2f}s")
                print(f"  Avg Loss: {avg_epoch_loss:.4f}")
                print(f"  Batches: {num_batches}")
                
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_time_s": epoch_time,
                    "epoch_avg_loss": avg_epoch_loss
                })
                
                # Save checkpoint
                ckpt_dir = "checkpoints"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")
                
                save_model = model.module if is_ddp else model
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'config': model_config
                }, ckpt_path)
                print(f"Checkpoint saved -> {ckpt_path}\n")
                wandb.save(ckpt_path)
            
            # Synchronize after checkpoint save
            if is_ddp:
                dist.barrier()
        
        if rank == 0:
            print("\n" + "="*50)
            print("Training finished successfully!")
            print("="*50)
        
    except Exception as e:
        if rank == 0:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
        cleanup_ddp()
        if rank == 0:
            wandb.finish()
        raise
    
    # --- Save final model (only rank 0) ---
    if rank == 0:
        model_save_path = "simple_lm_custom_dataset_ddp.pt"
        save_model = model.module if is_ddp else model
        torch.save(save_model.state_dict(), model_save_path)
        print(f"\nFinal model saved to {model_save_path}")
        
        # --- Test Generation (only rank 0) ---
        print("\n" + "="*50)
        print("Testing Generation")
        print("="*50)
        
        gen_model = SimpleAttentionLM(**model_config).to(device)
        gen_model.load_state_dict(torch.load(model_save_path))
        gen_model.eval()
        
        prompt = "Once upon a time there was"
        print(f"Prompt: '{prompt}'")
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = gen_model.generate(
                prompt_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\nGenerated Text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # Log to W&B
        generation_table = wandb.Table(columns=["step", "prompt", "generated_text"])
        generation_table.add_data(global_step, prompt, generated_text)
        wandb.log({"generation_samples": generation_table})
        
        wandb.finish()
        print("\nW&B run finished.")
    
    # --- Cleanup DDP ---
    cleanup_ddp()
    if rank == 0:
        print("\nAll processes cleaned up. Exiting.")