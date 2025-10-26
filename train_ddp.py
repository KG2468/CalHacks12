import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
import time
import numpy as np
import os
import wandb
import sys

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_CONSOLE"] = "off"

# Enable TF32 for B200s - CRITICAL for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# --- Import your model ---
try:
    from attention import SimpleAttentionLM
except ImportError:
    print("Error: Could not import SimpleAttentionLM from attention.py")
    sys.exit(1)

# --- 1. Config ---
CONFIG = {
    "vocab_size": 50257,
    "block_size": 4096,
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "ff_hidden": 512 * 4,
    "dropout": 0.1,
    "batch_size": 32,  # INCREASED for B200s (192GB memory)
    "learning_rate": 2e-4,
    "num_epochs": 5,
    "eval_interval": 500,
    "log_interval": 10,
    "checkpoint_interval": 1000,  # <-- ADDED THIS
    "device": 'cuda',
    "tokenizer_name": 'EleutherAI/gpt-neo-125M',
    "use_amp": True,  # Use BF16 mixed precision
}

# --- 2. Data Pipeline ---
class MmapTrainDataset(Dataset):
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


def cleanup():
    """End DDP training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# --- 3. Evaluation Function ---
@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, rank):
    """Calculates average validation loss."""
    model.eval()
    
    score = torch.tensor(0.0, device=device)
    n_samples = torch.tensor(0, device=device)
    
    num_eval_batches = min(100, len(val_loader))
    
    for k, (x, y) in enumerate(val_loader):
        if k >= num_eval_batches:
            break
        
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        try:
            # Get the vocab_size from the unwrapped model
            vocab_size = model.module.vocab_size if hasattr(model, 'module') else model.vocab_size
            logits, _ = model(x, past_kv_caches=None)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            score += loss
            n_samples += 1
        except Exception as e:
            if rank == 0:
                print(f"Error during evaluation: {e}", flush=True)
            break
    
    # Reduce across all GPUs
    dist.all_reduce(score, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
    
    avg_loss = (score / n_samples).item() if n_samples > 0 else 0.0
    
    model.train()
    return avg_loss


# --- 4. Main Training Script ---
def main():
    # Setup DDP
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    dist.init_process_group(backend='nccl', init_method='env://')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    seed = 42 * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    if rank == 0:
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {CONFIG['batch_size']}")
        print(f"Effective batch size: {CONFIG['batch_size'] * world_size}")
    
    # Check data file exists
    DATA_FILE = "train_dataset.bin"
    if not os.path.exists(DATA_FILE):
        if rank == 0:
            print(f"Error: {DATA_FILE} not found.")
        cleanup()
        sys.exit(1)
    
    # W&B Init (only rank 0)
    if rank == 0:
        try:
            wandb.init(
                project="simple-attention-lm-hackathon-ddp",
                config={**CONFIG, "world_size": world_size}
            )
            print(f"W&B Run: {wandb.run.name}")
        except Exception as e:
            print(f"W&B init failed: {e}")
    
    # Init Tokenizer
    if rank == 0:
        print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Init Data
    if rank == 0:
        print("Loading data...")
    
    train_data = MmapTrainDataset(DATA_FILE, CONFIG['block_size'], train_split=0.9)
    val_data = MmapValDataset(DATA_FILE, CONFIG['block_size'], train_split=0.9)
    
    if rank == 0:
        print(f"Train sequences: {len(train_data):,}, Val sequences: {len(val_data):,}")
        # --- This calculation seems off, let's use len(train_loader) ---
        # total_tokens = 1_400_000_000 
        # effective_batch_size = CONFIG['batch_size'] * world_size
        # steps_per_epoch = total_tokens // effective_batch_size
        # print(f"Steps per epoch (approx): {steps_per_epoch:,}")
        
    # Create Samplers
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed
    )
    
    val_sampler = DistributedSampler(
        val_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        sampler=train_sampler,
        num_workers=12,  # INCREASED for 8 GPUs
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,  # Prefetch more batches
        persistent_workers=True  # Keep workers alive
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    if rank == 0:
        steps_per_epoch = len(train_loader)
        print(f"Steps per epoch (from DataLoader): {steps_per_epoch:,}")
        
    # Init Model
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
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Model parameters: {param_count:.2f}M")
    
    # Wrap with DDP
    model = DDP(model, device_ids=[device])
    
    if rank == 0:
        try:
            wandb.watch(model, log='all', log_freq=CONFIG['log_interval'])
        except:
            pass
    
    # Init Optimizer & Loss
    optimizer = AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=0,
        fused=True  # Fused optimizer for B200s
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # Setup gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['use_amp'])
    
    # Training Loop
    if rank == 0:
        print("\nStarting training...")
        print("="*70)
    
    model.train()
    train_steps = 0
    running_loss = 0
    log_steps = 0
    start_time = time.time()
    
    try:
        for epoch in range(CONFIG['num_epochs']):
            train_sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
                print("-"*70)
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                # BF16 autocast for forward pass
                with torch.amp.autocast('cuda', enabled=CONFIG['use_amp'], dtype=torch.bfloat16):
                    logits, _ = model(x, past_kv_caches=None)
                    loss = loss_fn(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
                
                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step optimizer
                scaler.step(optimizer)
                scaler.update()
                
                # Accumulate loss
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                
                # Logging
                if train_steps % CONFIG['log_interval'] == 0:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    
                    # Reduce loss across all processes
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                    
                    if rank == 0:
                        print(f"Step {train_steps:6d} | Loss: {avg_loss:.4f} | Steps/Sec: {steps_per_sec:.2f}", flush=True)
                        
                        try:
                            wandb.log({
                                "step": train_steps,
                                "epoch": epoch + 1,
                                "train_loss": avg_loss,
                                "steps_per_sec": steps_per_sec,
                                "learning_rate": optimizer.param_groups[0]['lr']
                            })
                        except:
                            pass
                    
                    # Reset monitoring variables
                    running_loss = 0
                    log_steps = 0
                    start_time = time.time()
                
                # Evaluation
                if train_steps % CONFIG['eval_interval'] == 0 and train_steps > 0:
                    eval_start_time = time.time()
                    
                    val_loss = evaluate(model, val_loader, loss_fn, device, rank)
                    
                    # Synchronize after evaluation
                    dist.barrier()
                    
                    eval_time = time.time() - eval_start_time
                    
                    if rank == 0:
                        print(f"\nValidation at Step {train_steps}")
                        print(f"Val Loss: {val_loss:.4f} | Eval Time: {eval_time:.2f}s\n", flush=True)
                        
                        try:
                            wandb.log({
                                "step": train_steps,
                                "epoch": epoch + 1,
                                "val_loss": val_loss,
                                "eval_time": eval_time
                            })
                        except:
                            pass
                    
                    model.train()
                    start_time = time.time()

                # --- CHECKPOINTING MODIFICATION ---
                if train_steps % CONFIG['checkpoint_interval'] == 0 and train_steps > 0:
                    # Save checkpoint only on rank 0
                    if rank == 0:
                        try:
                            ckpt_dir = "checkpoints"
                            os.makedirs(ckpt_dir, exist_ok=True)
                            # Change filename to use step
                            ckpt_path = os.path.join(ckpt_dir, f"model_step_{train_steps}.pt")
                            
                            # Unwrap DDP model for saving
                            model_to_save = model.module if hasattr(model, 'module') else model
                            
                            torch.save({
                                'epoch': epoch + 1,
                                'train_steps': train_steps,
                                'model_state_dict': model_to_save.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scaler_state_dict': scaler.state_dict(), # Also save scaler state
                                'config': model_config
                            }, ckpt_path)
                            print(f"\n[Checkpoint] Saved at step {train_steps}: {ckpt_path}\n", flush=True)
                            
                            try:
                                wandb.save(ckpt_path)
                            except:
                                pass
                        except Exception as e:
                            print(f"Error saving checkpoint at step {train_steps}: {e}", flush=True)
                # --- END MODIFICATION ---
            
            # End of epoch
            if rank == 0:
                print(f"\nEpoch {epoch+1} complete\n")
                
                # --- REMOVED EPOCH CHECKPOINTING LOGIC FROM HERE ---
        
        if rank == 0:
            print("\n" + "="*70)
            print("Training completed!")
            print("="*7Example 30)
        
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"[Rank {rank}] Error during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # Save final model (only rank 0)
        if rank == 0:
            try:
                model_save_path = "simple_lm_custom_dataset_ddp.pt"
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), model_save_path)
                print(f"\nFinal model saved to {model_save_path}")
                
                # Test Generation
                print("\n" + "="*70)
                print("Testing Generation")
                print("="*70)
                
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
                print("-" * 70)
                print(generated_text)
                print("-" * 70)
                
                try:
                    generation_table = wandb.Table(columns=["step", "prompt", "generated_text"])
                    generation_table.add_data(train_steps, prompt, generated_text)
                    wandb.log({"generation_samples": generation_table})
                    wandb.finish()
                except:
                    pass
                
            except Exception as e:
                print(f"Error during final save/generation: {e}")
        
        # Cleanup
        print(f"[Rank {rank}] Cleaning up...", flush=True)
        cleanup()


if __name__ == "__main__":
    main()