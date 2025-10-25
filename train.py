import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
import time
import numpy as np
import os
import wandb  # <-- W&B: Add this import

# --- Import your model from the other file ---
try:
    from attention import SimpleAttentionLM
except ImportError:
    print("Error: Could not import SimpleAttentionLM from attention.py")
    print("Please make sure attention.py is in the same directory and is corrected.")
    exit()

# --- 1. Config ---
# We move these into a 'config' dict for W&B
CONFIG = {
    "vocab_size": 50257,  # CRITICAL: This is the vocab size for EleutherAI/gpt-neo-125M
    "block_size": 4096,
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "ff_hidden": 512 * 4, # <-- W&B: Explicitly define this
    "dropout": 0.1,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "num_epochs": 1,
    "eval_interval": 100,
    "log_interval": 10,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "tokenizer_name": 'EleutherAI/gpt-neo-125M' # <-- W&B: Good to log this
}

# --- 2. Data Pipeline (MmapDataset) ---
# (This class is unchanged)
# --- 2. Data Pipeline (MmapDataset) ---
# (This class is UNCHANGED... except it is replaced by this new one)
class MmapDataset(Dataset):
    def __init__(self, bin_file_path, block_size, dtype=np.uint16, offset_tokens=0, max_tokens=None):
        """
        MmapDataset
        :param bin_file_path: Path to the .bin file
        :param block_size: Sequence length
        :param dtype: Numpy dtype of the tokens (e.g., np.uint16)
        :param offset_tokens: (NEW) Number of tokens to *skip* from the start of the file
        :param max_tokens: (NEW) Maximum number of tokens to load *after* the offset.
                           If None, loads all tokens from the offset to the end.
        """
        super().__init__()
        self.block_size = block_size
        self.dtype = dtype
        item_size = np.dtype(self.dtype).itemsize

        file_size_bytes = os.path.getsize(bin_file_path)
        total_tokens_in_file = file_size_bytes // item_size

        if file_size_bytes % item_size != 0:
            raise ValueError("File size is not a multiple of item size!")
        
        # Calculate the number of tokens to use
        if max_tokens is None:
            # Use all tokens from the offset to the end
            self.num_tokens = total_tokens_in_file - offset_tokens
        else:
            # Use 'max_tokens', but don't go past the end of the file
            self.num_tokens = min(max_tokens, total_tokens_in_file - offset_tokens)

        # Calculate the byte offset
        byte_offset = offset_tokens * item_size
        
        # Define the shape of the mmap array
        mmap_shape = (self.num_tokens,)

        print(f"Memory-mapping {bin_file_path}...")
        print(f"  - Using {self.num_tokens:,} tokens")
        print(f"  - Starting from token {offset_tokens:,} (byte offset {byte_offset:,})")

        self.mmap = np.memmap(
            bin_file_path, 
            dtype=self.dtype, 
            mode='r',
            offset=byte_offset, # <-- Use the byte offset
            shape=mmap_shape    # <-- Use the specific shape
        )
        
        if self.num_tokens < block_size + 1:
            raise ValueError(f"Dataset partition is too small ({self.num_tokens} tokens) for block_size ({block_size}).")

    def __len__(self):
        # The number of *sequences* we can make
        return self.num_tokens - self.block_size - 1

    def __getitem__(self, idx):
        # This logic remains the same
        chunk = self.mmap[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

# --- 3. Training & Evaluation Functions ---
@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    for k in range(100): 
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(CONFIG['device']), y.to(CONFIG['device']) # <-- W&B: Use config
        logits, _ = model(x, past_kv_caches=None)
        loss = loss_fn(logits.view(-1, CONFIG['vocab_size']), y.view(-1)) # <-- W&B: Use config
        losses.append(loss.item())
    model.train()
    if not losses:
        return 0.0
    return torch.tensor(losses).mean().item()

# --- 4. Main Training Script ---

if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}") # <-- W&B: Use config
    
    # --- W&B: Initialize Run ---
    wandb.init(
        project="simple-attention-lm-hackathon", # <-- W&B: Name your project
        config=CONFIG                          # <-- W&B: Pass in your config dict
    )
    # W&B will create a random run name (e.g., "gentle-brook-5")
    print(f"W&B Run: {wandb.run.name}")

    DATA_FILE = "train_dataset.bin"
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        print("Please run build_dataset.py first to create the dataset.")
        wandb.finish() # <-- W&B: Stop the run if data is missing
        exit()

    # --- Init Tokenizer ---
    print(f"Loading '{CONFIG['tokenizer_name']}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Init Data ---
    print("Loading data...")

    # --- THIS IS THE CHANGE YOU REQUESTED ---
    TOKENS_PART_1 = 476616445 
    
    # Load *only* the first part of the dataset
    print("Loading Dataset Part 1...")
    dataset_part_1 = MmapDataset(
        DATA_FILE, 
        CONFIG['block_size'], 
        dtype=np.uint16,
        offset_tokens=0,           # Start from the beginning
        max_tokens=TOKENS_PART_1   # Load only this many tokens
    )
    
    # (Optional) To load the *second* part later (e.g., for another training run),
    # you would just change the parameters like this:
    # print("Loading Dataset Part 2...")
    # dataset_part_2 = MmapDataset(
    #     DATA_FILE, 
    #     CONFIG['block_size'], 
    #     dtype=np.uint16,
    #     offset_tokens=TOKENS_PART_1, # Start *after* part 1
    #     max_tokens=None              # Load all remaining tokens
    # )
    # --- End of change ---

    
    # Now, we split dataset_part_1 into train/val
    # The rest of the script will *only* train on this first part.
    n = len(dataset_part_1) 
    train_n = int(n * 0.9)
    val_n = n - train_n
    
    # Handle the case where the validation set is empty
    if val_n == 0:
        print("Warning: Validation split is 0. Using entire partition for training.")
        train_n = n
        train_data = dataset_part_1
        # Create a dummy val_data to avoid errors, though it won't be used meaningfully
        # Or, better, adjust the evaluation logic
        # For simplicity, we'll just split 90/10 even if val_n is tiny
        if n > 10: # Ensure we have enough data to split
            train_n = int(n * 0.9)
            val_n = n - train_n
        else:
            train_n = n
            val_n = 0 # No validation
            
    if val_n > 0:
        train_data, val_data = torch.utils.data.random_split(
            dataset_part_1, # <-- Use the partial dataset
            [train_n, val_n],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_data = dataset_part_1
        val_data = None # Handle this in your eval loop
    
    print(f"Total sequences (from Part 1): {n:,}")
    print(f"Train sequences: {len(train_data):,}, Val sequences: {len(val_data) if val_data else 0:,}")
    
    train_loader = DataLoader(
        train_data, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4 if CONFIG['device'] == 'cuda' else 0
    )
    
    # Only create a val_loader if val_data exists
    val_loader = None
    if val_data:
        val_loader = DataLoader(
            val_data, 
            batch_size=CONFIG['batch_size'], 
            pin_memory=True, 
            num_workers=4 if CONFIG['device'] == 'cuda' else 0
        )

    # --- Init Model ---
    print("Initializing model...")
    # Use the 'config' dict for model init
    model = SimpleAttentionLM(
        vocab_size=CONFIG['vocab_size'],
        block_size=CONFIG['block_size'],
        n_layer=CONFIG['n_layer'],
        n_head=CONFIG['n_head'],
        n_embd=CONFIG['n_embd'],
        ff_hidden=CONFIG['ff_hidden'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # --- W&B: Watch Model ---
    # This logs gradients, parameters, and model architecture
    wandb.watch(model, log='all', log_freq=CONFIG['log_interval'])
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model parameters: {param_count:.2f}M")
    wandb.run.summary["model_parameters_M"] = param_count # <-- W&B: Save param count

    # --- Init Optimizer & Loss ---
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training...")
    model.train()
    step = 0
    for epoch in range(CONFIG['num_epochs']):
        print(f"--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        epoch_start_time = time.time()
        
        for i, (x, y) in enumerate(train_loader):
            batch_start_time = time.time()
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            logits, _ = model(x, past_kv_caches=None)
            loss = loss_fn(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            if step % CONFIG['log_interval'] == 0:
                batch_time = (time.time() - batch_start_time) * 1000 # in ms
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}ms")
                
                # --- W&B: Log training metrics ---
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "batch_time_ms": batch_time
                })
            
            if step % CONFIG['eval_interval'] == 0 and step > 0:
                val_loss = evaluate(model, val_loader, loss_fn)
                print(f"--- Validation ---")
                print(f"Step {step} | Val Loss: {val_loss:.4f}")
                print(f"--------------------")
                model.train()
                
                # --- W&B: Log validation metrics ---
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "val_loss": val_loss
                })

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")
        
    print("Training finished.")
    
    # --- Save the model ---
    model_save_path = "simple_lm_custom_dataset.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # --- Test Generation ---
    print("\n--- Testing Generation ---")
    model.load_state_dict(torch.load(model_save_path)) # Load weights
    
    prompt = "Once upon a time there was"
    print(f"Prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(CONFIG['device'])
    
    generated_ids = model.generate(
        prompt_ids, 
        max_new_tokens=50, 
        temperature=0.8, 
        top_k=50
    )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n--- Generated Text ---")
    print(generated_text)
    
    # --- W&B: Log generated text sample as a Table ---
    generation_table = wandb.Table(columns=["step", "prompt", "generated_text"])
    generation_table.add_data(step, prompt, generated_text)
    wandb.log({"generation_samples": generation_table})

    # --- W&B: Finish the run ---
    wandb.finish()
    print("W&B run finished.")