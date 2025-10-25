import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
# from datasets import load_dataset # <-- No longer needed
import time
import numpy as np # <-- Add numpy
import os # <-- Add os

# --- Import your model from the other file ---
# Make sure attention.py is in the same directory
try:
    from attention import SimpleAttentionLM
except ImportError:
    print("Error: Could not import SimpleAttentionLM from attention.py")
    print("Please make sure attention.py is in the same directory and is corrected.")
    exit()

# --- 1. Config ---
VOCAB_SIZE = 50257  # CRITICAL: This is the vocab size for EleutherAI/gpt-neo-125M
BLOCK_SIZE = 256    # Context window size
N_LAYER = 6         # Number of transformer blocks
N_HEAD = 8          # Number of attention heads
N_EMBD = 512        # Embedding dimension
DROPOUT = 0.1
BATCH_SIZE = 32     # How many sequences to process in parallel
LEARNING_RATE = 3e-4 # Optimizer learning rate
NUM_EPOCHS = 1      # How many times to go over the full dataset
EVAL_INTERVAL = 100 # How often to run evaluation (in steps)
LOG_INTERVAL = 10   # How often to print training loss (in steps)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Data Pipeline (NEW) ---
# We replace TextDataset and get_data_loaders with MmapDataset

class MmapDataset(Dataset):
    """
    A Dataset that memory-maps a large binary file of token IDs.
    This is very fast and uses almost no RAM.
    """
    def __init__(self, bin_file_path, block_size, dtype=np.uint16):
        super().__init__()
        self.block_size = block_size
        self.dtype = dtype
        
        # Get file size to calculate length
        file_size_bytes = os.path.getsize(bin_file_path)
        item_size = np.dtype(self.dtype).itemsize
        if file_size_bytes % item_size != 0:
            raise ValueError("File size is not a multiple of item size!")
            
        self.num_tokens = file_size_bytes // item_size
        
        # Memory-map the file
        print(f"Memory-mapping {bin_file_path} ({self.num_tokens:,} tokens)")
        self.mmap = np.memmap(bin_file_path, dtype=self.dtype, mode='r')

    def __len__(self):
        # -1 because we need a target for each input
        return self.num_tokens - self.block_size - 1

    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) tokens
        chunk = self.mmap[idx : idx + self.block_size + 1]
        
        # Convert numpy slice to torch tensor
        # .astype(np.int64) is important because torch.long is 64-bit
        # and CrossEntropyLoss expects long tensors.
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

# --- 3. Training & Evaluation Functions ---
@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    """Calculates average validation loss."""
    model.eval() # Set model to evaluation mode
    losses = []
    # --- Create a smaller loop for validation ---
    # We don't want to evaluate the *entire* val split, just a sample
    val_iter = iter(val_loader)
    for k in range(100): # Evaluate on 100 batches
        try:
            x, y = next(val_iter)
        except StopIteration:
            break # Stop if val loader is exhausted
            
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x, past_kv_caches=None)
        loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
        losses.append(loss.item())
        
    model.train() # Set model back to training mode
    if not losses:
        return 0.0
    return torch.tensor(losses).mean().item()

# --- 4. Main Training Script ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    DATA_FILE = "train_dataset.bin"
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        print("Please run build_dataset.py first to create the dataset.")
        exit()

    # --- Init Tokenizer ---
    print("Loading 'EleutherAI/gpt-neo-125M' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Init Data (NEW) ---
    print("Loading data...")
    # Create one big MmapDataset
    full_dataset = MmapDataset(DATA_FILE, BLOCK_SIZE, dtype=np.uint16)
    
    # Create a 90/10 train/val split
    n = len(full_dataset)
    train_n = int(n * 0.9)
    val_n = n - train_n
    train_data, val_data = torch.utils.data.random_split(
        full_dataset, 
        [train_n, val_n],
        generator=torch.Generator().manual_seed(42) # for reproducibility
    )
    
    print(f"Total sequences: {n:,}")
    print(f"Train sequences: {len(train_data):,}, Val sequences: {len(val_data):,}")
    
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4 if DEVICE == 'cuda' else 0
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        pin_memory=True, 
        num_workers=4 if DEVICE == 'cuda' else 0
    )

    # --- Init Model ---
    print("Initializing model...")
    model_config = {
        "vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "n_embd": N_EMBD,
        "ff_hidden": N_EMBD * 4,
        "dropout": DROPOUT
    }
    model = SimpleAttentionLM(**model_config).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # --- Init Optimizer & Loss ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print("Starting training...")
    model.train()
    step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        epoch_start_time = time.time()
        
        for i, (x, y) in enumerate(train_loader):
            batch_start_time = time.time()
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits, _ = model(x, past_kv_caches=None)
            loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            if step % LOG_INTERVAL == 0:
                batch_time = (time.time() - batch_start_time) * 1000 # in ms
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}ms")
            
            if step % EVAL_INTERVAL == 0 and step > 0:
                val_loss = evaluate(model, val_loader, loss_fn)
                print(f"--- Validation ---")
                print(f"Step {step} | Val Loss: {val_loss:.4f}")
                print(f"--------------------")
                model.train()

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
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(
        prompt_ids, 
        max_new_tokens=50, 
        temperature=0.8, 
        top_k=50
    )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n--- Generated Text ---")
    print(generated_text)

