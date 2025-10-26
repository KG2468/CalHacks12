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
    print("Please make sure attention.py is in the same directory.")
    # We won't exit here in case the user is just reading the file
    # exit()

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
    "checkpoint_interval": 1000, # <-- NEW: Save checkpoint every N steps
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "tokenizer_name": 'EleutherAI/gpt-neo-125M' # <-- W&B: Good to log this
}

# --- 2. Data Pipeline (MmapDataset) ---
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
        # We need block_size + 1 tokens for one (x, y) pair
        return self.num_tokens - self.block_size

    def __getitem__(self, idx):
        # This logic remains the same
        # Get a chunk of block_size + 1 tokens
        chunk = self.mmap[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

# --- 3. Training & Evaluation Functions ---
@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    losses = []
    # Check if val_loader is None (in case of no validation split)
    if val_loader is None:
        model.train()
        return 0.0
        
    val_iter = iter(val_loader)
    # Run for a fixed number of eval steps to speed up validation
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
        config=CONFIG                      # <-- W&B: Pass in your config dict
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
    
    val_data = None
    
    # Handle the case where the validation set is too small or zero
    if val_n < 10: # If val split is tiny, just use everything for training
        print(f"Warning: Validation split is very small ({val_n}). Using entire partition for training.")
        train_n = n
        val_n = 0
        train_data = dataset_part_1
        val_data = None # Explicitly set to None
    else:
        train_data, val_data = torch.utils.data.random_split(
            dataset_part_1, # <-- Use the partial dataset
            [train_n, val_n],
            generator=torch.Generator().manual_seed(42)
        )
    
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
    # Make sure the SimpleAttentionLM class definition is available (from attention.py)
    if 'SimpleAttentionLM' in globals():
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
    else:
        print("Error: SimpleAttentionLM class not found. Skipping model initialization.")
        print("Please ensure attention.py is imported correctly.")
        wandb.finish()
        exit()


    # --- Init Optimizer & Loss ---
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    # --- W&B: Create a checkpoint directory for this run ---
    checkpoint_dir = f"checkpoints/{wandb.run.name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

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

            # --- NEW: Checkpointing Logic ---
            if step % CONFIG['checkpoint_interval'] == 0 and step > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"--- Checkpoint Saved ---")
                print(f"Step {step} | Model saved to {checkpoint_path}")
                print(f"------------------------")
                
                # --- W&B: Log that a checkpoint was saved ---
                wandb.log({
                    "step": step,
                    "checkpoint_saved_at_step": step
                })
                
                # (Optional) Save as a W&B Artifact
                # This will upload the .pt file to W&B
                # artifact = wandb.Artifact(f'model-{wandb.run.name}', type='model')
                # artifact.add_file(checkpoint_path, name=f"step_{step}.pt")
                # wandb.log_artifact(artifact, aliases=[f"step_{step}"])
                # print("Logged checkpoint artifact to W&B.")
            # --- End of Checkpointing Logic ---

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")
        
    print("Training finished.")
    
    # --- Save the final model ---
    final_model_path = os.path.join(checkpoint_dir, "final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # --- Test Generation ---
    print("\n--- Testing Generation ---")
    # Load the final model weights for generation
    model.load_state_dict(torch.load(final_model_path)) 
    model.eval() # Set model to evaluation mode for generation
    
    prompt = "Once upon a time there was"
    print(f"Prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(CONFIG['device'])
    
    # Ensure the model's generate function exists
    if hasattr(model, 'generate') and callable(model.generate):
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
    else:
        print("Error: model.generate() method not found.")
        print("Please ensure your SimpleAttentionLM class has a 'generate' method.")

    # --- W&B: Finish the run ---
    wandb.finish()
    print("W&B run finished.")
