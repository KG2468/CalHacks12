import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import time

# --- Import your model ---
# Make sure your corrected attention.py is in the same directory
from attention import SimpleAttentionLM 

# --- 1. Config ---
# Model Config
VOCAB_SIZE = 50257  # GPT-2's vocab size
BLOCK_SIZE = 256    # Context window
N_LAYER = 6
N_HEAD = 8
N_EMBD = 512
DROPOUT = 0.1

# Training Config
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
EVAL_INTERVAL = 100
LOG_INTERVAL = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Data Pipeline ---

class TextDataset(Dataset):
    """A simple dataset to hold tokenized text."""
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        # -1 because we need a target for each input
        return len(self.token_ids) - self.block_size - 1

    def __getitem__(self, idx):
        # Grab a chunk of text
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        # Input is all but the last token
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # Target is all but the first token (shifted by one)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_data_loaders(tokenizer, block_size, batch_size):
    print("Loading and tokenizing data...")
    # Load wikitext-103
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Concatenate all text and tokenize
    train_text = "\n".join(dataset['train']['text'])
    val_text = "\n".join(dataset['validation']['text'])
    
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    
    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")
    
    train_data = TextDataset(train_ids, block_size)
    val_data = TextDataset(val_ids, block_size)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=True, num_workers=4)
    
    return train_loader, val_loader

# --- 3. Training & Evaluation Functions ---

@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    """Calculates average validation loss."""
    model.eval()
    losses = []
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # We don't need the cache, just the logits
        logits, _ = model(x, past_kv_caches=None)
        
        # Calculate loss
        # logits: (B, T, V) -> (B*T, V)
        # y: (B, T) -> (B*T)
        loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
        losses.append(loss.item())
        
    model.train()
    return torch.tensor(losses).mean().item()

# --- 4. Main Training Script ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Init data
    train_loader, val_loader = get_data_loaders(tokenizer, BLOCK_SIZE, BATCH_SIZE)

    # Init model
    model = SimpleAttentionLM(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        ff_hidden=N_EMBD * 4,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # Init optimizer (AdamW is good for transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Loss function (ignore padding tokens if you have them, here we don't)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training Loop ---
    model.train()
    step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        epoch_start_time = time.time()
        
        for i, (x, y) in enumerate(train_loader):
            batch_start_time = time.time()
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # --- Forward pass ---
            # We run in training mode (past_kv_caches=None)
            # We ignore the returned cache
            logits, _ = model(x, past_kv_caches=None)
            
            # --- Calculate loss ---
            loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
            
            # --- Backward pass & optimization ---
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # --- Logging ---
            if step % LOG_INTERVAL == 0:
                batch_time = (time.time() - batch_start_time) * 1000 # in ms
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}ms")
            
            # --- Evaluation ---
            if step % EVAL_INTERVAL == 0:
                val_loss = evaluate(model, val_loader, loss_fn)
                print(f"--- Validation ---")
                print(f"Step {step} | Val Loss: {val_loss:.4f}")
                print(f"--------------------")
                model.train() # Set back to train mode

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")
        
    print("Training finished.")
    
    # --- Save the model ---
    torch.save(model.state_dict(), "simple_lm_pretrained.pt")
    print("Model saved to simple_lm_pretrained.pt")
    
    # --- Test Generation ---
    print("\n--- Testing Generation ---")
    model.load_state_dict(torch.load("simple_lm_pretrained.pt"))
    
    prompt = "Hello, my name is"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(
        prompt_ids, 
        max_new_tokens=50, 
        temperature=0.8, 
        top_k=50
    )
    
    generated_text = tokenizer.decode(generated_ids[0])
    print(generated_text)