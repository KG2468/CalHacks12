import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
import time
import numpy as np
import os
import argparse

# --- Import your model from the other file ---
try:
    from attention import SimpleAttentionLM
except ImportError:
    print("Error: Could not import SimpleAttentionLM from attention.py")
    print("Please make sure attention.py is in the same directory.")
    exit()

# --- 1. Config ---
# These should match the pre-trained model
VOCAB_SIZE = 50257  # GPT-Neo tokenizer vocab size
BLOCK_SIZE = 256    # Context window size
N_LAYER = 6         # Number of transformer blocks
N_HEAD = 8          # Number of attention heads
N_EMBD = 512        # Embedding dimension
DROPOUT = 0.1

# Finetuning hyperparameters
BATCH_SIZE = 16     # Smaller batch size for finetuning
LEARNING_RATE = 1e-5 # Much lower learning rate for finetuning
NUM_EPOCHS = 3      # More epochs for finetuning
EVAL_INTERVAL = 50  # How often to run evaluation
LOG_INTERVAL = 10   # How often to print training loss
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tokenizer
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"

# --- 2. Data Pipeline ---
class StructuredCoTDataset(Dataset):
    """
    Dataset for structured chain-of-thought data.
    Each example is a list of token tensors without the system prompt or EOS token.
    The system prompt and EOS are stored once in the dataset file.
    """
    def __init__(self, data_file, block_size):
        super().__init__()
        self.block_size = block_size
        
        print(f"Loading structured dataset from {data_file}...")
        data = torch.load(data_file)
        
        self.system_prompt = data['system_prompt']
        self.eos_token = data['eos_token']
        self.examples = data['examples']
        
        print(f"Loaded {len(self.examples):,} examples")
        print(f"System prompt: {len(self.system_prompt)} tokens")
        print(f"EOS token: {self.eos_token.tolist()}")
        
        # Filter examples that would be too long when system prompt and EOS are added
        original_len = len(self.examples)
        system_eos_len = len(self.system_prompt) + len(self.eos_token)
        self.examples = [ex for ex in self.examples if self._total_length(ex) + system_eos_len <= block_size + 1]
        
        if len(self.examples) < original_len:
            print(f"Filtered out {original_len - len(self.examples)} examples that exceed block_size")
    
    def _total_length(self, example):
        """Calculate total length of all steps in an example."""
        return sum(len(step) for step in example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Reconstruct full sequence: system_prompt + example steps + eos_token
        all_tokens = torch.cat([self.system_prompt] + example + [self.eos_token], dim=0)
        
        # Create input and target (shifted by 1)
        if len(all_tokens) > self.block_size + 1:
            all_tokens = all_tokens[:self.block_size + 1]
        
        x = all_tokens[:-1]
        y = all_tokens[1:]
        
        # Pad if necessary
        if len(x) < self.block_size:
            pad_len = self.block_size - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.int64)])
            y = torch.cat([y, torch.zeros(pad_len, dtype=torch.int64)])
        
        return x, y

# --- 3. Training & Evaluation Functions ---
@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    """Calculates average validation loss."""
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    for k in range(100):  # Evaluate on 100 batches
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
            
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x, past_kv_caches=None)
        loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
        losses.append(loss.item())
        
    model.train()
    if not losses:
        return 0.0
    return torch.tensor(losses).mean().item()

# --- 4. Main Finetuning Script ---
def main(args):
    print(f"Using device: {DEVICE}")
    print(f"Finetuning mode: Loading pretrained model from {args.pretrained_model}")
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: {args.data_file} not found.")
        print("Please provide a valid dataset file.")
        exit()
    
    # Check if pretrained model exists
    if not os.path.exists(args.pretrained_model):
        print(f"Error: Pretrained model {args.pretrained_model} not found.")
        print("Please train a model first using train.py")
        exit()

    # --- Init Tokenizer ---
    print(f"Loading '{TOKENIZER_NAME}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Init Data ---
    print("Loading finetuning data...")
    full_dataset = StructuredCoTDataset(args.data_file, BLOCK_SIZE)
    
    # Create a 90/10 train/val split
    n = len(full_dataset)
    train_n = int(n * 0.9)
    val_n = n - train_n
    train_data, val_data = torch.utils.data.random_split(
        full_dataset, 
        [train_n, val_n],
        generator=torch.Generator().manual_seed(42)
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
    print("Initializing model architecture...")
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
    
    # --- Load Pretrained Weights ---
    print(f"Loading pretrained weights from {args.pretrained_model}...")
    state_dict = torch.load(args.pretrained_model, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Pretrained weights loaded successfully!")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # --- Init Optimizer & Loss ---
    # Use a lower learning rate for finetuning
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # Optional: Learning rate scheduler
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        print(f"Using CosineAnnealingLR scheduler with {total_steps} total steps")

    # --- Finetuning Loop ---
    print("Starting finetuning...")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("-" * 50)
    
    model.train()
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        epoch_start_time = time.time()
        
        for i, (x, y) in enumerate(train_loader):
            batch_start_time = time.time()
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits, _ = model(x, past_kv_caches=None)
            loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Optional: Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            if args.use_scheduler:
                scheduler.step()
            
            step += 1
            
            if step % LOG_INTERVAL == 0:
                batch_time = (time.time() - batch_start_time) * 1000  # in ms
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step} | Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Time: {batch_time:.2f}ms")
            
            if step % EVAL_INTERVAL == 0 and step > 0:
                val_loss = evaluate(model, val_loader, loss_fn)
                print(f"\n--- Validation ---")
                print(f"Step {step} | Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = args.output_model.replace('.pt', '_best.pt')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"✓ New best model saved to {best_model_path}")
                
                print(f"--------------------\n")
                model.train()

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")
        
        # Save checkpoint after each epoch
        if args.save_checkpoints:
            checkpoint_path = args.output_model.replace('.pt', f'_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
    print("\nFinetuning finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # --- Save the final finetuned model ---
    torch.save(model.state_dict(), args.output_model)
    print(f"Final model saved to {args.output_model}")
    
    # --- Test Generation ---
    print("\n" + "="*50)
    print("--- Testing Generation with Finetuned Model ---")
    print("="*50)
    
    model.eval()
    
    # Test with multiple prompts
    test_prompts = [
        args.test_prompt,
        "The future of AI is",
        "In a world where"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        generated_ids = model.generate(
            prompt_ids, 
            max_new_tokens=50, 
            temperature=0.8, 
            top_k=50
        )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune a SimpleAttentionLM model')
    
    # Required arguments
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained model checkpoint (.pt file)')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to finetuning dataset (.pt file with structured format)')
    
    # Optional arguments
    parser.add_argument('--output_model', type=str, default='finetuned_model.pt',
                        help='Path to save finetuned model (default: finetuned_model.pt)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use cosine annealing learning rate scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable, default: 1.0)')
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Save model checkpoint after each epoch')
    parser.add_argument('--test_prompt', type=str, default='Once upon a time',
                        help='Test prompt for generation (default: "Once upon a time")')
    
    args = parser.parse_args()
    
    # Override global config with command line args if provided
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
    if args.learning_rate != LEARNING_RATE:
        LEARNING_RATE = args.learning_rate
    if args.num_epochs != NUM_EPOCHS:
        NUM_EPOCHS = args.num_epochs
    
    main(args)
