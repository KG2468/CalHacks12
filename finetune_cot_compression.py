
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import argparse
import re
from typing import List, Tuple

# --- Import your model from the other file ---
try:
    from attention import SimpleAttentionLM
except ImportError:
    print("Error: Could not import SimpleAttentionLM from attention.py")
    print("Please make sure attention.py is in the same directory.")
    exit()

# --- 1. Config ---
# Small model (the one we're training)
SMALL_VOCAB_SIZE = 50257
SMALL_BLOCK_SIZE = 2048
SMALL_N_LAYER = 6
SMALL_N_HEAD = 8
SMALL_N_EMBD = 512
SMALL_DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 1  # Process one example at a time due to complex logic
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
LOG_INTERVAL = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Large model (frozen, used for generation)
LARGE_MODEL_NAME = "Qwen/Qwen3-8B"  # You can change this to any larger model

# Loss weights
TOKEN_LENGTH_WEIGHT = 1  # Weight for token length penalty

# Tokenizer
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
LARGE_TOKENIZER_NAME = "Qwen/Qwen3-8B"
LARGE_MODEL_NAME = "Qwen/Qwen3-8B"

# --- 2. Helper Functions ---
def split_cot_into_groups(cot_steps: List[torch.Tensor], min_groups: int = 3, max_group_size: int = 5) -> List[List[torch.Tensor]]:
    """
    Split CoT steps into groups. Scale up group size first, then number of groups.
    
    Args:
        cot_steps: List of CoT step tensors
        min_groups: Minimum number of groups (default 3)
        max_group_size: Maximum statements per group (default 5)
    
    Returns:
        List of groups, where each group is a list of tensors
    """
    num_steps = len(cot_steps)
    
    # Start with min_groups and max_group_size
    group_size = min(max_group_size, num_steps // min_groups)
    if group_size == 0:
        group_size = 1
    
    # Calculate number of groups needed
    num_groups = (num_steps + group_size - 1) // group_size  # Ceiling division
    
    # If we need more groups than min_groups, that's fine
    # Otherwise, try to balance the groups
    if num_groups < min_groups:
        num_groups = min_groups
        group_size = num_steps // num_groups
    
    # Create groups
    groups = []
    for i in range(0, num_steps, group_size):
        group = cot_steps[i:i + group_size]
        if group:  # Only add non-empty groups
            groups.append(group)
    
    return groups

def count_cot_delimiters(text: str) -> int:
    """Count the number of CoT delimiters (\\n\\n) in text."""
    return text.count('\n\n')

def extract_final_output(text: str) -> str:
    """Extract text after <|end_of_thought|> tag."""
    if '<|end_of_thought|>' in text:
        return text.split('<|end_of_thought|>', 1)[1].strip()
    return text

def generate_with_cot_limit(model, tokenizer, input_ids, max_cot_groups: int, device: str) -> Tuple[str, torch.Tensor]:
    """
    Generate from model, stopping after max_cot_groups CoT delimiters or <|end_of_thought|>.
    
    Returns:
        Generated text and token IDs
    """
    model.eval()
    generated_ids = input_ids.clone()
    max_new_tokens = 512  # Safety limit
    cot_count = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get next token prediction
            outputs = model(generated_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Decode to check for stopping conditions
            decoded = tokenizer.decode(generated_ids[0])
            
            # Check for end of thought tag
            if '<|end_of_thought|>' in decoded:
                break
            
            # Count CoT delimiters
            current_cot_count = count_cot_delimiters(decoded)
            if current_cot_count >= max_cot_groups:
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return generated_text, generated_ids

# --- 3. Dataset ---
class StructuredCoTDataset(Dataset):
    """Dataset for structured chain-of-thought data."""
    
    def __init__(self, data_file, tokenizer, custom_sys_prompt=None):
        super().__init__()
        
        print(f"Loading structured dataset from {data_file}...")
        data = torch.load(data_file)

        self.system_prompt = custom_sys_prompt if custom_sys_prompt else data['system_prompt']
        self.eos_token = data['eos_token']
        self.tokenizer = tokenizer
        
        # Filter examples with at least 3 CoT steps
        # Structure: [prompt, cot1, cot2, ..., final_answer]
        self.examples = []
        for ex in data['examples']:
            
            # ex has: [prompt, cot_step1, cot_step2, ..., final_answer]
            # We need at least: prompt + 3 cot steps + final answer = 5 elements minimum
            if len(ex) >= 5:  # prompt + at least 3 CoT + final
                self.examples.append(ex)
        
        print(f"Loaded {len(self.examples):,} examples with at least 3 CoT steps")
        print(f"System prompt: {len(self.system_prompt)} tokens")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Structure: [prompt, cot1, cot2, ..., cotn, final_answer]
        prompt = example[0]
        cot_steps = example[1:-1]  # All middle elements are CoT steps
        final_answer = example[-1]
        
        return {
            'system_prompt': self.system_prompt,
            'prompt': prompt,
            'cot_steps': cot_steps,
            'final_answer': final_answer
        }

# --- 4. Custom Training Loop ---
def train_epoch(small_model, large_model, large_tokenizer, train_loader, optimizer, tokenizer, device, epoch):
    """Custom training loop for CoT compression."""
    small_model.train()
    large_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        for i in range(BATCH_SIZE):
            system_prompt = batch['system_prompt'][i]
            prompt = batch['prompt'][i]
            cot_steps = batch['cot_steps'][i]  # List of tensors
            final_answer = batch['final_answer'][i]
            
            # Split CoT into groups
            cot_groups = split_cot_into_groups(cot_steps, min_groups=3, max_group_size=5)
            
            if len(cot_groups) < 2:
                # Skip if we can't make at least 2 groups
                continue
            
            print(f"\n[Batch {batch_idx+1}] Processing example with {len(cot_steps)} CoT steps -> {len(cot_groups)} groups")
            
            batch_loss = 0.0
            
            # Start with just the user prompt (no system prompt)
            current_input = prompt.unsqueeze(0).to(device)
            
            # Process each group (except the last one)
            for group_idx in range(len(cot_groups) - 1):
                current_group = cot_groups[group_idx]
                next_group = cot_groups[group_idx + 1]
                
                # Add current group to input
                current_group_tokens = torch.cat(current_group, dim=0).unsqueeze(0).to(device)
                full_input = torch.cat([current_input, current_group_tokens], dim=1)
                
                # Forward pass through small model in TRAIN mode to get trainable logits
                small_model.train()
                logits, _ = small_model(full_input, past_kv_caches=None)
                
                # Generate compressed tokens autoregressively from small model
                # We'll generate a fixed number of tokens (e.g., 50) as the compressed representation
                # compressed_length = min(50, full_input.shape[1] // 2)  # Compress to half size or 50 tokens
                compressed_ids = full_input.clone()  # Start with the input
                eos_token_id = tokenizer.eos_token_id
                
                with torch.no_grad():
                    for _ in range(4096-full_input.shape[1]):
                        # Get logits for next token
                        temp_logits, _ = small_model(compressed_ids, past_kv_caches=None)
                        logits = temp_logits
                        next_token = torch.argmax(temp_logits[:, -1, :], dim=-1, keepdim=True)
                        compressed_ids = torch.cat([compressed_ids, next_token], dim=1)
                        
                        # Stop if we generated the EOS token
                        if next_token.item() == eos_token_id:
                            break
                
                # Take only the newly generated tokens (not the input)
                compressed_ids = compressed_ids[:, full_input.shape[1]:]
                compressed_length = compressed_ids.shape[1]

                # compressed_prompt = tokenizer.decode(compressed_ids[0], skip_special_tokens=False)

                # Pass system prompt + small model output to large model
                large_input_ids = torch.cat([system_prompt.unsqueeze(0).to(device), compressed_ids], dim=1)

                large_input_ids = tokenizer.decode(large_input_ids, skip_special_tokens=False)

                large_input_ids = large_tokenizer.encode(large_input_ids, return_tensors='pt').to(device)
                
                # Generate from large model
                generated_text, generated_ids = generate_with_cot_limit(
                    large_model, large_tokenizer, large_input_ids,
                    max_cot_groups=len(current_group), device=device
                )
                
                # Extract the generated CoT (everything after the input)
                input_len = large_input_ids.shape[1]
                generated_cot_ids = generated_ids[:, input_len:]
                generated_cot_text = large_tokenizer.decode(generated_cot_ids[0], skip_special_tokens=False)
                generated_cot_tokens = tokenizer.encode(generated_cot_text, return_tensors='pt').to(device)

                # Get target tokens (next group)
                target_tokens = torch.cat(next_group, dim=0).unsqueeze(0).to(device)

                
                # Compute a reward signal: how many tokens match
                pred_len = min(generated_cot_tokens.shape[1], target_tokens.shape[1]) #Possible shape error for generated_cot_tokens
                generated_for_comparison = generated_cot_tokens[:, :pred_len]
                target_for_comparison = target_tokens[:, :pred_len]
                
                # Token-level accuracy as reward (percentage of matching tokens)
                matches = (generated_for_comparison == target_for_comparison).float()
                reward = matches.mean()
                
                # Use REINFORCE-style loss: encourage the small model to generate tokens
                # that lead to good large model outputs
                # We compute negative log likelihood of the compressed tokens and weight by reward
                # This is a simplified policy gradient approach
                
                # Get the log probabilities of the tokens the small model generated
                compressed_logits_for_generated = logits[:, -compressed_length:, :]
                log_probs = torch.log_softmax(compressed_logits_for_generated, dim=-1)
                
                # Get the actual tokens that were generated
                compressed_tokens = compressed_ids[:, :compressed_length]
                
                # Gather log probs for the tokens that were actually selected
                selected_log_probs = torch.gather(
                    log_probs.reshape(-1, log_probs.shape[-1]),
                    1,
                    compressed_tokens.reshape(-1, 1)
                ).squeeze(-1)
                
                # REINFORCE loss: -log_prob * (reward - baseline)
                # Using reward directly (baseline = 0 for simplicity)
                # Negative because we want to maximize reward (minimize negative reward)
                ce_loss = -(selected_log_probs * reward.detach()).mean()
                
                # Length penalty (encourage shorter outputs)
                ideal_length = 7 * full_input.shape[1] // 8  # Hard coded 7/8 compression factor
                length_penalty = (max(compressed_length - ideal_length, 0) / ideal_length) * TOKEN_LENGTH_WEIGHT #Hard coded 7/8 compression factor

                group_loss = ce_loss + length_penalty
                batch_loss += group_loss

                print(f"  Group {group_idx+1}/{len(cot_groups)-1}: CE Loss={ce_loss.item():.4f}, Length={compressed_length}, Total={group_loss.item():.4f}")

                # Update current_input for next iteration
                current_input = compressed_ids
            
            # Final step: generate final answer
            last_cot_group = cot_groups[-1]
            last_cot_tokens = torch.cat(last_cot_group, dim=0).unsqueeze(0).to(device)
            final_input = torch.cat([current_input, last_cot_tokens], dim=1)
            
            # Forward pass in TRAIN mode to get trainable logits
            small_model.train()
            final_logits, _ = small_model(final_input, past_kv_caches=None)
            
            # Generate compressed prompt for final answer autoregressively
            compressed_final_ids = final_input.clone()
            eos_token_id = tokenizer.eos_token_id
            
            with torch.no_grad():
                for _ in range(4096-final_input.shape[1]):
                    temp_logits, _ = small_model(compressed_final_ids, past_kv_caches=None)
                    final_logits = temp_logits
                    next_token = torch.argmax(temp_logits[:, -1, :], dim=-1, keepdim=True)
                    compressed_final_ids = torch.cat([compressed_final_ids, next_token], dim=1)
                    
                    # Stop if we generated the EOS token
                    if next_token.item() == eos_token_id:
                        break
            
            # Take only the newly generated tokens
            compressed_final_ids = compressed_final_ids[:, final_input.shape[1]:]
            compressed_final_length = compressed_final_ids.shape[1]
            
            # Generate final answer from large model: system prompt + small model output
            large_final_input = torch.cat([system_prompt.unsqueeze(0).to(device), compressed_final_ids], dim=1)
            large_final_input = tokenizer.decode(large_final_input, skip_special_tokens=False)
            large_final_input = large_tokenizer.encode(large_final_input, return_tensors='pt').to(device)
            final_generated_text, final_generated_ids = generate_with_cot_limit(
                large_model, large_tokenizer, large_final_input,
                max_cot_groups=1e9,  # No limit, generate to completion
                device=device
            )
            
            # Extract final output (after thinking tags)
            final_output = extract_final_output(final_generated_text)
            final_output_tokens = tokenizer.encode(final_output, return_tensors='pt').to(device)
            # target_final_text = tokenizer.decode(final_answer, skip_special_tokens=False)
            

            # Compute a reward signal: how many tokens match
            pred_len = min(final_output_tokens.shape[1], final_answer.shape[1]) #Possible shape error for generated_cot_tokens
            generated_for_comparison = final_output_tokens[:, :pred_len]
            target_for_comparison = final_answer[:, :pred_len]

            # Token-level accuracy as reward (percentage of matching tokens)
            matches = (generated_for_comparison == target_for_comparison).float()
            reward = matches.mean()
            
            # Use REINFORCE-style loss: encourage the small model to generate tokens
            # that lead to good large model outputs
            # We compute negative log likelihood of the compressed tokens and weight by reward
            # This is a simplified policy gradient approach
            
            # Get the log probabilities of the tokens the small model generated
            compressed_logits_for_generated = final_logits[:, -compressed_final_length:, :]
            log_probs = torch.log_softmax(compressed_logits_for_generated, dim=-1)
            
            # Get the actual tokens that were generated
            compressed_tokens = compressed_final_ids[:, :compressed_final_length]
            
            # Gather log probs for the tokens that were actually selected
            selected_log_probs = torch.gather(
                log_probs.reshape(-1, log_probs.shape[-1]),
                1,
                compressed_tokens.reshape(-1, 1)
            ).squeeze(-1)
            
            # REINFORCE loss: -log_prob * (reward - baseline)
            # Using reward directly (baseline = 0 for simplicity)
            # Negative because we want to maximize reward (minimize negative reward)
            final_ce_loss = -(selected_log_probs * reward.detach()).mean()
            
            # Length penalty (encourage shorter outputs)
            final_length_penalty = max(compressed_final_length - 7*final_input.shape[1] // 8, 0) * TOKEN_LENGTH_WEIGHT #Hard coded 7/8 compression factor


            # Compute final loss using the logits we already computed
            # No redundant forward pass needed!
            # target_final_ids = final_answer.unsqueeze(0).to(device)
            # pred_len = min(compressed_final_length, target_final_ids.shape[1])
            # final_logits_for_loss = final_logits[:, -pred_len:, :]
            # final_target = target_final_ids[:, :pred_len]
            
            # final_ce_loss = nn.functional.cross_entropy(
            #     final_logits_for_loss.reshape(-1, final_logits_for_loss.shape[-1]),
            #     final_target.reshape(-1)
            # )
            
            # # Length penalty for final generation
            # input_len = large_final_input.shape[1]
            # final_gen_len = final_generated_ids.shape[1] - input_len
            # final_length_penalty = final_gen_len * TOKEN_LENGTH_WEIGHT
            
            final_loss = final_ce_loss + final_length_penalty
            batch_loss += final_loss

            print(f"  Final: CE Loss={final_ce_loss.item():.4f}, Length={compressed_final_length}, Total={final_loss.item():.4f}")
            print(f"  Batch Total Loss: {batch_loss.item():.4f}")
            
            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(small_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = batch_loss / BATCH_SIZE
                print(f"\n[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} | Avg Loss: {avg_loss:.4f}")
        
    return total_loss / max(num_batches, 1)

# --- 5. Main ---
def main(args):
    print(f"Using device: {DEVICE}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    large_tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_NAME)
    if large_tokenizer.pad_token is None:
        large_tokenizer.pad_token = large_tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = StructuredCoTDataset(args.data_file, tokenizer, custom_sys_prompt=CUSTOM_SYS_PROMPT)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize small model (the one we're training)
    print("Initializing small model...")
    small_model_config = {
        "vocab_size": SMALL_VOCAB_SIZE,
        "block_size": SMALL_BLOCK_SIZE,
        "n_layer": SMALL_N_LAYER,
        "n_head": SMALL_N_HEAD,
        "n_embd": SMALL_N_EMBD,
        "ff_hidden": SMALL_N_EMBD * 4,
        "dropout": SMALL_DROPOUT
    }
    small_model = SimpleAttentionLM(**small_model_config).to(DEVICE)
    
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pretrained weights from {args.pretrained_model}...")
        state_dict = torch.load(args.pretrained_model, map_location=DEVICE)
        small_model.load_state_dict(state_dict)
    
    print(f"Small model parameters: {sum(p.numel() for p in small_model.parameters())/1e6:.2f}M")
    
    # Load large model (frozen)
    print(f"Loading large model: {args.large_model}...")
    large_model = AutoModelForCausalLM.from_pretrained(args.large_model).to(DEVICE)
    large_model.eval()
    for param in large_model.parameters():
        param.requires_grad = False
    print(f"Large model parameters: {sum(p.numel() for p in large_model.parameters())/1e6:.2f}M")
    
    # Optimizer
    optimizer = AdamW(small_model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        epoch_loss = train_epoch(small_model, large_model, large_tokenizer, train_loader, optimizer, tokenizer, DEVICE, epoch)
        print(f"\nEpoch {epoch+1} finished | Average Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = args.output_model.replace('.pt', f'_epoch{epoch+1}.pt')
        torch.save(small_model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    torch.save(small_model.state_dict(), args.output_model)
    print(f"\n✅ Training complete! Final model saved to {args.output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune small model for CoT compression')
    
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to structured dataset (.pt file)')
    parser.add_argument('--output_model', type=str, default='cot_compressor.pt',
                        help='Path to save trained model')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pretrained small model (optional)')
    parser.add_argument('--large_model', type=str, default=LARGE_MODEL_NAME,
                        help=f'Large model for generation (default: {LARGE_MODEL_NAME})')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--custom_sys_prompt', type=str, default=None,
                        help='Custom system prompt (optional)')
    
    args = parser.parse_args()
    
    if args.num_epochs != NUM_EPOCHS:
        NUM_EPOCHS = args.num_epochs

    if args.custom_sys_prompt:
        CUSTOM_SYS_PROMPT = args.custom_sys_prompt
    if args.learning_rate != LEARNING_RATE:
        LEARNING_RATE = args.learning_rate
    
    main(args)
