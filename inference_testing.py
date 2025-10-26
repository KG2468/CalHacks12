import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import re

# Assuming 'attention.py' with SimpleAttentionLM is in the same directory
from attention import SimpleAttentionLM 

# --- Constants ---
LARGE_TOKENIZER_NAME = "Qwen/Qwen3-8B"
LARGE_MODEL_NAME = "Qwen/Qwen3-8B"
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper Functions ---

def count_cot_delimiters(text: str) -> int:
    """
    Counts the number of CoT delimiters ("\n\n").
    """
    # This directly counts non-overlapping occurrences of "\n\n"
    return text.count("\n\n")

def generate_with_cot_limit(model, tokenizer, input_ids, max_cot_groups: int, device: str) -> Tuple[str, torch.Tensor]:
    """
    Generate from model, stopping after max_cot_groups CoT delimiters ("\n\n") 
    or <|end_of_thought|>.
    
    Returns:
        Generated text (including prompt) and token IDs
    """
    model.eval()
    generated_ids = input_ids.clone().to(device)
    max_new_tokens = 512  # Safety limit
    
    # Get the special token ID for <|end_of_thought|> if it exists
    try:
        end_of_thought_token_id = tokenizer.convert_tokens_to_ids('<|end_of_thought|>')
    except (KeyError, AttributeError, TypeError):
        end_of_thought_token_id = None
        # print("Warning: '<|end_of_thought|>' token not found in tokenizer.") # Optional

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get next token prediction
            outputs = model(generated_ids.to(device))
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # --- Check for stopping conditions ---
            
            # 1. Check for <|end_of_thought|> token ID (fast)
            if end_of_thought_token_id is not None and next_token[0].item() == end_of_thought_token_id:
                # print("Stop condition: <|end_of_thought|> token found.")
                break
            
            # 2. Check for CoT delimiters (slower, requires decoding)
            # We decode only the newly generated part for efficiency, but it's complex.
            # Decoding the whole thing is safer to implement.
            decoded = tokenizer.decode(generated_ids[0])
            
            # Fallback check for the text tag
            if end_of_thought_token_id is None and '<|end_of_thought|>' in decoded:
                # print("Stop condition: <|end_of_thought|> text found.")
                break
            
            # Check for \n\n
            current_cot_count = count_cot_delimiters(decoded)
            if current_cot_count >= max_cot_groups:
                # print(f"Stop condition: {current_cot_count} CoT groups found.")
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return generated_text, generated_ids

def generate_response(model, tokenizer, input_ids, max_new_tokens: int, device: str) -> Tuple[str, torch.Tensor]:
    """
    Generate from model, stopping at EOS token or max_new_tokens.
    
    Returns:
        Generated text (including prompt) and token IDs
    """
    model.eval()
    generated_ids = input_ids.clone().to(device)
    
    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get next token prediction
            outputs = model(generated_ids.to(device))
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for stopping condition (EOS)
            if eos_token_id is not None and next_token[0].item() == eos_token_id:
                # print("Stop condition: EOS token found.")
                break
    
    # Use skip_special_tokens=True for a clean final response
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True) 
    return generated_text, generated_ids

# --- Main Inference Function ---

def test_inference():
    print(f"Using device: {DEVICE}")
    
    # Define the checkpoint path
    SMALL_MODEL_CHECKPOINT_PATH = "checkpoints/model_step_13600.pt"

    # --- Load Large Model ---
    print(f"Loading large model: {LARGE_MODEL_NAME}")
    large_model = AutoModelForCausalLM.from_pretrained(
        LARGE_MODEL_NAME, 
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map=DEVICE
    )
    large_tokenizer = AutoTokenizer.from_pretrained(LARGE_TOKENIZER_NAME)
    
    # Set pad token if not present (common for generation)
    if large_tokenizer.pad_token_id is None:
        large_tokenizer.pad_token_id = large_tokenizer.eos_token_id
        print(f"Set large_tokenizer.pad_token_id to: {large_tokenizer.eos_token_id}")

    # --- Load Small Model ---
    print(f"Loading small model: {TOKENIZER_NAME} (structure)")
    CONFIG = {
        "vocab_size": 50257,
        "block_size": 256,
        "n_layer": 6,
        "n_head": 8,
        "n_embd": 512,
        "ff_hidden": 512 * 4,
        "dropout": 0.1,
    }

    model_config = { 
        "vocab_size": CONFIG['vocab_size'],
        "block_size": CONFIG['block_size'],
        "n_layer": CONFIG['n_layer'],
        "n_head": CONFIG['n_head'],
        "n_embd": CONFIG['n_embd'],
        "ff_hidden": CONFIG['ff_hidden'],
        "dropout": CONFIG['dropout']
    }
    small_model = SimpleAttentionLM(**model_config).to(DEVICE)
    
    # !!! --- LOAD WEIGHTS FOR SMALL MODEL --- !!!
    try:
        print(f"Loading small model weights from: {SMALL_MODEL_CHECKPOINT_PATH}")
        small_model.load_state_dict(torch.load(SMALL_MODEL_CHECKPOINT_PATH))
        print("Small model weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {SMALL_MODEL_CHECKPOINT_PATH}.")
        print("Proceeding with randomly initialized weights for small model.")
    except Exception as e:
        print(f"ERROR: Failed to load small model weights: {e}")
        print("Proceeding with randomly initialized weights for small model.")
    # --- END OF WEIGHT LOADING ---

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set tokenizer.pad_token_id to: {tokenizer.eos_token_id}")

    # --- Set Loop Parameters ---
    NUM_LOOPS = 4       # "repeat this loop around 4-5 times"
    MAX_COT_GROUPS = 3  # "3-5 \n\n" (using 3 as the limit)
    
    # --- Initial Prompt ---
    initial_prompt_text = "QUESTION: Explain the process of photosynthesis in simple terms."
    print(f"--- Initial Prompt ---\n{initial_prompt_text}\n")
    
    current_prompt_text = initial_prompt_text
    
    # --- The Inference Loop ---
    for i in range(NUM_LOOPS):
        print(f"================== LOOP {i+1} / {NUM_LOOPS} ==================\n")
        
        # --- Step 1: Large Model CoT Generation ---
        print("Step 1: Generating CoT from large model...")
        
        # Tokenize for large model
        large_model_input_ids = large_tokenizer(
            current_prompt_text, 
            return_tensors='pt'
        ).input_ids.to(DEVICE)
        
        cot_full_text, _ = generate_with_cot_limit(
            large_model, 
            large_tokenizer,  # <-- Use correct (large) tokenizer
            large_model_input_ids, 
            MAX_COT_GROUPS, 
            DEVICE
        )
        
        # Isolate just the newly generated CoT
        cot_only_text = cot_full_text[len(current_prompt_text):]
        print(f"--- [Large Model CoT] ---\n{cot_only_text}\n")

        # --- Step 2: Combine Prompt + CoT ---
        combined_text = current_prompt_text + cot_only_text
        
        # --- Step 3: Small Model Response Generation ---
        print("Step 2: Generating response from small model...")
        
        # Tokenize for small model
        small_model_input_ids = tokenizer(
            combined_text, 
            return_tensors='pt'
        ).input_ids.to(DEVICE)
        
        small_model_full_response, _ = generate_response(
            small_model,
            tokenizer,        # <-- Use correct (small) tokenizer
            small_model_input_ids,
            max_new_tokens=100, # Limit small model output
            device=DEVICE
        )
        
        # Isolate just the new part for printing
        small_response_only_text = small_model_full_response[len(combined_text):]
        print(f"--- [Small Model Response] ---\n{small_response_only_text}\n")

        # --- Step 4: Update Prompt for Next Loop ---
        current_prompt_text = small_model_full_response

    # --- Step 6: Final Generation ---
    print("================== FINAL GENERATION ==================\n")
    print("Generating final completion from large model...")
    
    # Tokenize the final prompt
    final_input_ids = large_tokenizer(
        current_prompt_text, 
        return_tensors='pt'
    ).input_ids.to(DEVICE)
    
    # Use the 'generate_response' function for a standard completion
    final_text, _ = generate_response(
        large_model,
        large_tokenizer,
        final_input_ids,
        max_new_tokens=512, # Let it generate a full answer
        device=DEVICE
    )
    
    print(f"--- [Final Answer] ---\n{final_text}\n")


if __name__ == "__main__":
    test_inference()