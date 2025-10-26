import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import re
import os

# --- Config ---
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
OUTPUT_FILE = "finetune_dataset.pt"  # Changed to .pt for PyTorch format

# We use uint16 since vocab_size (50257) fits
# This saves 50% of memory vs. int32
DTYPE = np.uint16 

def format_chain_of_thought(example):
    """
    Format a chain-of-thought example into a training sequence.
    The dataset has 'instruction', 'input', and 'output' fields.
    We'll format it as: Instruction: {instruction}\nInput: {input}\nResponse: {output}
    """
    system_prompt = example.get("system")
    example = example["conversations"]
    prompt = example[0].get('value')

    out = example[1].get('value')

    # Split response at the first occurrence of any trigger tag
    triggers = ['<|begin_of_thought|>', '<|end_of_thought|>']
    pattern = '|'.join(re.escape(t) for t in triggers)
    out = re.split(pattern, out)
    cot = out[0].strip()
    final = out[1].strip()
    cot = re.split(r'\n\n', cot)
   

    # Build the formatted text
    # if input_text:
    #     formatted = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
    # else:
    #     formatted = f"Instruction: {instruction}\nResponse: {output}"
    # print(formatted)
    return [system_prompt] + [prompt] + cot + [final]

def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # List to hold all our token IDs
    all_token_ids = []
    
    # --- Load Chain-of-Thought Dataset ---
    print("Loading 'open-thoughts/OpenThoughts-114k' dataset...")
    try:
        cot_dataset = load_dataset("open-thoughts/OpenThoughts-114k")
        
        # Get the train split (or combine all splits if available)
        if 'train' in cot_dataset:
            dataset = cot_dataset['train']
            print(f"Loaded train split with {len(dataset)} examples")
        else:
            # If no train split, use the first available split
            split_name = list(cot_dataset.keys())[0]
            dataset = cot_dataset[split_name]
            print(f"Loaded '{split_name}' split with {len(dataset)} examples")
        
        # If there are multiple splits, you might want to combine them
        if len(cot_dataset.keys()) > 1:
            print(f"Available splits: {list(cot_dataset.keys())}")
            from datasets import concatenate_datasets
            all_splits = [cot_dataset[split] for split in cot_dataset.keys()]
            dataset = concatenate_datasets(all_splits)
            print(f"Combined all splits: {len(dataset)} total examples")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check if the dataset name is correct and you have internet connection.")
        exit()
    
    # --- Process and Tokenize ---
    print("Processing and tokenizing chain-of-thought examples...")
    start_time = time.time()
    tags = {}
    num_tags = 0
    for i, example in enumerate(dataset):
        # Format the example
        formatted_text = format_chain_of_thought(example)
        
        

        
        # Tokenize - we add special tokens here to mark document boundaries
        # Add EOS token at the end of each example so the model knows when a response ends
        tokens = [tokenizer.encode(step, add_special_tokens=False) for step in formatted_text]
        tokens.append(tokenizer.eos_token_id)  # Add EOS at the end
        
        all_token_ids.append(tokens)
        
        if (i+1) % 1000 == 0:
            elapsed = time.time() - start_time
            examples_per_sec = (i+1) / elapsed
            print(f"  Processed {i+1}/{len(dataset)} examples | {len(all_token_ids):,} tokens | {examples_per_sec:.1f} examples/sec")
    
    total_tokens = sum([len(t) for t in all_token_ids])
    elapsed = time.time() - start_time
    print(f"\nFinished processing dataset.")
    print(f"Total examples: {len(dataset):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Average tokens per example: {total_tokens/len(dataset):.1f}")
    print(f"Unique tags found: {len(tags)}")
    print(f"Average tags per example: {num_tags/len(dataset):.1f}")
    print(f"Tags: {tags}")
    print(f"Tags: {dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))}")  #Keys in tags dictionary sorted by values descending

    # --- Save to PyTorch File (preserves structure) ---
    print(f"\nSaving structured data to {OUTPUT_FILE}...")
    
    # Convert to PyTorch tensors while preserving the nested structure
    # Each example is a list of token lists (one per step) plus an EOS token
    structured_data = []
    for example_tokens in all_token_ids:
        example_tensors = []
        for step_tokens in example_tokens[:-1]:  # All steps except EOS
            if isinstance(step_tokens, list):
                example_tensors.append(torch.tensor(step_tokens, dtype=torch.int64))
        # Add the EOS token as a single-element tensor
        if isinstance(example_tokens[-1], int):
            example_tensors.append(torch.tensor([example_tokens[-1]], dtype=torch.int64))
        structured_data.append(example_tensors)
    
    # Save using torch.save to preserve the structure
    torch.save(structured_data, OUTPUT_FILE)
    
    file_size_gb = os.path.getsize(OUTPUT_FILE) / 1e9
    print(f"Successfully saved {len(structured_data):,} structured examples to {OUTPUT_FILE}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Structure: Each example contains {len(structured_data[0])} steps (system, prompt, CoT steps..., final, EOS)")
    
    # --- Print Sample ---
    print("\n" + "="*60)
    print("Sample formatted example:")
    print("="*60)
    sample_formatted = format_chain_of_thought(dataset[0])
    print(sample_formatted[:500] + "..." if len(sample_formatted) > 500 else sample_formatted)
    print("="*60)
    
    print("\nFinetune dataset build complete.")
    print(f"You can now use this with: python finetune.py --pretrained_model <model.pt> --data_file {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
