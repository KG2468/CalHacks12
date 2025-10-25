import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import time

# --- Config ---
TARGET_SP_TOKENS = 1_000_000_000  # 1 Billion tokens from SlimPajama
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
OUTPUT_FILE = "train_dataset.bin"

# We use uint16 since vocab_size (50257) fits
# This saves 50% of memory vs. int32
# (1.48B tokens * 2 bytes/token = ~2.96 GB file)
DTYPE = np.uint16 

def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # List to hold all our token IDs
    all_token_ids = []
    
    # --- Part 1: Process TinyStories ---
    print("Loading and tokenizing 'roneneldan/TinyStories'...")
    ts_dataset = load_dataset("roneneldan/TinyStories")
    
    # Combine train and validation splits
    from datasets import concatenate_datasets
    ts_combined = concatenate_datasets([ts_dataset['train'], ts_dataset['validation']])

    print("Tokenizing TinyStories... (this is memory-efficient now)")

    def tokenize_function(examples):
        # Tokenize the text, but don't add special tokens
        return tokenizer(examples['text'], add_special_tokens=False)

    # Use .map() to tokenize the dataset.
    # This processes the data in batches and is highly memory-efficient.
    tokenized_ts = ts_combined.map(
        tokenize_function,
        batched=True,
        num_proc=4, # Use multiple processes to speed it up (adjust as needed)
        remove_columns=['text'] # We don't need the original text column anymore
    )

    print("Extracting tokens...")
    for item in tokenized_ts:
        all_token_ids.extend(item['input_ids'])
    
    ts_token_count = len(all_token_ids)
    print(f"Added {ts_token_count:,} tokens from TinyStories.")
    
    # --- Part 2: Process SlimPajama ---
    print(f"Loading and streaming 'cerebras/SlimPajama-627B'...")
    sp_dataset = load_dataset(
        "cerebras/SlimPajama-627B",
        streaming=True,
        split='train'
    )
    
    # Shuffle the stream. This buffer gives us a good mix of the sources
    # without needing to see the whole dataset.
    sp_shuffled = sp_dataset.shuffle(buffer_size=10_000, seed=42)
    
    print(f"Streaming and tokenizing {TARGET_SP_TOKENS:,} tokens from SlimPajama...")
    sp_token_count = 0
    start_time = time.time()
    
    for i, doc in enumerate(sp_shuffled):
        if sp_token_count >= TARGET_SP_TOKENS:
            print("Target SlimPajama token count reached.")
            break
            
        # Tokenize the document and append
        # We set add_special_tokens=False because we'll just concatenate
        # all text into one giant sequence.
        sp_tokens = tokenizer.encode(doc['text'], add_special_tokens=False)
        all_token_ids.extend(sp_tokens)
        sp_token_count += len(sp_tokens)
        
        if (i+1) % 1000 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = sp_token_count / elapsed
            print(f"  Processed {i+1} docs | {sp_token_count:,} SP tokens | {tokens_per_sec:.0f} tokens/sec")

    print(f"Finished processing SlimPajama. Added {sp_token_count:,} tokens.")
    
    # --- Part 3: Combine and Save ---
    total_tokens = len(all_token_ids)
    print(f"\nTotal tokens from all sources: {total_tokens:,}")
    
    print(f"Converting to {DTYPE} and saving to {OUTPUT_FILE}...")
    
    # Convert the giant list to a numpy array
    final_token_array = np.array(all_token_ids, dtype=DTYPE)
    
    # Save to binary file
    with open(OUTPUT_FILE, 'wb') as f:
        f.write(final_token_array.tobytes())
        
    print(f"Successfully saved {total_tokens:,} tokens to {OUTPUT_FILE}")
    print(f"File size: {len(final_token_array.tobytes()) / 1e9:.2f} GB")
    print("Dataset build complete.")

if __name__ == "__main__":
    main()
