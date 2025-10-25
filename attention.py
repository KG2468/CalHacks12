import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Simple 2-layer MLP with GELU activation and dropout."""
    def __init__(self, n_embd, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal (autoregressive) self-attention using nn.MultiheadAttention (batch_first)."""
    def __init__(self, n_embd, n_head, attn_dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False) 
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, kv_cache=None, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_key, past_value = kv_cache
            k = torch.cat((past_key, k), dim=2)
            v = torch.cat((past_value, v), dim=2)

        new_kv_cache = (k.detach(), v.detach())

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False 
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        return attn_output, new_kv_cache


class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FeedForward -> Residual"""
    def __init__(self, n_embd, n_head, ff_hidden, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, ff_hidden, dropout)

    def forward(self, x, kv_cache=None, attn_mask=None):
        attn_output, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, attn_mask=attn_mask)
        x = x + attn_output
        x = x + self.ff(self.ln2(x))
        return x, new_kv_cache


class SimpleAttentionLM(nn.Module):
    """Minimal autoregressive language model with causal attention and 2-layer MLP in each block."""
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=6,
        n_head=8,
        n_embd=512,
        ff_hidden=None,
        dropout=0.1,
    ):
        super().__init__()
        if ff_hidden is None:
            ff_hidden = 4 * n_embd
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer # <-- Store n_layer
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd=n_embd, n_head=n_head, ff_hidden=ff_hidden, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _causal_mask(self, T, device):
        mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(
        self, 
        idx: torch.Tensor, 
        past_kv_caches
    ):
        """
        Main forward pass.
        - If past_kv_caches is None (TRAINING / PREFILL):
          Runs a full forward pass. `attn_mask` is created.
        - If past_kv_caches is not None (GENERATION):
          Runs an incremental forward pass. `attn_mask` is None.
        """
        B, T = idx.shape
        device = idx.device
        
        if past_kv_caches is None:
            # Training or Prefill: T can be > 1
            assert T <= self.block_size, f"Sequence length {T} exceeds model block size {self.block_size}"
            pos_ids = torch.arange(T, device=device)
            attn_mask = self._causal_mask(T, device)
        else:
            # Generation: T can be 1 or more (for the test script)
            past_len = past_kv_caches[0][0].shape[2] # Get seq len from cache
            pos_ids = torch.arange(past_len, past_len + T, device=device)
            if pos_ids.shape[0] > self.block_size:
                 raise ValueError(f"Sequence length {pos_ids.shape[0]} exceeds model block size {self.block_size}")
            attn_mask = None

        tok = self.tok_emb(idx)  # (B, T, E)
        pos = self.pos_emb(pos_ids)[None, :, :] # (1, T, E)
        x = self.drop(tok + pos)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            block_kv_cache = past_kv_caches[i] if past_kv_caches is not None else None
            # --- FIX: Correctly unpack the tuple returned by block ---
            x, new_kv_cache = block(x, kv_cache=block_kv_cache, attn_mask=attn_mask)
            new_kv_caches.append(new_kv_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Always return logits and the new cache
        return logits, new_kv_caches

    # --- FIX: Added _sample_logits helper function ---
    def _sample_logits(self, logits, temperature, top_k):
        """Helper function to sample from logits."""
        logits = logits / (temperature if temperature > 0 else 1.0)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_v = v[:, -1].unsqueeze(1)
            logits = torch.where(logits < min_v, torch.full_like(logits, float('-inf')), logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # (B, 1)

    # --- FIX: Updated generate to use the KV cache ---
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        kv_caches = None
        
        # --- Prefill Phase ---
        # Process the initial context (idx) all at once
        # This pass populates the *first* KV cache
        logits, kv_caches = self.forward(idx, past_kv_caches=None)
        
        # Get the logits for the very last token
        next_logits = logits[:, -1, :] # (B, V)
        
        # Sample the first new token
        next_tok = self._sample_logits(next_logits, temperature, top_k) # (B, 1)
        
        # Add to our running sequence
        idx = torch.cat([idx, next_tok], dim=1)

        # --- Generation Phase ---
        for _ in range(max_new_tokens - 1):
            # Now, we only pass the *newest* token (next_tok)
            # and the cache we just got (kv_caches)
            logits, kv_caches = self.forward(next_tok, past_kv_caches=kv_caches)
            
            # Get the logits for this single token
            next_logits = logits[:, -1, :] # (B, V)
            
            # Sample the *next* token
            next_tok = self._sample_logits(next_logits, temperature, top_k) # (B, 1)
            
            # Append it to the sequence
            idx = torch.cat([idx, next_tok], dim=1)
            
        self.train() # Set model back to training mode
        return idx

if __name__ == "__main__":
    # (The test script is unchanged)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Optional, Tuple, List # Make sure these are imported at the top

    # --- Test Config ---
    B, T = 4, 16
    VOCAB = 1000
    BLOCK_SIZE = 64
    N_LAYER = 2
    N_HEAD = 4
    N_EMBD = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"--- Running Tests on {DEVICE} ---")

    # --- 1. Smoke Test (Instantiation & Shape Check) ---
    print("\n--- Test 1: Smoke Test ---")
    try:
        model = SimpleAttentionLM(
            vocab_size=VOCAB, 
            block_size=BLOCK_SIZE, 
            n_layer=N_LAYER, 
            n_head=N_HEAD, 
            n_embd=N_EMBD
        ).to(DEVICE)
        
        x = torch.randint(0, VOCAB, (B, T), device=DEVICE)
        
        # Test training pass
        logits, caches = model(x, past_kv_caches=None)
        print(f"Training logits shape: {logits.shape}")
        assert logits.shape == (B, T, VOCAB)
        assert caches is not None, "Caches should be returned even from prefill"
        assert len(caches) == N_LAYER
        
        # Test generation
        out = model.generate(x, max_new_tokens=10)
        print(f"Generated shape: {out.shape}")
        assert out.shape == (B, T + 10)
        
        print("✅ Smoke Test Passed: Model instantiated and runs.")
    
    except Exception as e:
        print(f"❌ Smoke Test FAILED: {e}")
        import traceback
        traceback.print_exc()


    # --- 2. KV Cache Correctness Test (Numerical Check) ---
    print("\n--- Test 2: KV Cache Correctness Test ---")
    try:
        model.eval() # CRITICAL: Disable dropout for numerical comparison
        
        # Create a prompt
        prompt_len = 8
        prompt = torch.randint(0, VOCAB, (1, prompt_len), device=DEVICE)

        # Path A: Get "ground truth" logits using the (non-cached) training-style pass
        # We want the logits for the *last* token
        logits_full, _ = model(prompt, past_kv_caches=None)
        target_logits = logits_full[:, -1, :] # Logits for token at T=7

        # Path B: Get logits using the KV cache token-by-token
        # 1. Prefill on the first (T-1) tokens
        prefill_prompt = prompt[:, :-1] # Tokens 0 to 6
        _, cache = model(prefill_prompt, past_kv_caches=None)
        
        # 2. Generate the *last* token using the cache
        last_token = prompt[:, -1:] # Token 7
        logits_cached, _ = model(last_token, past_kv_caches=cache)
        cached_logits = logits_cached[:, -1, :] # Logits for token at T=7

        # 3. Compare
        are_close = torch.allclose(target_logits, cached_logits, atol=1e-5)
        print(f"Logits are close: {are_close}")
        assert are_close, "KV cache logits do not match ground truth logits"
        
        print("✅ KV Cache Test Passed: Inference logic is numerically correct.")
    
    except Exception as e:
        print(f"❌ KV Cache Test FAILED: {e}")
        import traceback
        traceback.print_exc()


    # --- 3. Overfit Test (Gradient Check) ---
    print("\n--- Test 3: Overfit Test ---")
    try:
        model.train() # Set back to train mode
        
        # A single, tiny batch
        X_tiny = torch.randint(0, VOCAB, (2, 8), device=DEVICE) # input
        Y_tiny = torch.randint(0, VOCAB, (2, 8), device=DEVICE) # target
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        print("Training on one batch for 50 steps...")
        initial_loss = -1.0
        final_loss = -1.0
        
        for i in range(50):
            optimizer.zero_grad()
            logits, _ = model(X_tiny, past_kv_caches=None)
            loss = loss_fn(logits.view(-1, VOCAB), Y_tiny.view(-1))
            
            if i == 0:
                initial_loss = loss.item()
            if i == 49:
                final_loss = loss.item()
                
            loss.backward()
            optimizer.step()
            
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss:   {final_loss:.4f}")
        
        assert final_loss < initial_loss / 5, "Loss did not decrease significantly"
        print("✅ Overfit Test Passed: Model can learn (gradients are flowing).")

    except Exception as e:
        print(f"❌ Overfit Test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- All Tests Finished ---")