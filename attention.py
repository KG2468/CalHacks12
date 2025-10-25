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

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.mha = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_dropout, batch_first=True)
        self.proj_dropout = nn.Dropout(attn_dropout)
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
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd=n_embd, n_head=n_head, ff_hidden=ff_hidden, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _causal_mask(self, T, device):
        # mask shape (T, T) with float mask where True -> -inf
        mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, idx):
        # idx: (B, T) token ids
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds model block size"

        device = idx.device
        tok = self.tok_emb(idx)                   # (B, T, E)
        pos = self.pos_emb(torch.arange(T, device=device))[None, :, :]  # (1, T, E)
        x = self.drop(tok + pos)

        attn_mask = self._causal_mask(T, device)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T) initial context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] if idx.size(1) > self.block_size else idx
            logits = self.forward(idx_cond)  # (B, T_cond, V)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                min_v = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_v, torch.full_like(logits, float('-inf')), logits)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


if __name__ == "__main__":
    # quick smoke test
    B, T = 2, 16
    vocab = 10000
    model = SimpleAttentionLM(vocab_size=vocab, block_size=64, n_layer=2, n_head=4, n_embd=128)
    

    for module_name, module in model.named_modules():
            print("Module:", module_name, type(module))
            # list direct params (your code)
            direct = list(module.named_parameters(recurse=False))
            # list all params (module + submodules)
            all_params = list(module.named_parameters(recurse=True))
            # just Parameter objects iterator
            params_iter = list(module.parameters())
            # compare counts
            print(len(direct), len(all_params), len(list(module.parameters())))
            # inspect state dict (includes buffers too)
            # print(list(module.state_dict().keys())[:20])

    x = torch.randint(0, vocab, (B, T))
    logits = model(x)
    print("logits shape:", logits.shape)  # expected (B, T, vocab)
    out = model.generate(x, max_new_tokens=10, temperature=1.0, top_k=50)
    print("generated shape:", out.shape)