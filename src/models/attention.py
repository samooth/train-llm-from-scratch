import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Head(nn.Module):
    """
    A single attention head.

    This module calculates attention scores and applies them to the values.
    It includes key, query, and value projections, and uses causal masking
    to prevent attending to future tokens.

    Args:
        head_size (int): The dimensionality of the key, query, and value projections.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence, used for causal masking.
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
    """
    def __init__(self, head_size: int, n_embed: int, context_length: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        head_size = self.key.out_features
        k = self.key(x)
        q = self.query(x)
        scale_factor = 1.0 / math.sqrt(head_size)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        v = self.value(x)
        out = attn_weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module with efficient single QKV projection.

    This implementation uses a single linear projection for Q, K, V followed by
    reshaping, which is significantly more efficient than separate per-head projections.
    It supports optional KV-caching for efficient autoregressive generation.

    Args:
        n_head (int): The number of parallel attention heads.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
        dropout (float, optional): Dropout probability for attention weights and residual. Defaults to 0.0.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert n_embed % n_head == 0, f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_size = n_embed // n_head
        self.context_length = context_length

        # Single QKV projection - much more efficient than per-head projections
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        # Output projection
        self.proj = nn.Linear(n_embed, n_embed)
        # Dropout for attention weights and residual
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Causal mask buffer
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor, kv_cache: dict = None) -> torch.Tensor:
        """
        Forward pass through multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            kv_cache (dict, optional): KV-cache dictionary with 'k' and 'v' tensors
                                       for efficient autoregressive generation.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape

        # Single QKV projection and split
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embed, dim=-1)

        # Reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Handle KV-cache for efficient generation
        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache:
                k = torch.cat([kv_cache['k'], k], dim=2)
                v = torch.cat([kv_cache['v'], v], dim=2)
            kv_cache['k'] = k
            kv_cache['v'] = v

        # Scaled dot-product attention
        scale_factor = 1.0 / math.sqrt(self.head_size)
        attn_weights = (q @ k.transpose(-2, -1)) * scale_factor

        # Apply causal mask
        if kv_cache is not None:
            # With KV-cache, we only have 1 new token but k contains all previous
            # No masking needed since all previous tokens are valid to attend to
            pass
        else:
            # Full sequence training - apply causal mask
            attn_weights = attn_weights.masked_fill(
                self.mask[:T, :T].unsqueeze(0).unsqueeze(0),
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # (B, n_head, T, head_size)

        # Reshape back: (B, n_head, T, head_size) -> (B, T, n_embed)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with residual dropout
        out = self.resid_dropout(self.proj(out))
        return out

    def forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy forward using per-head projections (for comparison/testing)."""
        B, T, C = x.shape
        head_size = C // self.n_head
        # Manually split for legacy compatibility
        qkv_out = self.qkv(x)
        q, k, v = qkv_out.split(C, dim=-1)
        # Reshape and process each head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        scale_factor = 1.0 / math.sqrt(head_size)
        attn_weights = (q @ k.transpose(-2, -1)) * scale_factor
        attn_weights = attn_weights.masked_fill(
            self.mask[:T, :T].unsqueeze(0).unsqueeze(0),
            float('-inf')
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


if __name__ == '__main__':
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    multihead_attn = MultiHeadAttention(n_head=num_heads, n_embed=embedding_dim, context_length=context_len, dropout=0.1)
    output_tensor = multihead_attn(input_tensor)

    print("MultiHeadAttention Input Shape:", input_tensor.shape)
    print("MultiHeadAttention Output Shape:", output_tensor.shape)

    # Test KV-cache
    print("\n--- KV-Cache Test ---")
    cache = {}
    token_1 = input_tensor[:, :1, :]
    out_1 = multihead_attn(token_1, kv_cache=cache)
    print(f"Step 1 - Input: {token_1.shape}, Output: {out_1.shape}, Cache k: {cache['k'].shape}")

    token_2 = input_tensor[:, 1:2, :]
    out_2 = multihead_attn(token_2, kv_cache=cache)
    print(f"Step 2 - Input: {token_2.shape}, Output: {out_2.shape}, Cache k: {cache['k'].shape}")

    token_3 = input_tensor[:, 2:3, :]
    out_3 = multihead_attn(token_3, kv_cache=cache)
    print(f"Step 3 - Input: {token_3.shape}, Output: {out_3.shape}, Cache k: {cache['k'].shape}")
