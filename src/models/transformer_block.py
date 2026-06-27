import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.mlp import MLP


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is simpler and often more stable than LayerNorm for deep transformers.
    Used in Llama, Qwen, and other modern architectures.

    Args:
        dim (int): Dimension to normalize over.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class Block(nn.Module):
    """
    A single Transformer block with configurable normalization and dropout.

    This block consists of multi-head attention followed by MLP,
    with pre-normalization and residual connections.

    Args:
        n_head (int): The number of attention heads.
        n_embed (int): The dimensionality of the embedding space.
        context_length (int): The maximum sequence length.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        norm_type (str, optional): Normalization type - 'layernorm' or 'rmsnorm'. Defaults to 'layernorm'.
        activation (str, optional): MLP activation - 'gelu' or 'swiglu'. Defaults to 'gelu'.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int,
                 dropout: float = 0.0, norm_type: str = 'layernorm', activation: str = 'gelu') -> None:
        super().__init__()

        # Choose normalization layer
        norm_layer = nn.LayerNorm if norm_type.lower() == 'layernorm' else RMSNorm

        self.ln1 = norm_layer(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length, dropout=dropout)
        self.ln2 = norm_layer(n_embed)
        self.mlp = MLP(n_embed, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor, kv_cache: dict = None) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            kv_cache (dict, optional): KV-cache for attention layer.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        # Pre-norm: normalize before attention and MLP
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass focusing on the embedding and attention parts.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output after MLP embedding and the residual.
        """
        res = x + self.attn(self.ln1(x))
        x = self.mlp.forward_embedding(self.ln2(res))
        return x, res


if __name__ == '__main__':
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    # Test with LayerNorm
    block_ln = Block(n_head=num_heads, n_embed=embedding_dim, context_length=context_len,
                     dropout=0.1, norm_type='layernorm')
    output_ln = block_ln(input_tensor)
    print("Block (LayerNorm) Output Shape:", output_ln.shape)

    # Test with RMSNorm
    block_rms = Block(n_head=num_heads, n_embed=embedding_dim, context_length=context_len,
                      dropout=0.1, norm_type='rmsnorm')
    output_rms = block_rms(input_tensor)
    print("Block (RMSNorm) Output Shape:", output_rms.shape)

    # Test with KV-cache
    cache = {}
    token = input_tensor[:, :1, :]
    for i in range(3):
        out = block_ln(token, kv_cache=cache)
        print(f"Cache step {i+1}: input={token.shape}, output={out.shape}")
        token = out
