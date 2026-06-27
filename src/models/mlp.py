import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Transformer blocks.

    Supports standard GELU activation (GPT-2 style) or SwiGLU (Llama/Mistral style).
    SwiGLU uses ~1.5x the expressiveness for the same parameter count.

    Args:
        n_embed (int): The dimensionality of the input embedding.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        activation (str, optional): Activation function - 'gelu' or 'swiglu'. Defaults to 'gelu'.
        hidden_mult (int, optional): Hidden dimension multiplier. Defaults to 4.
    """
    def __init__(self, n_embed: int, dropout: float = 0.0, activation: str = 'gelu', hidden_mult: int = 4) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        self.n_embed = n_embed
        self.hidden_size = hidden_mult * n_embed

        if self.activation_name == 'swiglu':
            # SwiGLU: three linear layers - w1, w2 (gate), and projection
            # Forward: proj(w2(SiLU(w1(x)) * gate(x)))
            # We implement this as w1, w3 (gate), w2 (projection back)
            self.w1 = nn.Linear(n_embed, self.hidden_size, bias=False)
            self.w3 = nn.Linear(n_embed, self.hidden_size, bias=False)  # gate
            self.w2 = nn.Linear(self.hidden_size, n_embed, bias=False)   # projection
        else:
            # Standard GELU (GPT-2 style)
            self.hidden = nn.Linear(n_embed, self.hidden_size)
            self.gelu = nn.GELU()
            self.proj = nn.Linear(self.hidden_size, n_embed)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        x = self.forward_embedding(x)
        x = self.project_embedding(x)
        return x

    def forward_embedding(self, x: Tensor) -> Tensor:
        """
        Applies the hidden transformation (activation + hidden layers).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the hidden transformation.
        """
        if self.activation_name == 'swiglu':
            # SwiGLU: SiLU(w1(x)) * w3(x)
            x = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        else:
            x = self.gelu(self.hidden(x))
        return x

    def project_embedding(self, x: Tensor) -> Tensor:
        """
        Applies the projection linear layer with dropout.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the projection layer.
        """
        if self.activation_name == 'swiglu':
            x = self.w2(x)
        else:
            x = self.proj(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    batch_size = 2
    sequence_length = 3
    embedding_dim = 16
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    # Test GELU
    mlp_gelu = MLP(n_embed=embedding_dim, dropout=0.1, activation='gelu')
    output_gelu = mlp_gelu(input_tensor)
    print("MLP (GELU) Input Shape:", input_tensor.shape)
    print("MLP (GELU) Output Shape:", output_gelu.shape)

    # Test SwiGLU
    mlp_swiglu = MLP(n_embed=embedding_dim, dropout=0.1, activation='swiglu')
    output_swiglu = mlp_swiglu(input_tensor)
    print("MLP (SwiGLU) Input Shape:", input_tensor.shape)
    print("MLP (SwiGLU) Output Shape:", output_swiglu.shape)
