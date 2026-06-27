from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from src.models.transformer_block import Block, RMSNorm


class Transformer(nn.Module):
    """
    Improved Transformer model with KV-cache, weight tying, dropout,
    configurable normalization, and efficient generation.

    Args:
        n_head (int): The number of attention heads in each transformer block.
        n_embed (int): The dimensionality of the embedding space.
        context_length (int): The maximum length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        N_BLOCKS (int): The number of transformer blocks in the model.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        norm_type (str, optional): Normalization type ('layernorm' or 'rmsnorm'). Defaults to 'layernorm'.
        activation (str, optional): MLP activation ('gelu' or 'swiglu'). Defaults to 'gelu'.
        tie_weights (bool, optional): Whether to tie token embedding and lm_head weights. Defaults to True.
    """
    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int,
        vocab_size: int,
        N_BLOCKS: int,
        dropout: float = 0.0,
        norm_type: str = 'layernorm',
        activation: str = 'gelu',
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        # --- Config validation ---
        self._validate_config(n_head, n_embed, context_length, vocab_size, N_BLOCKS)

        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights

        # Opt-in gradient checkpointing
        self.gradient_checkpointing = False

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.embed_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Transformer blocks
        self.attn_blocks = nn.ModuleList([
            Block(n_head, n_embed, context_length, dropout=dropout, norm_type=norm_type, activation=activation)
            for _ in range(N_BLOCKS)
        ])

        # Final normalization
        if norm_type.lower() == 'rmsnorm':
            self.layer_norm = RMSNorm(n_embed)
        else:
            self.layer_norm = nn.LayerNorm(n_embed)

        # Language modeling head
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: share token embedding and lm_head weights
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        # Positional indices buffer
        self.register_buffer('pos_idxs', torch.arange(context_length))

        # GPT-2 style weight initialization
        self.apply(self._init_weights)

    def _validate_config(self, n_head, n_embed, context_length, vocab_size, N_BLOCKS):
        """Validate model configuration to catch errors early."""
        assert n_embed % n_head == 0, (
            f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        )
        assert context_length > 0, f"context_length must be positive, got {context_length}"
        assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
        assert N_BLOCKS > 0, f"N_BLOCKS must be positive, got {N_BLOCKS}"
        assert n_embed > 0, f"n_embed must be positive, got {n_embed}"
        assert n_head > 0, f"n_head must be positive, got {n_head}"

    def _init_weights(self, module):
        """GPT-2 style initialization for transformer weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Combines token and position embeddings with dropout.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).

        Returns:
            torch.Tensor: Sum of token and position embeddings with dropout applied.
        """
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        x = self.embed_dropout(tok_embedding + pos_embedding)
        return x

    def forward_hidden(self, idx: torch.Tensor, kv_caches: list = None) -> torch.Tensor:
        """
        Run the backbone and return the final hidden states.

        Args:
            idx (torch.Tensor): Input token indices, shape (B, T).
            kv_caches (list, optional): List of KV-cache dicts for each block.

        Returns:
            torch.Tensor: Final hidden states, shape (B, T, n_embed).
        """
        x = self._pre_attn_pass(idx)
        for i, block in enumerate(self.attn_blocks):
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                cache = kv_caches[i] if kv_caches is not None else None
                x = block(x, kv_cache=cache)
        return self.layer_norm(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the Transformer.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).
            targets (torch.Tensor, optional): Target token indices for loss calculation.

        Returns:
            tuple: Logits of shape (B, T, vocab_size) and loss (if targets provided).
        """
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
        return logits, loss

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass focusing on the embedding and attention blocks.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            tuple: Output after attention blocks and the residual.
        """
        x = self._pre_attn_pass(idx)
        residual = x
        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)
        return x, residual

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generates new tokens given a starting sequence with KV-cache and advanced sampling.

        Args:
            idx (torch.Tensor): Initial sequence of token indices, shape (B, T).
            max_new_tokens (int): Number of tokens to generate.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): If set, only sample from the top k tokens. Defaults to None.
            top_p (float, optional): If set, use nucleus sampling with this cumulative probability. Defaults to None.
            use_kv_cache (bool, optional): Whether to use KV-cache for efficient generation. Defaults to True.

        Returns:
            torch.Tensor: The extended sequence of tokens.
        """
        self.eval()

        # Initialize KV-caches if enabled
        kv_caches = [{} for _ in range(self.N_BLOCKS)] if use_kv_cache else None

        for _ in range(max_new_tokens):
            # Use full context only for the first step if using KV-cache
            if use_kv_cache and len(kv_caches[0]) > 0:
                # Only feed the last token
                idx_input = idx[:, -1:]
            else:
                idx_input = idx[:, -self.context_length:]

            # Forward pass
            if use_kv_cache:
                logits = self.lm_head(self.forward_hidden(idx_input, kv_caches=kv_caches))
            else:
                logits, _ = self(idx_input)

            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def get_perplexity(self, idx: torch.Tensor, targets: torch.Tensor = None) -> float:
        """
        Calculate perplexity for the given input.

        Args:
            idx (torch.Tensor): Input token indices.
            targets (torch.Tensor, optional): Target token indices.

        Returns:
            float: Perplexity value.
        """
        self.eval()
        _, loss = self(idx, targets=targets if targets is not None else idx)
        perplexity = math.exp(loss.item())
        return perplexity

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        For non-embedding count, only count parameters in the transformer blocks and lm_head.
        Weight-tied parameters are counted only once.

        Args:
            non_embedding (bool): If True, exclude position and token embedding parameters.

        Returns:
            int: Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # When weight tying is used, token_embed and lm_head share weights
            # so we don't double-count
            n_params -= self.position_embed.weight.numel()
        return n_params


if __name__ == '__main__':
    batch_size = 2
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32
    num_heads = 4
    num_blocks = 2
    context_len = 5
    input_indices = torch.randint(0, vocab_size, (batch_size, sequence_length))

    transformer_model = Transformer(
        n_head=num_heads, n_embed=embedding_dim, context_length=context_len,
        vocab_size=vocab_size, N_BLOCKS=num_blocks,
        dropout=0.1, norm_type='layernorm', activation='gelu', tie_weights=True
    )

    total_params = transformer_model.get_num_params(non_embedding=False)
    print(f"Total parameters: {total_params:,}")

    logits, loss = transformer_model(input_indices, targets=input_indices)
    print("Transformer Logits Shape:", logits.shape)
    print("Transformer Loss:", loss)

    # Test perplexity
    ppl = transformer_model.get_perplexity(input_indices, input_indices)
    print(f"Perplexity: {ppl:.2f}")

    # Test generation with KV-cache
    start_indices = input_indices[:, :1]
    generated = transformer_model.generate(start_indices, max_new_tokens=5, use_kv_cache=True)
    print("Generated (with KV-cache) Shape:", generated.shape)

    # Test generation with top-k and top-p
    generated_sampling = transformer_model.generate(
        start_indices, max_new_tokens=5, top_k=10, top_p=0.9, temperature=0.8
    )
    print("Generated (top-k=10, top-p=0.9) Shape:", generated_sampling.shape)

    # Test without KV-cache
    generated_no_cache = transformer_model.generate(
        start_indices, max_new_tokens=5, use_kv_cache=False
    )
    print("Generated (no KV-cache) Shape:", generated_no_cache.shape)
