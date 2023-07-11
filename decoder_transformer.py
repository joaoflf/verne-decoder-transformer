from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderTransformer(nn.Module):
    def __init__(
        self, num_heads: int, embed_size: int, block_size: int, vocab_size: int
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        head_size = embed_size // num_heads
        self.multi_head_attention = MultiHeadAttention(
            num_heads, head_size, embed_size, block_size
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        token_embedding = self.token_embedding_table(x)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=x.device)
        )
        x = token_embedding + position_embedding
        x = self.multi_head_attention(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            logits = logits.view(B * T, self.vocab_size)
            loss = F.cross_entropy(logits, target.view(-1))

        return logits, loss

    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        # generate tokens
        with torch.no_grad():
            for i in range(max_tokens):
                cond_idx = idx[:, -self.block_size :]
                logits, _ = self.forward(cond_idx)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                idx = torch.cat((idx, next_token), dim=1)
            return idx


class MultiHeadAttention(nn.Module):
    """
    A multi-head attention layer.
    Takees in a number of heads retruen a concatenated output of all heads.
    """

    def __init__(
        self, num_heads: int, head_size: int, embed_size: int, block_size: int
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embed_size, block_size) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


class Head(nn.Module):
    """
    A single head of a multi-head attention layer.
    """

    def __init__(self, head_size: int, embed_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.queries = nn.Linear(embed_size, head_size, bias=False)
        self.keys = nn.Linear(embed_size, head_size, bias=False)
        self.values = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the head. Takes in a batch of sequences,
        computes the queries, keys, and values, and then computes the attention weights.
        Returns the weighted sum of the values.
        """
        B, T, C = x.shape
        q = self.queries(x)  # (B, T, H)
        k = self.keys(x)  # (B, T, H)
        v = self.values(x)  # (B, T, H)
        weights = (
            q @ k.transpose(-2, -1) / (self.head_size**0.5)
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
        weights = weights.masked_fill(
            self.tril_mask[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        weights = F.softmax(weights, dim=-1)

        return weights @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)
