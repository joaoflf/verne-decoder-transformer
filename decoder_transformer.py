import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(
        self, head_size: int, embed_size: int, block_size: int, vocab_size: int
    ):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.queries = nn.Linear(embed_size, head_size, bias=False)
        self.keys = nn.Linear(embed_size, head_size, bias=False)
        self.values = nn.Linear(embed_size, head_size, bias=False)
        self.lm_head = nn.Linear(head_size, vocab_size)
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        token_embedding = self.token_embedding_table(x)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=x.device)
        )
        x = token_embedding + position_embedding
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        weights = q @ k.transpose(-2, -1) / (self.head_size**0.5)
        weights = weights.masked_fill(self.tril_mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        logits = self.lm_head(weights @ v)

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
