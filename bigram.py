import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        #construct a lookup table where each row corresponds to each token
        #and contains the logits for the next tokcn
        self.embedding_table= nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx:torch.Tensor, target:torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        #look up the logits for the next token
        logits = self.embedding_table(idx)

        if target is None:
            loss = None
        else:
            #compute the loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            loss = F.cross_entropy(logits, target.view(-1))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_tokens:int) -> torch.Tensor:
        #generate tokens
        with torch.no_grad():
            for _ in range(max_tokens):
                logits, loss = self.forward(idx)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                idx = torch.cat((idx, next_token), dim=1)
            return idx