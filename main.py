import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from bigram import BigramLanguageModel

with open("verne.txt", "r") as f:
    text = f.read()

vocab_size = len(set(text))
batch_size = 32
block_size = 8
device = torch.device("mps")
learning_rate = 1e-3
total_iters = 10000
eval_iters = total_iters // 10

# construct a character level tokenizer
ctoi = {c: i for i, c in enumerate(set(text))}
itoc = {i: c for i, c in enumerate(set(text))}
encode = lambda x: [ctoi[c] for c in x]
decode = lambda x: "".join([itoc[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


torch.manual_seed(1337)

model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
progress_bar = tqdm(range(total_iters))

# train the model
for i in progress_bar:
    model.train()
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % eval_iters == 0:
        losses = eval_loss(model)
        progress_bar.set_postfix(losses)

# generate some text
model.eval()
print("\n\nGenerated text:\n")
print(
    decode(
        model.generate(torch.zeros(1, 1, dtype=torch.long, device=device), 500)[
            0
        ].tolist()
    )
)
