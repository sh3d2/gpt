import torch
from encoders.char import CharTokenizer
from models.model import GPT, GPTConfig
import os

filename = "data/shakespeare.txt"

with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()


tokenizer = CharTokenizer(text)
GPTConfig.vocab_size = tokenizer.vocab_size()

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - GPTConfig.block_size, (GPTConfig.batch_size,))
    x = torch.stack([data[i:i + GPTConfig.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + GPTConfig.block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if os.path.isfile('model.bin'):
    model = GPT.from_file('model.bin')
else:
    model = GPT(GPTConfig)

for i in range(100000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    model.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    model.optimizer.step()
    if i % 100 == 0:
        print("step", i, "loss: ", estimate_loss())
    if i % 1000 == 0 or i == 0:
        model.to_file()
