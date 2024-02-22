import torch
from time import sleep
import os
from encoders.char import CharTokenizer
from encoders.bpe import BpeTokenizer
from models.model import GPT, GPTConfig

filename = "data/shakespeare.txt"

with open(filename, 'r', encoding='utf-8') as f:  # just to get vocabulary...
    text = f.read()

bpe = BpeTokenizer(300)
bpe.train(text)
bpe.to_file('./data/bpe.bin')
b = BpeTokenizer.from_file('./data/bpe.bin')
print(b.vocab_size())
tokenizer = CharTokenizer(text)

GPTConfig.vocab_size = tokenizer.vocab_size()

model = GPT.from_file('model.bin')

model.eval()
model.to('cpu')

t = (torch.tensor(tokenizer.encode('\n'), dtype=torch.long, device='cpu')[None, ...])
while True:
    g = model.generate(t, max_new_tokens=1)
    os.system('cls' if os.name == 'nt' else 'clear')

    print(tokenizer.decode(g[0].tolist()))
    t = g
    sleep(0.1)
