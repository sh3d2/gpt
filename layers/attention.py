import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, idx):
        B, T, C = idx.shape
        k = self.key(idx)
        q = self.query(idx)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(idx)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(config, head_size) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.residual_dropout(self.proj(out))
        return out
