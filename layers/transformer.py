import torch.nn as nn
from .attention import FeedForward
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_heads
        self.attention = MultiHeadAttention(config, head_size)
        self.ln = nn.LayerNorm(config.n_embed)
        self.ffwd = FeedForward(config)

        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, idx):
        idx = idx + self.attention(self.ln(idx))
        idx = idx + self.ffwd(self.ln2(idx))
        return idx


def transformer(config):
    return nn.ModuleDict(dict(
        wte=nn.Embedding(config.vocab_size, config.n_embed),
        wpe=nn.Embedding(config.block_size, config.n_embed),
        head=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
        layer_norm=nn.LayerNorm(config.n_embed)
    ))
