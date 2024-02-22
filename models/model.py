import torch.nn as nn
import torch
from dataclasses import dataclass, asdict
from layers.transformer import TransformerBlock, transformer
from torch.nn import functional as F

@dataclass
class GPTConfig:
    n_embed = 96
    vocab_size = 65  # just for now..
    block_size = 32
    n_layers = 6
    bias = False
    device = 'cuda' if torch.cuda.is_available() else 'mps' # for macbook
    n_heads = 4
    batch_size = 32  # how many in
    dropout = 0.1

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.n_embed is not None
        self.config = config

        self.transformer = transformer(config)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=5e-3)

        print("Number of model parameters: %.2fMillion(s)" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        token_embed = self.transformer.wte(idx)
        pos_embed = self.transformer.wpe(pos)
        x = token_embed + pos_embed
        for block in self.transformer.head:
            x = block(x)
        x = self.transformer.layer_norm(x)

        logits = self.lm_head(x)
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # print(idx_next)
        return idx

    @classmethod
    def from_file(self, file_path):
        checkpoint = torch.load(file_path, map_location='cpu')
        gptconf = checkpoint['config']
        print(gptconf)
        print(asdict(GPTConfig()))
        state_dict = checkpoint['model']
        model = GPT(gptconf)
        unwanted_prefix = '_orig_mod.'

        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model

    def to_file(self):
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, 'model.bin')
