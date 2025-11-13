import torch
from torch import nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class config:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    vocab_size: int = 50257
    block_size: int = 256 # max_seq_len


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embd, 3*config.n_embd)
        self.linear2 = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        qkv = self.linear1(x)
        q, k, v = qkv.split(config.n_embd, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.linear2(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffd = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embd = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.unembd = nn.Linear(config.n_embd, config.vocab_size)
        self.token_embd.weight = self.unembd.weight #weight sharing

    def forward(self, inputs):
        B, T = inputs.shape
        pos = torch.arange(0, T, dtype=torch.long, device=inputs.device)
        pos_embd = self.pos_embd(pos.unsqueeze(0)) #shape: (1, T, n_embd)
        token_embd = self.token_embd(inputs) #shape: (B, T, n_embd)
        x = token_embd + pos_embd
        for block in self.blocks:
            x = block(x)
        x = self.unembd(self.ln(x))
        return x