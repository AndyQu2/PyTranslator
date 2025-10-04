from typing import Optional
import torch
from torch import nn
from Transformer.attention import attention

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert q.shape[0] == k.shape[0]
        assert q.shape[0] == v.shape[0]
        assert k.shape[0] == v.shape[0]

        n, q_len = q.shape[0:2]
        n, k_len = k.shape[0:2]
        q_ = self.q(q).reshape(n, q_len, self.heads, self.d_k).transpose(1, 2)
        k_ = self.k(k).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)
        v_ = self.v(v).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)

        attention_res = attention(q_, k_, v_, mask)
        concat_res = attention_res.transpose(1, 2).reshape(n, q_len, self.d_model)
        concat_res = self.dropout(concat_res)

        output = self.out(concat_res)
        return output