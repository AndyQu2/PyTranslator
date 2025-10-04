from typing import Optional
import torch
from torch import nn
from .feed_forward import FeedForward
from .multihead_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, src_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)
        tmp = self.ffn(x)
        tmp = self.dropout2(tmp)
        x = self.norm2(tmp + x)
        return x
