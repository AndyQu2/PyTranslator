from typing import Optional

import torch
from torch import nn
from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_kv: torch.Tensor, dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, dst_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)
        tmp = self.attention(x, encoder_kv, encoder_kv, src_dst_mask)
        tmp = self.dropout2(tmp)
        x = self.norm2(x + tmp)
        tmp = self.ffn(x)
        tmp = self.dropout3(tmp)
        x = self.norm3(x + tmp)
        return x