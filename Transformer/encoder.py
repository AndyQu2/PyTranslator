from typing import Optional
import torch
from torch import nn
from Transformer.encoder_layer import EncoderLayer
from Transformer.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, d_model: int,
                 d_ff: int, n_layers: int, heads: int, dropout: float = 0.1, max_seq_len: int = 120):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(EncoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x