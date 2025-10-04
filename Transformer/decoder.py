from typing import Optional

import torch
from torch import nn
from Transformer.decoder_layer import DecoderLayer
from Transformer.positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, d_model: int, d_ff: int,
                 n_layers: int, heads: int, dropout: float = 0.1, max_len: int = 120):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = []

        for i in range(n_layers):
            self.layers.append(DecoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_kv, dst_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_kv, dst_mask, src_mask)
        return x