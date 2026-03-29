import math
import torch
import torch.nn as nn
from .encoder import PositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4,
                 dim_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerDecoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def _causal_mask(self, sz, device):
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device)

    def forward(self, tgt, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.layers(
            x, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.norm(x)
