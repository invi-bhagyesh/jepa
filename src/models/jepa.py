import copy
import torch
import torch.nn as nn
from .encoder import TransformerEncoder


class JEPA(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4,
                 dim_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.context_encoder = TransformerEncoder(
            vocab_size, d_model, nhead, num_layers, dim_ff, dropout, max_len
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.mask_token = nn.Parameter(torch.randn(d_model))

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    @torch.no_grad()
    def update_target(self, momentum):
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1 - momentum)

    def forward(self, src, mask_indices, padding_mask=None):
        # target representations (full sequence, no masking)
        with torch.no_grad():
            target_repr = self.target_encoder(src, src_key_padding_mask=padding_mask)

        # build masked input embeddings
        import math
        emb = self.context_encoder.embedding(src) * math.sqrt(self.context_encoder.d_model)
        emb[mask_indices] = self.mask_token

        x = self.context_encoder.pos_enc(emb)
        x = self.context_encoder.layers(x, src_key_padding_mask=padding_mask)
        x = self.context_encoder.norm(x)

        predicted = self.predictor(x[mask_indices])
        target = target_repr[mask_indices].detach()

        return predicted, target
