import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4,
                 num_layers=4, dim_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, nhead, num_layers, dim_ff, dropout, max_len
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, nhead, num_layers, dim_ff, dropout, max_len
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
        dec_out = self.decoder(
            tgt, memory,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output_proj(dec_out)

    @classmethod
    def from_pretrained_encoder(cls, jepa_model, tgt_vocab_size, **kwargs):
        model = cls(
            src_vocab_size=jepa_model.context_encoder.embedding.num_embeddings,
            tgt_vocab_size=tgt_vocab_size,
            **kwargs
        )
        model.encoder.load_state_dict(jepa_model.context_encoder.state_dict())
        return model

    @torch.no_grad()
    def generate(self, src, sos_id, eos_id, max_len=256, beam_width=5,
                 src_padding_mask=None):
        device = src.device
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)

        # greedy if beam_width <= 1
        if beam_width <= 1:
            return self._greedy_decode(memory, sos_id, eos_id, max_len,
                                       src_padding_mask)

        return self._beam_decode(memory, sos_id, eos_id, max_len, beam_width,
                                 src_padding_mask)

    def _greedy_decode(self, memory, sos_id, eos_id, max_len, src_padding_mask):
        B = memory.size(0)
        device = memory.device
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            out = self.decoder(ys, memory,
                               memory_key_padding_mask=src_padding_mask)
            logits = self.output_proj(out[:, -1:])
            next_tok = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return ys

    def _beam_decode(self, memory, sos_id, eos_id, max_len, beam_width,
                     src_padding_mask):
        # single-sample beam search (B=1 assumed for evaluation)
        device = memory.device
        beams = [(torch.tensor([[sos_id]], device=device), 0.0)]

        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[0, -1].item() == eos_id:
                    candidates.append((seq, score))
                    continue
                out = self.decoder(seq, memory,
                                   memory_key_padding_mask=src_padding_mask)
                logits = self.output_proj(out[:, -1])
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_lp, topk_ids = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    tok = topk_ids[0, k].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, tok], dim=1)
                    candidates.append((new_seq, score + topk_lp[0, k].item()))

            beams = sorted(candidates, key=lambda x: -x[1])[:beam_width]
            if all(b[0][0, -1].item() == eos_id for b in beams):
                break

        return [b[0] for b in beams]
