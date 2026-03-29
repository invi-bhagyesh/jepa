import torch


def generate_span_mask(seq_len, mask_ratio=0.35, num_spans=3, min_span=2):
    """Generate a boolean mask with contiguous spans covering ~mask_ratio of seq_len."""
    total_masked = max(1, int(seq_len * mask_ratio))
    mask = torch.zeros(seq_len, dtype=torch.bool)

    if total_masked < min_span:
        start = torch.randint(0, seq_len, (1,)).item()
        end = min(start + total_masked, seq_len)
        mask[start:end] = True
        return mask

    # distribute masked tokens across spans
    per_span = max(min_span, total_masked // num_spans)
    placed = 0
    attempts = 0
    while placed < total_masked and attempts < 50:
        span_len = min(per_span, total_masked - placed)
        start = torch.randint(0, max(1, seq_len - span_len), (1,)).item()
        mask[start:start + span_len] = True
        placed = mask.sum().item()
        attempts += 1

    return mask


def batch_span_masks(batch_size, seq_len, mask_ratio=0.35, num_spans=3):
    """Generate span masks for a batch, returned as boolean tensor [B, T]."""
    return torch.stack([
        generate_span_mask(seq_len, mask_ratio, num_spans) for _ in range(batch_size)
    ])
