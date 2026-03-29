import torch
from tqdm import tqdm


@torch.no_grad()
def token_accuracy(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_pad = (src == 0)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_pad_in = (tgt_in == 0)

        logits = model(src, tgt_in, src_padding_mask=src_pad,
                       tgt_padding_mask=tgt_pad_in)
        preds = logits.argmax(dim=-1)

        mask = tgt_out != 0
        correct += ((preds == tgt_out) & mask).sum().item()
        total += mask.sum().item()

    return correct / max(1, total)


@torch.no_grad()
def sequence_exact_match(model, loader, tokenizer, max_len=256, device="cpu"):
    model.eval()
    correct, total = 0, 0
    for src, tgt in tqdm(loader, desc="Exact match eval"):
        src, tgt = src.to(device), tgt.to(device)
        src_pad = (src == 0)

        for i in range(src.size(0)):
            s = src[i:i+1]
            sp = src_pad[i:i+1]
            pred = model.generate(s, tokenizer.sos_id, tokenizer.eos_id,
                                  max_len=max_len, beam_width=1,
                                  src_padding_mask=sp)
            pred_ids = pred[0, 1:].tolist()  # strip SOS
            tgt_ids = tgt[i].tolist()
            # strip padding and EOS from target
            tgt_ids = [t for t in tgt_ids if t != 0]

            if pred_ids == tgt_ids:
                correct += 1
            total += 1

    return correct / max(1, total)


@torch.no_grad()
def top_k_accuracy(model, loader, tokenizer, k=5, max_len=256, device="cpu"):
    model.eval()
    correct, total = 0, 0
    for src, tgt in tqdm(loader, desc=f"Top-{k} eval"):
        src, tgt = src.to(device), tgt.to(device)
        src_pad = (src == 0)

        for i in range(src.size(0)):
            s = src[i:i+1]
            sp = src_pad[i:i+1]
            memory = model.encoder(s, src_key_padding_mask=sp)
            beams = model._beam_decode(memory, tokenizer.sos_id,
                                       tokenizer.eos_id, max_len, k, sp)
            tgt_ids = [t for t in tgt[i].tolist() if t != 0]

            hit = any(
                b[0, 1:].tolist() == tgt_ids for b in beams
            )
            if hit:
                correct += 1
            total += 1

    return correct / max(1, total)


def evaluate_model(model, test_loader, tokenizer, device="cpu"):
    tok_acc = token_accuracy(model, test_loader, device)
    exact = sequence_exact_match(model, test_loader, tokenizer, device=device)
    top5 = top_k_accuracy(model, test_loader, tokenizer, k=5, device=device)
    return {
        "token_accuracy": tok_acc,
        "sequence_exact_match": exact,
        "top_5_accuracy": top5,
    }
