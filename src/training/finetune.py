import torch
import torch.nn as nn
from tqdm import tqdm
from ..utils.scheduling import get_cosine_lr_scheduler


def finetune_seq2seq(model, train_loader, val_loader, epochs=100, lr=3e-4,
                     weight_decay=0.01, warmup_steps=200, patience=10,
                     device="cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_lr_scheduler(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # train
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Finetune {epoch+1}/{epochs}")
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            src_pad = (src == 0)
            tgt_pad = (tgt == 0)

            # teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_pad_in = tgt_pad[:, :-1]

            logits = model(src, tgt_in, src_padding_mask=src_pad,
                           tgt_padding_mask=tgt_pad_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_train)

        # validate
        val_loss = _validate(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        print(f"  Epoch {epoch+1} — train: {avg_train:.4f}, val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    total = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_pad = (src == 0)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_pad_in = (tgt_in == 0)

        logits = model(src, tgt_in, src_padding_mask=src_pad,
                       tgt_padding_mask=tgt_pad_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         tgt_out.reshape(-1))
        total += loss.item()
    return total / max(1, len(loader))
