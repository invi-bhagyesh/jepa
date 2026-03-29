import torch
import torch.nn.functional as F
from tqdm import tqdm
from ..utils.masking import batch_span_masks
from ..utils.scheduling import cosine_ema_schedule, get_cosine_lr_scheduler


def pretrain_jepa(model, train_loader, epochs=50, lr=1e-4, weight_decay=0.01,
                  mask_ratio=0.35, warmup_steps=500, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_lr_scheduler(optimizer, warmup_steps, total_steps)

    history = {"loss": []}
    step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"JEPA epoch {epoch+1}/{epochs}")

        for src, _ in pbar:
            src = src.to(device)
            B, T = src.shape
            padding_mask = (src == 0)

            # generate masks, exclude padding positions
            masks = batch_span_masks(B, T, mask_ratio=mask_ratio)
            masks = masks & ~padding_mask.cpu()
            masks = masks.to(device)

            if masks.sum() == 0:
                continue

            predicted, target = model(src, masks, padding_mask=padding_mask)
            loss = F.smooth_l1_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            momentum = cosine_ema_schedule(step, total_steps)
            model.update_target(momentum)

            epoch_loss += loss.item()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        history["loss"].append(avg_loss)
        print(f"  Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

    return history
