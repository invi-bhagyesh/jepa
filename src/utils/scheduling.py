import math
from torch.optim.lr_scheduler import LambdaLR


def cosine_ema_schedule(step, total_steps, base=0.996, final=1.0):
    progress = step / max(1, total_steps)
    return final - (final - base) * (1 + math.cos(math.pi * progress)) / 2


def get_cosine_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
