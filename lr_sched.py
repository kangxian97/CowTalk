import math


def adjust_learning_rate(
    optimizer,
    progress: float,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    step_decay_epochs: int = 80,
    step_decay_factor: float = 0.8,
):
    """
    Cosine schedule with warmup and step decay.
    - Warmup phase: linear increase
    - After warmup: cosine decay
    - Step decay: multiply by step_decay_factor every step_decay_epochs
    `progress` can be fractional (epoch + iteration progress).
    """
    # Apply warmup
    if progress < warmup_epochs:
        curr_lr = lr * progress / float(max(1, warmup_epochs))
    else:
        ratio = (progress - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        ratio = min(1.0, max(0.0, ratio))
        curr_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * ratio))
    
    # Apply step decay: multiply by step_decay_factor every step_decay_epochs
    # Only apply after warmup is complete
    if progress >= warmup_epochs:
        # Calculate how many step decay periods have passed
        epochs_after_warmup = progress - warmup_epochs
        num_steps = int(epochs_after_warmup / step_decay_epochs)
        # Apply step decay multiplier
        step_decay_multiplier = step_decay_factor ** num_steps
        curr_lr = curr_lr * step_decay_multiplier
        # Ensure lr doesn't go below min_lr
        curr_lr = max(curr_lr, min_lr)

    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = curr_lr * scale
    return curr_lr

