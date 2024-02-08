import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def lr_with_warmup_helper(optimizer, base_lr, warmup_epochs, epochs, iters_per_epoch, schedule_fn):
    warmup_length = iters_per_epoch * warmup_epochs
    length = iters_per_epoch * epochs

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = length - warmup_length
            # schedule_fn takes progress after warmup in [0, 1]
            lr = schedule_fn(e / es)
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_epochs, epochs, iters_per_epoch, gamma, step_size):
    def schedule_fn(progress):
        return 0.5 * (1 + np.cos(np.pi * progress)) * base_lr
    return lr_with_warmup_helper(optimizer, base_lr, warmup_epochs, epochs, iters_per_epoch, schedule_fn)


def triangle_lr(optimizer, base_lr, warmup_epochs, epochs, iters_per_epoch, gamma, step_size):
    def schedule_fn(progress):
        return base_lr * (1 - progress)
    return lr_with_warmup_helper(optimizer, base_lr, warmup_epochs, epochs, iters_per_epoch, schedule_fn)


def step_lr(optimizer, base_lr, warmup_epochs, epochs, iters_per_epoch, gamma, step_size):
    return lambda step: assign_learning_rate(optimizer, base_lr * gamma ** ((step // iters_per_epoch) // step_size))


def constant_lr(optimizer, base_lr, warmup_epochs, epochs, iters_per_epoch, gamma, step_size):
    return lambda step: assign_learning_rate(optimizer, base_lr)