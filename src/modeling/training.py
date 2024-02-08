import os
import json
import torch as ch
from torch.cuda.amp import autocast, GradScaler
from fastargs.decorators import param

import src.modeling.schedulers as schedulers
import src.modeling.models as models
from src.utils import tqdm


def get_trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def get_optimizer(parameters, optimizer_name, lr, momentum, weight_decay):
    if optimizer_name == "sgd":
        return ch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return ch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return ch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer \"{optimizer_name}\"")


class AverageAccumulator:
    def __init__(self):
        self.total_value = 0.0
        self.total_examples = 0

    def register(self, value, examples):
        self.total_value += value * examples
        self.total_examples += examples

    @property
    def value(self):
        return self.total_value / self.total_examples


def pretty_print_metrics(metrics):
    metrics_strs = []
    for key in metrics:
        metrics_strs.append(f"{key}: {metrics[key]:.4g}")
    print(", ".join(metrics_strs))


def get_loss_fn(label_smoothing=0.0):
    label_criterion = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    def loss_fn(output, batch):
        _, y = batch
        return label_criterion(output, y)
    return loss_fn


@param('training.lr')
@param('training.lr_schedule')
@param('training.step_size')
@param('training.gamma')
@param('training.warmup_epochs')
@param('training.epochs')
@param('training.optimizer')
@param('training.weight_decay')
@param('training.momentum')
@param('training.label_smoothing')
@param('training.use_scaler')
@param('training.clip_grad')
@param('training.grad_clip_norm')
@param('training.freeze_features')
def train(
    model,
    train_loader,
    lr=None,
    lr_schedule=None,
    step_size=None,
    gamma=None,
    warmup_epochs=None,
    epochs=None,
    optimizer=None,
    weight_decay=None,
    momentum=None,
    label_smoothing=None,
    use_scaler=None,
    clip_grad=None,
    grad_clip_norm=None,
    freeze_features=None,
    val_loaders=None,
    save_path=None,
    verbose_epochs=False,
    eval_every=1,
    save_every=None,
    checkpoint_every=None,
    overwrite=False,
):
    model.cuda()
    if freeze_features:
        models.freeze_features(model)
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    if val_loaders is not None and not isinstance(val_loaders, dict):
        val_loaders = {"val": val_loaders}
    loss_fn = get_loss_fn(label_smoothing=label_smoothing)
    parameters = get_trainable_parameters(model)
    optimizer = get_optimizer(parameters, optimizer, lr, momentum, weight_decay)
    scheduler = schedulers.__dict__[f"{lr_schedule}_lr"](optimizer, lr, warmup_epochs, epochs, len(train_loader), gamma, step_size)
    if use_scaler:
        scaler = GradScaler()

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    if save_path is not None and (save_path / "final.pt").exists() and not overwrite:
        state_dict = ch.load(save_path / "final.pt")
        model.load_state_dict(state_dict)
        with open(save_path / "logs.json", "r") as f:
            metrics = json.load(f)
        return metrics
    if save_path is not None and (save_path / "checkpoint.pt").exists() and not overwrite:
        checkpoint = ch.load(save_path / "checkpoint.pt")
        epoch_iterator = tqdm(range(checkpoint["epoch"] + 1, epochs), desc='Epoch')
        scheduler_step = checkpoint["scheduler_step"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        metrics = checkpoint["metrics"]
    else:
        epoch_iterator = tqdm(range(epochs), desc='Epoch')
        scheduler_step = 0
        metrics = []
    for epoch in epoch_iterator:
        model.train()
        loader_iter = tqdm(train_loader, total=len(train_loader), disable=not verbose_epochs)
        train_loss = AverageAccumulator()
        for batch in loader_iter:
            scheduler(scheduler_step)
            scheduler_step += 1
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                x = batch[0]
                output = model(x)
                loss = loss_fn(output, batch)

            if use_scaler:
                scaler.scale(loss).backward()

                if clip_grad:
                    scaler.unscale_(optimizer)
                    ch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad:
                    ch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2)
                optimizer.step()

            msg = f'loss: {loss.item():.4g}'
            loader_iter.set_postfix_str(msg)
            train_loss.register(loss.item(), len(x))

        epoch_metrics = {"train_loss": train_loss.value}
        if val_loaders is not None and (epoch + 1) % eval_every == 0:
            for key, val_loader in val_loaders.items():
                model.eval()
                val_loss = AverageAccumulator()
                val_accuracy = AverageAccumulator()
                for batch in tqdm(val_loader, disable=not verbose_epochs):
                    x, y = batch[:2]
                    with autocast(), ch.no_grad():
                        output = model(x)
                        loss = loss_fn(output, batch)

                    val_loss.register(loss.item(), len(x))

                    accuracy = (output.argmax(dim=1) == y).sum() / len(x)
                    val_accuracy.register(accuracy.item(), len(x))
                epoch_metrics[f"{key}_loss"] = val_loss.value
                epoch_metrics[f"{key}_accuracy"] = val_accuracy.value
        if verbose_epochs:
            pretty_print_metrics(epoch_metrics)
        metrics.append(epoch_metrics)

        if save_path is not None and save_every is not None and (epoch + 1) % save_every == 0:
            models.save_model(save_path / f"epoch={epoch}.pt", model)

        if save_path is not None and checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_step": scheduler_step,
                "metrics": metrics,
            }
            ch.save(checkpoint, save_path / "checkpoint.pt")
    if save_path is not None:
        models.save_model(save_path / "final.pt", model)
        with open(save_path / "logs.json", "w") as f:
            json.dump(metrics, f)
    if checkpoint_every is not None and (save_path / "checkpoint.pt").exists():
        os.remove(save_path / "checkpoint.pt")
    return metrics