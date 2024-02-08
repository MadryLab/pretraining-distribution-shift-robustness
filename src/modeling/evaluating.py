from tqdm.notebook import tqdm

import torch as ch
from torch.cuda.amp import autocast
from torchvision.transforms.functional import hflip
from copy import deepcopy

from fastargs.decorators import param


@param("evaluation.lr_tta")
def evaluate(model, loader, lr_tta=False, verbose=True, autocast_dtype=None):
    model.eval()
    all_margins = []
    all_outputs = []
    all_labels = []
    for x, y in tqdm(loader, disable=not verbose):
        with autocast(dtype=autocast_dtype), ch.no_grad():
            output = model(x)
            if lr_tta:
                output += model(hflip(x))
                output /= 2

            all_outputs.append(output.cpu())
            class_logits = output[ch.arange(output.shape[0]), y].clone()
            output[ch.arange(output.shape[0]), y] = -1000
            next_classes = output.argmax(dim=1)
            class_logits -= output[ch.arange(output.shape[0]), next_classes]
            all_margins.append(class_logits.cpu())
            all_labels.append(deepcopy(y))
    outputs = ch.cat(all_outputs)
    margins = ch.cat(all_margins)
    labels = ch.cat(all_labels)
    accuracy = (margins > 0).sum().item() / len(margins)
    if verbose:
        print(f"Accuracy: {accuracy:.4g}")
    return {"margins": margins, "outputs": outputs, "labels": labels, "accuracy": accuracy}
