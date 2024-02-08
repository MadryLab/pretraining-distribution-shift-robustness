import torch as ch
from copy import deepcopy
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from .utils import ResizeWrapper


def construct_timm_model(num_classes, model_name="timm_convnext_tiny", pretrained=None):
    prefix = "timm_"
    assert model_name[:len(prefix)] == prefix
    model_name = model_name[len(prefix):]
    if pretrained is not None:
        pretrained_cfg = None if pretrained == "DEFAULT" else pretrained
        pretrained = True
    else:
        pretrained_cfg = None
        pretrained = False
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, pretrained_cfg=pretrained_cfg)
    model.transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model = model.cuda()
    return model


def freeze_timm_features(model):
    if isinstance(model, ResizeWrapper):
        model = model.model
    for param in model.parameters():
        param.requires_grad = False
    modified = False
    for linear_name in ["fc", "head", "classifier"]:
        if hasattr(model, linear_name):
            for param in getattr(model, linear_name).parameters():
                param.requires_grad = True
            modified = True
    assert modified


def get_timm_feature_extractor(model, copy=True):
    if copy:
        model = deepcopy(model)
    if isinstance(model, ResizeWrapper):
        model.model.fc = ch.nn.Identity()
    else:
        model.fc = ch.nn.Identity()
    return model


def update_timm_classifier(model, weight, bias):
    if isinstance(model, ResizeWrapper):
        model = model.model
    model.fc.weight.data = weight
    model.fc.bias.data = bias