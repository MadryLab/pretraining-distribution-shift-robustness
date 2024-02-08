import torch as ch
from pathlib import Path

from .utils import ResizeWrapper
from .clip_models import construct_clip_model, freeze_clip_features, get_clip_feature_extractor, update_clip_classifier
from .timm_models import construct_timm_model, freeze_timm_features, get_timm_feature_extractor, update_timm_classifier

from fastargs.decorators import param


@param("model.model_name")
@param("model.pretrained")
@param("model.resize")
def construct_model(num_classes, model_name=None, pretrained=None, resize=None, features_only=False, **clip_kwargs):
    if pretrained == "None":
        pretrained = None
    if model_name[:len("timm")] == "timm":
        model = construct_timm_model(num_classes, model_name=model_name, pretrained=pretrained)
    elif model_name[:len("clip")] == "clip":
        model = construct_clip_model(num_classes, model_name, pretrained, features_only=features_only, **clip_kwargs)
    else:
        raise NotImplementedError
    model.cuda()
    if resize is not None and resize > 0:
        assert model_name[:len("clip")] != "clip", (model_name, resize) # CLIP models have their own resizing
        model = ResizeWrapper(model, resize)
    return model


@param("model.model_name")
def freeze_features(model, model_name=None):
    if "timm" in model_name:
        freeze_timm_features(model)
    elif "clip" in model_name:
        freeze_clip_features(model)
    else:
        raise NotImplementedError


@param("model.model_name")
def get_feature_extractor(model, model_name=None, copy=True):
    if "timm" in model_name:
        return get_timm_feature_extractor(model, copy=copy)
    elif "clip" in model_name:
        return get_clip_feature_extractor(model, copy=copy)
    else:
        raise NotImplementedError


@param("model.model_name")
def update_classifier(model, weight, bias, model_name=None):
    if "timm" in model_name:
        update_timm_classifier(model, weight, bias)
    elif "clip" in model_name:
        update_clip_classifier(model, weight, bias)
    else:
        raise NotImplementedError


def save_model(path, model):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    ch.save(model.state_dict(), path)


@param("model.model_name")
@param("model.pretrained")
@param("model.resize")
def load_model(path, num_classes, model_name=None, pretrained=None, resize=None, **clip_kwargs):
    model = construct_model(num_classes, model_name=model_name, pretrained=pretrained, resize=resize, **clip_kwargs)
    state_dict = ch.load(path)
    model.load_state_dict(state_dict)
    model.cuda()
    return model
