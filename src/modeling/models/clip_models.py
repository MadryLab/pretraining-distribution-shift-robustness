import torch as ch
import open_clip
from copy import deepcopy

from .utils import ResizeWrapper


def get_zero_shot_weights(model, model_name, class_names, templates=None):
    if templates is None:
        templates = [lambda c: f"a photo of a {c}."]
    tokenizer = open_clip.get_tokenizer(model_name)
    with ch.no_grad():
        zero_shot_weights = []
        for class_name in class_names:
            if isinstance(class_name, list):
                texts = [template(sub_class_name) for template in templates for sub_class_name in class_name]
            else:
                assert isinstance(class_name, str)
                texts = [template(class_name) for template in templates]
            texts = tokenizer(texts).cuda()
            embeddings = model.encode_text(texts)

            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zero_shot_weights.append(embeddings)

    zero_shot_weights = ch.stack(zero_shot_weights, dim=0).cuda()
    zero_shot_weights = ch.transpose(zero_shot_weights, 0, 2)
    zero_shot_weights *= model.logit_scale.exp()

    zero_shot_weights = zero_shot_weights.squeeze().float()
    zero_shot_weights = ch.transpose(zero_shot_weights, 0, 1)

    return zero_shot_weights


class NormalizeFeatures(ch.nn.Module):
    def __init__(self, normalize):
        super().__init__()
        self.normalize = normalize

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return inputs


class ClassificationHead(ch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = ch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = ch.nn.Parameter(biases.clone())
        else:
            self.bias = ch.nn.Parameter(ch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class CLIPClassifier(ch.nn.Module):
    def __init__(self, model_name, pretrained, num_classes, class_names=None, templates=None):
        super().__init__()
        model, _, self.preprocessor = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.cuda()
        self.image_encoder = model.visual
        if class_names is None:
            if isinstance(self.image_encoder, open_clip.transformer.VisionTransformer):
                num_features, = self.image_encoder.ln_post.normalized_shape
                self.image_encoder.proj = None
            else:
                num_features = self.image_encoder.output_dim
            dummy = ch.nn.Linear(num_features, num_classes)
            self.classification_head = ClassificationHead(False, dummy.weight, dummy.bias)
        else:
            weights = get_zero_shot_weights(model, model_name, class_names, templates=templates)
            self.classification_head = ClassificationHead(True, weights)

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.classification_head(x)
        return x


def construct_clip_model(num_classes, model_name, pretrained, class_names=None, templates=None, features_only=False):
    prefix = "clip_"
    assert model_name[:len(prefix)] == prefix
    model_name = model_name[len(prefix):]
    model = CLIPClassifier(model_name, pretrained, num_classes, class_names=class_names, templates=templates)
    size = model.preprocessor.transforms[0].size
    if features_only:
        model = model.image_encoder
    model = ResizeWrapper(model, size)
    return model


def freeze_clip_features(model):
    for param in model.model.image_encoder.parameters():
        param.requires_grad = False


def get_clip_feature_extractor(model, copy=True):
    if copy:
        model = deepcopy(model)
    model.model.classification_head = NormalizeFeatures(model.model.classification_head.normalize)
    return model


def update_clip_classifier(model, weight, bias):
    model.model.classification_head.weight.data = weight
    model.model.classification_head.bias.data = bias
