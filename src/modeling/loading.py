import numpy as np
import torch as ch
import torchvision

from fastargs.decorators import param

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate


def get_decoder_and_augmentations(
    decoder="simple",
    augmentation="",
    augment=True,
    normalization_mean=None,
):
    decoder_and_augmentations = []

    if decoder == "simple":
        decoder_and_augmentations.append(SimpleRGBImageDecoder())
    elif decoder == "center_crop":
        decoder_and_augmentations.append(CenterCropRGBImageDecoder((224, 224), 1))
    elif decoder == "random_crop":
        if augment:
            decoder_and_augmentations.append(RandomResizedCropRGBImageDecoder((224, 224), scale=(0.9, 1.0)))
        else:
            decoder_and_augmentations.append(CenterCropRGBImageDecoder((224, 224), 1))
    elif decoder == "imagenet_random_crop":
        if augment:
            decoder_and_augmentations.append(RandomResizedCropRGBImageDecoder((224, 224)))
        else:
            decoder_and_augmentations.append(CenterCropRGBImageDecoder((224, 224), 224 / 256))

    if augment:
        if "flip" in augmentation:
            decoder_and_augmentations.append(RandomHorizontalFlip())
        if "translate" in augmentation:
            decoder_and_augmentations.append(RandomTranslate(padding=2, fill=tuple(map(int, normalization_mean))))
        if "cutout" in augmentation:
            decoder_and_augmentations.append(Cutout(4, tuple(map(int, normalization_mean))))

    return decoder_and_augmentations

    
class ApplyWithoutNormalizationWrapper(ch.nn.Module):
    def __init__(self, transform, mean, std):
        super().__init__()
        self.transform = transform
        self.unnormalize = torchvision.transforms.Normalize(-mean / std, 1 / std)
        self.normalize = torchvision.transforms.Normalize(mean, std)

    def forward(self, x, y, indices):
        x = self.unnormalize(x)
        # torchvision transforms expect a range of [0, 1]
        x /= 255
        x, y = self.transform(x, y, indices)
        x *= 255
        x = self.normalize(x)
        return x, y

class SamplingLoader:
    def __init__(self, loader, sample_indices):
        self.loader = loader
        assert isinstance(loader, Loader)
        self.sample_indices = sample_indices
        num_examples = len(self.sample_indices(0))
        self.length = num_examples // self.loader.batch_size
        if (num_examples % self.loader.batch_size > 0) and not self.loader.drop_last:
            self.length += 1

    def __iter__(self):
        selected_indices = self.sample_indices(self.loader.next_epoch)
        self.loader.indices = selected_indices
        self.loader.traversal_order.indices = selected_indices
        for batch in self.loader:
            yield batch

    def __len__(self):
        return self.length


class TransformedLoader:
    def __init__(self, loader, transform, normalization_params):
        assert isinstance(loader, Loader)
        self.loader = loader
        self.transform = ApplyWithoutNormalizationWrapper(transform, *normalization_params)

    def __iter__(self):
        order = ch.tensor(self.loader.next_traversal_order().astype(int), device="cuda:0")
        current_count = 0
        for batch in self.loader:
            batch_count = len(batch[0])
            indices = order[current_count: current_count + batch_count]
            yield self.transform(*batch, indices=indices)
            current_count += batch_count

    def __len__(self):
        return len(self.loader)


@param("training.batch_size")
@param("training.decoder")
@param("training.augmentation")
@param("training.image_dtype")
@param("training.num_workers")
def make_loader(
    dataset,
    indices=None,
    train=False,
    batch_size=None,
    decoder=None,
    augmentation=None,
    image_dtype=None,
    num_workers=None,
    normalization_params=None,
    custom_augmentation=None,
):
    if normalization_params is None:
        normalization_mean = dataset.normalization_mean
        normalization_std = dataset.normalization_std
    else:
        normalization_mean, normalization_std = normalization_params

    label_pipeline = [
        IntDecoder(),
        *dataset.remap_label,
        ToTensor(),
        ToDevice(ch.device("cuda:0")),
        Squeeze(),
    ]
    image_pipeline = get_decoder_and_augmentations(
        decoder,
        augmentation,
        augment=train,
        normalization_mean=normalization_mean,
    )
    if custom_augmentation is None:
        custom_augmentation_pipeline = []
    else:
        custom_augmentation_pipeline = [custom_augmentation]
    image_pipeline.extend(
        [
            ToTensor(),
            ToDevice(ch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            Convert(ch.__dict__[image_dtype]),
            *custom_augmentation_pipeline,
            torchvision.transforms.Normalize(normalization_mean, normalization_std),
        ]
    )

    order = OrderOption.RANDOM if train else OrderOption.SEQUENTIAL

    loader = Loader(
        dataset.ffcv_path,
        indices=indices,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        drop_last=train,
        pipelines={
            "image": image_pipeline,
            "label": label_pipeline,
        },
    )
    if dataset.sample_indices is not None:
        loader = SamplingLoader(loader, dataset.sample_indices)
    if dataset.transform is not None:
        loader = TransformedLoader(loader, dataset.transform, (normalization_mean, normalization_std))
    return loader
