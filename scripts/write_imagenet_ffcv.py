import sys
import numpy as np
import torch as ch
import torchvision
from pathlib import Path
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from src import paths


IMAGE_SIZE = 256


class TransformedDataset(ch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    split = sys.argv[1]

    dataset = torchvision.datasets.ImageFolder(Path(paths.IMAGENET_PATH) / split)
    labels_path = Path(paths.IMAGENET_LABELS_BASE_PATH.format(split=split))
    labels_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(labels_path, np.array(dataset.targets))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE),
        torchvision.transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    ])

    # For the validation set, we just want center crops
    if split == "val":
        dataset = TransformedDataset(
            dataset,
            transform,
        )

    write_path = Path(paths.IMAGENET_FFCV_BASE_PATH.format(split=split))
    write_path.parent.mkdir(exist_ok=True, parents=True)

    writer = DatasetWriter(write_path, {
        'image': RGBImageField(max_resolution=IMAGE_SIZE),
        'label': IntField(),
    })

    writer.from_indexed_dataset(dataset)
