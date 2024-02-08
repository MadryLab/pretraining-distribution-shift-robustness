import torch as ch
import torchvision
from pathlib import Path
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from src.datasets.imagenet_ood_utils import ImageNetRDataset, ImageNetSketchDataset, ImageNetV2Dataset
from src import paths


IMAGE_SIZE = 256


datasets = {
    "source_train": torchvision.datasets.ImageFolder(Path(paths.IMAGENET_PATH) / "train"),
    "source_val": torchvision.datasets.ImageFolder(Path(paths.IMAGENET_PATH) / "val"),
    "v2": ImageNetV2Dataset(),
    "sketch": ImageNetSketchDataset(),
    "r": ImageNetRDataset(),
}

class TransformedDataset(ch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.dataset)

class AggregateDataset(ch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        
    def __getitem__(self, index):
        cur_dataset_index = 0
        while index >= len(self.datasets[cur_dataset_index]):
            index -= len(self.datasets[cur_dataset_index])
            cur_dataset_index += 1
        return self.datasets[cur_dataset_index][index]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE),
        torchvision.transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    ])

    dataset = TransformedDataset(
        AggregateDataset(list(datasets.values())),
        transform,
    )

    write_path = Path(paths.IMAGENET_WITH_OOD_FFCV_PATH)
    write_path.parent.mkdir(exist_ok=True, parents=True)

    writer = DatasetWriter(write_path, {
        'image': RGBImageField(),
        'label': IntField(),
    })

    writer.from_indexed_dataset(dataset)
