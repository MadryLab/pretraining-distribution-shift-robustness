import numpy as np
import torch as ch
import torchvision

from src.paths import IMAGENET_OOD_PATHS
from src.dataset_utils import IMAGENET_NORMALIZATION_MEAN, IMAGENET_NORMALIZATION_STD, FFCVDataset
from src.datasets.imagenet_utils import IMAGENET_SYNSET_ID_TO_IDX, IMAGENET_COMMON_CLASS_NAMES, IMAGENET_TEMPLATES


class ImageNetOODDataset(ch.utils.data.Dataset):
    def __init__(self, image_folder_path):
        self.dataset = torchvision.datasets.ImageFolder(image_folder_path)

        label_to_imagenet_label = {}
        for i, class_id in enumerate(self.dataset.classes):
            # If the class ids are synset ids, use mapping. Otherwise, we assume class ids are imagenet ids
            label_to_imagenet_label[i] = IMAGENET_SYNSET_ID_TO_IDX.get(class_id, class_id)

        self.labels = np.array([label_to_imagenet_label[label] for label in self.dataset.targets], dtype=int)

    @property
    def class_subset_mask(self):
        mask = np.zeros(1_000, dtype=bool)
        mask[np.unique(self.labels)] = True
        return mask

    def postprocess_preds(self, preds):
        # Never predict classes not present in the dataset
        preds[:, ~self.class_subset_mask] = -1_000
        return preds

    def process_labels(self, labels):
        return labels

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.dataset)


class ImageNetV2Dataset(ImageNetOODDataset):
    def __init__(self):
        super().__init__(IMAGENET_OOD_PATHS["v2"])


class ImageNetRDataset(ImageNetOODDataset):
    def __init__(self):
        super().__init__(IMAGENET_OOD_PATHS["r"])


class ImageNetSketchDataset(ImageNetOODDataset):
    def __init__(self):
        super().__init__(IMAGENET_OOD_PATHS["sketch"])

    
class ImageNetWithOODFFCVDataset(FFCVDataset):
    def __init__(self, path, custom_label_indices=None):
        num_examples = {
            'source_train': 1281167,
            'source_val': 50000,
            'v2_val': 10000,
            'sketch_val': 50889,
            'r_val': 30000,
        }
        
        self.indices = {}
        total = 0
        for split in num_examples:
            self.indices[split] = np.arange(total, total + num_examples[split])
            total += num_examples[split]
            
        super().__init__(
            path,
            1_000,
            normalization_mean=IMAGENET_NORMALIZATION_MEAN,
            normalization_std=IMAGENET_NORMALIZATION_STD,
            label_names=IMAGENET_COMMON_CLASS_NAMES,
            templates=IMAGENET_TEMPLATES,
            custom_label_indices=custom_label_indices,
        )
        
    def get_indices(self, split):
        return self.indices[split]
    
    def __len__(self):
        return sum(len(indices) for indices in self.indices.values())
