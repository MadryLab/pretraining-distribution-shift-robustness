import numpy as np
from pathlib import Path

from ffcv.transforms import ReplaceLabel


IMAGENET_NORMALIZATION_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_NORMALIZATION_STD = np.array([0.229, 0.224, 0.225]) * 255


class FFCVDataset:
    def __init__(
        self,
        ffcv_path,
        num_classes,
        normalization_mean=IMAGENET_NORMALIZATION_MEAN,
        normalization_std=IMAGENET_NORMALIZATION_STD,
        labels=None,
        label_names=None,
        templates=None,
        custom_label_indices=None,
        transform=None,
        sample_indices=None,
    ):
        self._ffcv_path = Path(ffcv_path)
        if custom_label_indices is None:
            self.custom_labels = False
        else:
            num_classes = len(custom_label_indices)
            size = np.concatenate(custom_label_indices).max() + 1
            labels = np.full(size, -1)
            for label, indices in enumerate(custom_label_indices):
                labels[indices] = label
            self.custom_labels = True
        self._num_classes = num_classes
        self._normalization_mean = normalization_mean
        self._normalization_std = normalization_std
        self._labels = labels
        self.label_names = label_names
        self.templates = templates
        self.transform = transform
        self.sample_indices = sample_indices

    @property
    def _remap_label(self):
        return []

    @property
    def remap_label(self):
        if self.custom_labels:
            operations = []
            if self._labels is not None:
                for label in np.unique(self.labels):
                    operations.append(ReplaceLabel(np.where(self.labels==label)[0], label))
        else:
            operations = self._remap_label
        return operations

    @property
    def ffcv_path(self):
        return self._ffcv_path

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def normalization_mean(self):
        return self._normalization_mean

    @property
    def normalization_std(self):
        return self._normalization_std

    @property
    def labels(self):
        if self._labels is None:
            raise NotImplementedError
        else:
            return self._labels

    def __len__(self):
        return len(self.labels)