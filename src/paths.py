from pathlib import Path

PROJECT_PATH = Path("/mnt/xfs/projects/pretraining_distribution_shift_robustness")
STORE_PATH = PROJECT_PATH / "store"

IMAGENET_PATH = "/mnt/cfs/datasets/pytorch_imagenet"
IMAGENET_FFCV_BASE_PATH = "/mnt/xfs/projects/pretraining_distribution_shift_robustness/datasets/imagenet_betons/imagenet_{split}_256px.beton"
IMAGENET_LABELS_BASE_PATH = "/mnt/xfs/projects/pretraining_distribution_shift_robustness/datasets/imagenet_betons/{split}_labels.npy"

IMAGENET_OOD_PATHS = {
    "sketch": Path("/mnt/cfs/datasets/imagenet_ood/sketch"),
    "r": Path("/mnt/cfs/datasets/imagenet_ood/imagenet-r"),
    "v2": Path("/mnt/cfs/datasets/imagenetv2-matched-frequency-format-val"),
}
IMAGENET_WITH_OOD_FFCV_PATH = "/mnt/xfs/projects/pretraining_distribution_shift_robustness/datasets/imagenet_with_ood.beton"
