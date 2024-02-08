import sys
import numpy as np
import torch as ch
from torchvision.transforms.functional import vflip
from copy import deepcopy
from collections import defaultdict
    
from src.experiments.utils import generate_configs, sample_indices
import src.experiment_manager.model_manager as model_manager
from src.experiment_manager import ExperimentManager
from src import modeling, dataset_utils, paths
from src.datasets.imagenet_utils import IMAGENET_TEMPLATES, IMAGENET_COMMON_CLASS_NAMES


BASELINE_CONFIG = {
    "training": {
        "optimizer": "adamw",
        "lr": 0.003,
        "lr_schedule": "cosine",
        "warmup_epochs": 10,
        "epochs": 100,
        "batch_size": 512,
        "weight_decay": 0.1,
        "label_smoothing": 0.1,
        "use_scaler": True,
        "clip_grad": True,
        "grad_clip_norm": 1.0,
        "image_dtype": "float16",
        "decoder": "imagenet_random_crop",
        "augmentation": "flip",
        "num_workers": 10,
    },
    "evaluation": {
        "lr_tta": False,
    },
    "model": {
        "model_name": "clip_ViT-B-32",
        "pretrained": "None",
        "resize": -1,
    },
}

EPOCH_START = 51
EPOCH_END = 85
EPOCH_STEP = 1

PRETRAINED_BASE_CONFIG = {
    "training": {
        "freeze_features": False,
        "optimizer": "adamw",
        "lr": 3e-5,
        "lr_schedule": "cosine",
        "warmup_epochs": 1,
        "epochs": 8,
        "batch_size": 512,
        "weight_decay": 0.1,
        "label_smoothing": 0.0,
        "use_scaler": True,
        "clip_grad": True,
        "grad_clip_norm": 1.0,
        "image_dtype": "float16",
        "decoder": "imagenet_random_crop",
        "augmentation": "flip",
        "num_workers": 10,
    },
    "evaluation": {
        "lr_tta": False,
    },
}

PRETRAINED_CLIP_CONFIG = deepcopy(PRETRAINED_BASE_CONFIG)
PRETRAINED_CLIP_CONFIG["model"] = {
    "model_name": "clip_ViT-B-32",
    "pretrained": "openai",
    "resize": -1,
}

PRETRAINED_IN21K_CONFIG = deepcopy(PRETRAINED_BASE_CONFIG)
PRETRAINED_IN21K_CONFIG["model"] = {
    "model_name": "timm_vit_base_patch32_224",
    "pretrained": "augreg_in21k",
    "resize": -1,
}

PRETRAINED_CONFIGS_AND_MANAGERS = {
    "openai_clip": (PRETRAINED_CLIP_CONFIG, model_manager.CLIPFinetunedModelManager, "CLIP"),
    "augreg_in21k": (PRETRAINED_IN21K_CONFIG, model_manager.TimmFinetunedModelManager, "AugReg"),
}

PRETRAINED_PARAM_OPTIONS = {
    ("training", "lr"): [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
}

LINEAR_PROBE_BASE_CONFIG = {
    "training": {
        "freeze_features": True,
        "optimizer": "adamw",
        "lr": 0.001,
        "lr_schedule": "cosine",
        "warmup_epochs": 0,
        "epochs": 4,
        "batch_size": 512,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "use_scaler": True,
        "clip_grad": False,
        "grad_clip_norm": 1.0,
        "image_dtype": "float16",
        "decoder": "imagenet_random_crop",
        "augmentation": "flip",
        "num_workers": 10,
    },
    "evaluation": {
        "lr_tta": False,
    },
}


INIT_STRATEGIES = {
    "random": (False, None, "FT"),
    "linear_probe": (False, LINEAR_PROBE_BASE_CONFIG, "LP-FT"),
    "zero_shot": (True, None, "ZS-FT"),
}


HPARAM_EXPERIMENT_OPTIONS = {
    "epochs": {
        (("training", "epochs"), ("training", "warmup_epochs")): [(1, 0.125), (2, 0.25), (4, 0.5), (8, 1), (16, 2), (32, 4)],
    },
    "lr": {
        ("training", "lr"): [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
    },
    "batch_size": {
        ("training", "batch_size"): [64, 128, 256, 512],
    },
    "weight_decay": {
        ("training", "weight_decay"): [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0],
    },
}


HPARAM_EXPERIMENT_MODEL_NAME = "openai_clip_zero_shot_head"


class ImageNetSyntheticShiftExperimentManager(ExperimentManager):
    def __init__(
        self,
        path,
        source_transform=None,
        source_class_sampling_rates=None,
        target_transforms={},
        target_class_sampling_rates={},
        imagenet_ffcv_base_path=paths.IMAGENET_FFCV_BASE_PATH,
        imagenet_labels_base_path=paths.IMAGENET_LABELS_BASE_PATH,
        num_tuning_copies=1,
        num_copies=4,
        adjust_config=None,
        epoch_start=EPOCH_START,
        epoch_end=EPOCH_END,
        epoch_step=EPOCH_STEP,
    ):
        self.ffcv_datasets = {}
        for split in ("train", "val"):
            self.ffcv_datasets[split] = dataset_utils.FFCVDataset(
                imagenet_ffcv_base_path.format(split=split),
                1_000,
                label_names=IMAGENET_COMMON_CLASS_NAMES,
                templates=IMAGENET_TEMPLATES,
            )
        labels = {}
        self.indices = defaultdict(lambda: None)
        for split in ("train", "val"):
            labels[split] = np.load(imagenet_labels_base_path.format(split=split))
        for split in ("train", "val"):
            self.ffcv_datasets[f"source_{split}"] = deepcopy(self.ffcv_datasets[split])
            self.ffcv_datasets[f"source_{split}"].transform = source_transform
            if source_class_sampling_rates is not None:
                self.indices[f"source_{split}"] = sample_indices(labels[split], class_sampling_rates=source_class_sampling_rates)
        self.target_names = set.union(set(target_transforms), set(target_class_sampling_rates))
        for target_name in self.target_names:
            self.ffcv_datasets[f"{target_name}_val"] = deepcopy(self.ffcv_datasets["val"])
            self.ffcv_datasets[f"{target_name}_val"].transform = target_transforms.get(target_name, None)
            if target_name in target_class_sampling_rates:
                self.indices[f"{target_name}_val"] = sample_indices(labels[split], class_sampling_rates=target_class_sampling_rates[target_name])
        self.num_tuning_copies = num_tuning_copies
        self.num_copies = num_copies
        self.adjust_config = (lambda config: config) if adjust_config is None else adjust_config
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.epoch_step = epoch_step
        super().__init__(path)

    def get_ffcv_dataset(self, split):
        return self.ffcv_datasets[split]

    def get_indices(self, split):
        return self.indices[split]

    def get_loader(self, split):
        modeling.populate_config(BASELINE_CONFIG)
        return modeling.make_loader(
            self.get_ffcv_dataset(split),
            indices=self.get_indices(split),
            train="train" in split,
            batch_size=512,
            normalization_params=(0.0, 1.0),
        )

    def _get_tuning_specs(self):
        tuning_specs = []
        train_ffcv_dataset = self.get_ffcv_dataset("source_train")
        train_indices = self.get_indices("source_train")

        for pretrained_name, (config, manager_cls, pretrained_abbr) in PRETRAINED_CONFIGS_AND_MANAGERS.items():
            for init_name, (zero_shot_init, linear_probe_base_config, init_abbr) in INIT_STRATEGIES.items():
                configs = generate_configs(PRETRAINED_PARAM_OPTIONS, config)
                if linear_probe_base_config is None:
                    linear_probe_config = None
                else:
                    linear_probe_config = deepcopy(linear_probe_base_config)
                    linear_probe_config["model"] = config["model"]
                tuning_kwargs = {
                    "ffcv_dataset": train_ffcv_dataset,
                    "indices": train_indices,
                    "num_copies": self.num_tuning_copies,
                    "linear_probe_config": linear_probe_config,
                    "group": f"{pretrained_abbr} ({init_abbr})",
                    "checkpoint_every": 1,
                    "verbose_epochs": True,
                }
                if manager_cls == model_manager.CLIPFinetunedModelManager:
                    tuning_kwargs["zero_shot_init"] = zero_shot_init
                # Zero-shot init is only relevant for CLIP
                elif zero_shot_init:
                    continue
                tuning_specs.append((
                    f"{pretrained_name}_{init_name}_head_tuning",
                    manager_cls,
                    tuning_kwargs,
                    configs,
                ))

        return tuning_specs

    def _get_baseline_model_name(self, config):
        lr = config["training"]["lr"]
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]
        weight_decay = config["training"]["weight_decay"]
        return f"baseline_lr={lr}_epochs={epochs}_batch_size={batch_size}_weight_decay={weight_decay}"

    def _make_model_managers(self):
        model_managers = {}
        train_ffcv_dataset = self.get_ffcv_dataset("source_train")
        train_indices = self.get_indices("source_train")

        for pretrained_name in PRETRAINED_CONFIGS_AND_MANAGERS:
            for init_name in INIT_STRATEGIES:
                model_name = f"{pretrained_name}_{init_name}_head"
                if f"{model_name}_tuning" in self.tuning_model_managers:
                    manager_cls, kwargs = self._get_selected_specs(f"{model_name}_tuning")
                    kwargs["num_copies"] = self.num_copies
                    model_managers[model_name] = manager_cls(**kwargs)

        model_managers["baseline"] = model_manager.SimpleModelManager(
            train_ffcv_dataset,
            train_indices,
            self.adjust_config(BASELINE_CONFIG),
            group="Baseline (fully-trained model)",
            save_every=1,
            checkpoint_every=1,
            verbose_epochs=True,
        )

        for epoch in range(self.epoch_start, self.epoch_end + 1, self.epoch_step):
            model_name = f"baseline_epoch={epoch}"
            model_managers[model_name] = model_manager.IntermediateEpochModelManager(
                model_managers["baseline"],
                epoch=epoch,
                group="Baseline",
            )
            # Needed so that these managers access the saved baseline model
            self._custom_model_paths[(model_name, None)] = self.get_model_path("baseline", index=None)

        manager_cls, kwargs = self._get_selected_specs(f"{HPARAM_EXPERIMENT_MODEL_NAME}_tuning")
        if kwargs["config"] is not None:
            for hparam_name, options in HPARAM_EXPERIMENT_OPTIONS.items():
                configs = generate_configs(options, kwargs["config"])
                for config in configs:
                    kwargs_copy = kwargs.copy()
                    kwargs_copy["config"] = config
                    kwargs_copy["num_copies"] = self.num_copies
                    model_name = f"{HPARAM_EXPERIMENT_MODEL_NAME}_{hparam_name}={config['training'][hparam_name]}"
                    model_managers[model_name] = manager_cls(**kwargs_copy)
            model_managers[f"{HPARAM_EXPERIMENT_MODEL_NAME}_zero_shot"] = model_manager.CLIPZeroShotModelManager(
                train_ffcv_dataset,
                kwargs["config"],
                group="CLIP zero-shot",
            )

        return model_managers


class VerticalFlipTransform(ch.nn.Module):
    def forward(self, x, y, indices):
        x = vflip(x)
        return x, y


class TintTransform(ch.nn.Module):
    def __init__(self, alpha=0.25, spurious_p=0.8, num_classes=1_000, num_examples=1_281_167):
        super().__init__()
        self.alpha = alpha
        random = np.random.RandomState(0)
        self.colors = ch.tensor(random.uniform(size=(num_classes, 3)), dtype=ch.float16, device="cuda:0")
        self.random_y = ch.tensor(random.choice(num_classes, size=(num_examples,)), device="cuda:0")
        self.spurious_mask = ch.tensor(random.uniform(size=(num_examples,)) < spurious_p, device="cuda:0")
        
    def forward(self, x, y, indices):
        random_y = self.random_y[indices]
        spurious_mask = self.spurious_mask[indices]
        selected_y = y * spurious_mask + random_y * (~spurious_mask)
        selected_colors = self.colors[selected_y]
        x = x * (1 - self.alpha) + selected_colors[:, :, None, None] * self.alpha
        return x, y

def get_probs(labels, label_p, num_classes=1_000):
    preference = np.zeros(shape=(num_classes,), dtype=bool)
    preference[:num_classes // 2] = 1
    np.random.RandomState(0).shuffle(preference)
    return preference[labels] * label_p + (1 - preference[labels]) * (1 - label_p)

def get_class_probs(label_p, num_classes=1_000):
    preference = np.zeros(shape=(num_classes,), dtype=bool)
    preference[:num_classes // 2] = 1
    np.random.RandomState(0).shuffle(preference)
    if label_p <= .5:
        return preference * label_p / (1 - label_p) + (1 - preference)
    else:
        return preference + (1 - preference) * (1 - label_p) / label_p


managers = {}

managers["out_of_support"] = ImageNetSyntheticShiftExperimentManager(
    paths.STORE_PATH / "imagenet_out_of_support_shift",
    target_transforms={
        "tint_alpha=0.25": TintTransform(alpha=0.25, spurious_p=0.0),
        "flip": VerticalFlipTransform(),
    },
)
for spurious_p in [0.5, 0.6, 0.7, 0.8, 0.9]:
    managers[f"tint_p={spurious_p}"] = ImageNetSyntheticShiftExperimentManager(
        paths.STORE_PATH / f"imagenet_tint_p={spurious_p}_shift",
        source_transform=TintTransform(spurious_p=spurious_p),
        target_transforms={
            "target": TintTransform(spurious_p=0.0),
        },
    )

def double_epochs(config):
    config_copy = deepcopy(config)
    config_copy["training"]["epochs"] *= 2
    config_copy["training"]["warmup_epochs"] *= 2
    return config_copy

for label_p in [0.1, 0.15, 0.2, 0.25, 0.3]:
    managers[f"label_p={label_p}"] = ImageNetSyntheticShiftExperimentManager(
        paths.STORE_PATH / f"imagenet_label_p={label_p}_shift",
        source_class_sampling_rates=get_class_probs(label_p),
        target_class_sampling_rates={
            "target": get_class_probs(1-label_p),
        },
        adjust_config=double_epochs,
        epoch_start=EPOCH_START * 2,
        epoch_end=EPOCH_END * 2,
        epoch_step=EPOCH_STEP * 2,
    )


if __name__ == "__main__":
    manager_name = sys.argv[1]
    manager = managers[manager_name]
    model_index = int(sys.argv[2])
    model_name, index = manager.model_names_with_copies[model_index]
    print(manager.path)
    print(model_name, index)
    manager.get_preds(model_name, split_name="source_val", index=index)
    for target_name in manager.target_names:
        manager.get_preds(model_name, split_name=f"{target_name}_val", index=index)
