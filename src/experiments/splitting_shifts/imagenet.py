import sys
from copy import deepcopy
import numpy as np
from collections import defaultdict

from src.experiments.utils import generate_configs
from src.experiment_manager import ExperimentManager
from src import paths, modeling
from src.experiments.splitting_shifts.density_ratio_estimation import DensityRatioEstimationManager
import src.experiment_manager.model_manager as model_manager
from src.datasets.imagenet_ood_utils import ImageNetRDataset, ImageNetSketchDataset, ImageNetV2Dataset, ImageNetWithOODFFCVDataset


EPSILONS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1)


SUPPORT_ESTIMATION_DEFAULT_CONFIG = {
    "training": {
        "optimizer": "adamw",
        "lr": 3e-5,
        "lr_schedule": "cosine",
        "warmup_epochs": 1,
        "epochs": 8,
        "batch_size": 64,
        "weight_decay": 0.1,
        "label_smoothing": 0.0,
        "use_scaler": True,
        "clip_grad": True,
        "grad_clip_norm": 1.0,
        "image_dtype": "float16",
        "decoder": "simple",
        "augmentation": "flip",
        "num_workers": 10,
    },
    "evaluation": {
        "lr_tta": False,
    },
    "model": {
        "model_name": "clip_ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "resize": -1,
    }
}


class SplittingImagenetShiftsExperimentManager(ExperimentManager):
    def __init__(self, path, estimate_support=False):
        self.datasets = {
            "v2": ImageNetV2Dataset(),
            "sketch": ImageNetSketchDataset(),
            "r": ImageNetRDataset(),
        }
        self.ffcv_dataset = ImageNetWithOODFFCVDataset(paths.IMAGENET_WITH_OOD_FFCV_PATH)
        self.sub_indices = defaultdict(dict)
        self.indices = self.compute_indices()
        super().__init__(path)
        source_val_labels = self.get_labels("source_val").cpu().numpy()
        r_subset_mask = np.isin(source_val_labels, np.where(self.datasets["r"].class_subset_mask)[0])
        self.sub_indices["source_val"]["r_subset"] = np.where(r_subset_mask)[0]
        self.density_ratio_estimation_managers = {}
        for target_split in self.target_splits:
            source_indices = self.get_support_estimation_source_indices(target_split)
            target_indices = self.get_indices(target_split)
            custom_label_indices = (source_indices, target_indices)
            support_estimation_ffcv_dataset = ImageNetWithOODFFCVDataset(
                paths.IMAGENET_WITH_OOD_FFCV_PATH, custom_label_indices=custom_label_indices
            )
            self.density_ratio_estimation_managers[target_split] = DensityRatioEstimationManager(
                path / "density_ratio_estimation" / target_split,
                support_estimation_ffcv_dataset,
                source_indices,
                target_indices,
                self.support_estimation_configs,
                self.support_estimation_linear_probe_config,
            )
        if estimate_support:
            for target_split in self.target_splits:
                if not self.density_ratio_estimation_managers[target_split].computed:
                    continue
                density_ratios = self.density_ratio_estimation_managers[target_split].get_density_ratios()["target_density_ratios"]
                target_indices = self.get_indices(target_split)
                for epsilon in EPSILONS:
                    target_support_indices = np.where(density_ratios >= 1 / epsilon)[0]
                    self.sub_indices[target_split][f"target_support_epsilon={epsilon}"] = target_support_indices
                    self.sub_indices[target_split][f"shared_support_epsilon={epsilon}"] = np.setdiff1d(
                        np.arange(len(target_indices)),
                        target_support_indices,
                    )

    def get_support_estimation_source_indices(self, target_split):
        source_indices = self.get_indices("source_train")
        source_labels = self.get_labels("source_train").cpu().numpy()
        if target_split == "r_val":
            mask = np.isin(source_labels, np.where(self.datasets["r"].class_subset_mask)[0])
            return np.random.RandomState(0).choice(source_indices[mask], size=100_000, replace=False)
        else:
            return np.random.RandomState(0).choice(source_indices, size=100_000, replace=False)

    def compute_indices(self):
        splits = ["source_train", "source_val"] + self.target_splits
        return {split: self.ffcv_dataset.get_indices(split) for split in splits}

    @property
    def support_estimation_source_split(self):
        return "source_val"

    @property
    def target_splits(self):
        return ["sketch_val", "r_val", "v2_val"]

    @property
    def support_estimation_configs(self):
        param_options = {
            ("training", "lr"): [2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6],
        }
        return generate_configs(param_options, SUPPORT_ESTIMATION_DEFAULT_CONFIG)

    @property
    def support_estimation_linear_probe_config(self):
        config = deepcopy(SUPPORT_ESTIMATION_DEFAULT_CONFIG)
        config["training"].update({
            "freeze_features": True,
            "weight_decay": 0.0,
            "lr": 0.001,
        })
        return config

    def get_ffcv_dataset(self, split):
        return self.ffcv_dataset

    def get_indices(self, split):
        return self.indices[split]

    def get_sub_indices(self, split, sub_split):
        return self.sub_indices[split][sub_split]

    def get_loader(self, split):
        return modeling.make_loader(
            self.get_ffcv_dataset(split),
            indices=self.get_indices(split),
            train=False,
            batch_size=64,
            decoder="simple",
            normalization_params=(0.0, 1.0),
            num_workers=10,
            image_dtype="float16",
        )

    def _postprocess_preds(self, preds, split_name, sub_split_name):
        if split_name == "source_val" and sub_split_name == "r_subset":
            preds[:, ~self.datasets["r"].class_subset_mask] = -1_000
        return preds

    def _make_baseline_model_managers(self, train_ffcv_dataset, train_indices):
        timm_model_names = [
            ('resnet18', 'a1_in1k'),
            ('resnet34', 'a1_in1k'),
            ('resnet50', 'a1_in1k'),
            ('resnet101', 'a1h_in1k'),
            ('resnet152', 'a1h_in1k'),
            ('resnetv2_50', 'a1h_in1k'),
            ('resnetv2_101', 'a1h_in1k'),
            ('legacy_seresnet18', 'in1k'),
            ('legacy_seresnet34', 'in1k'),
            ('legacy_seresnet50', 'in1k'),
            ('legacy_seresnet101', 'in1k'),
            ('legacy_seresnet152', 'in1k'),
            ('resnext50_32x4d', 'a1h_in1k'),
            ('resnext101_32x4d', 'gluon_in1k'),
            ('resnext101_32x8d', 'tv_in1k'),
            ('wide_resnet50_2', 'racm_in1k'),
            ('wide_resnet101_2', 'tv_in1k'),
            ('convit_tiny', 'fb_in1k'),
            ('convit_small', 'fb_in1k'),
            ('convit_base', 'fb_in1k'),
            ('convnext_tiny', 'fb_in1k'),
            ('convnext_small', 'fb_in1k'),
            ('convnext_base', 'fb_in1k'),
            ('convnext_large', 'fb_in1k'),
            ('convnextv2_atto', 'fcmae_ft_in1k'),
            ('convnextv2_femto', 'fcmae_ft_in1k'),
            ('convnextv2_pico', 'fcmae_ft_in1k'),
            ('convnextv2_nano', 'fcmae_ft_in1k'),
            ('convnextv2_tiny', 'fcmae_ft_in1k'),
            ('convnextv2_base', 'fcmae_ft_in1k'),
            ('convnextv2_large', 'fcmae_ft_in1k'),
            ('convnextv2_huge', 'fcmae_ft_in1k'),
            ('densenet121', 'ra_in1k'),
            ('densenet161', 'tv_in1k'),
            ('densenet169', 'tv_in1k'),
            ('densenet201', 'tv_in1k'),
            ('mobilenetv3_small_100', 'lamb_in1k'),
            ('mobilenetv3_large_100', 'ra_in1k'),
            ('mobilevitv2_050', 'cvnets_in1k'),
            ('mobilevitv2_075', 'cvnets_in1k'),
            ('mobilevitv2_100', 'cvnets_in1k'),
            ('mobilevitv2_125', 'cvnets_in1k'),
            ('mobilevitv2_150', 'cvnets_in1k'),
            ('mobilevitv2_175', 'cvnets_in1k'),
            ('mobilevitv2_200', 'cvnets_in1k'),
            ('spnasnet_100', 'rmsp_in1k'),
            ('swinv2_tiny_window8_256', 'ms_in1k'),
            ('swinv2_tiny_window16_256', 'ms_in1k'),
            ('swinv2_small_window8_256', 'ms_in1k'),
            ('swinv2_small_window16_256', 'ms_in1k'),
            ('swinv2_base_window8_256', 'ms_in1k'),
            ('swinv2_base_window16_256', 'ms_in1k'),
            ('efficientnet_b0', 'ra_in1k'),
            ('efficientnet_b1', 'ft_in1k'),
            ('efficientnet_b2', 'ra_in1k'),
            ('efficientnet_b3', 'ra2_in1k'),
            ('efficientnet_b4', 'ra2_in1k'),
            ('efficientformerv2_l', 'snap_dist_in1k'),
            ('efficientformerv2_s0', 'snap_dist_in1k'),
            ('efficientformerv2_s1', 'snap_dist_in1k'),
            ('efficientformerv2_s2', 'snap_dist_in1k'),
            ('deit3_base_patch16_224', 'fb_in1k'),
            ('deit3_base_patch16_384', 'fb_in1k'),
            ('deit3_small_patch16_224', 'fb_in1k'),
            ('deit3_small_patch16_384', 'fb_in1k'),
            ('deit3_medium_patch16_224', 'fb_in1k'),
            ('deit3_large_patch16_224', 'fb_in1k'),
            ('deit3_large_patch16_384', 'fb_in1k'),
            ('cait_xxs24_224', 'fb_dist_in1k'),
            ('cait_xxs24_384', 'fb_dist_in1k'),
            ('cait_xxs36_224', 'fb_dist_in1k'),
            ('cait_xxs36_384', 'fb_dist_in1k'),
            ('cait_xs24_384', 'fb_dist_in1k'),
            ('cait_s24_224', 'fb_dist_in1k'),
            ('cait_s24_384', 'fb_dist_in1k'),
            ('cait_s36_384', 'fb_dist_in1k'),
            ('cait_m36_384', 'fb_dist_in1k'),
            ('cait_m48_448', 'fb_dist_in1k'),
        ]
        model_managers = {}
        for model_name, pretrained_cfg in timm_model_names:
            config = {
                "model": {
                    "model_name": f"timm_{model_name}",
                    "pretrained": pretrained_cfg,
                    "resize": -1,
                }
            }
            model_managers[f"baseline_{model_name}"] = model_manager.TimmModelManager(config, group="Baseline")
        return model_managers

    def _make_pretrained_model_managers(self, train_ffcv_dataset, train_indices):
        timm_model_names = [
            ('resnet18', 'fb_swsl_ig1b_ft_in1k'),
            ('resnet50', 'fb_swsl_ig1b_ft_in1k'),
            ('resnext50_32x4d', 'fb_swsl_ig1b_ft_in1k'),
            ('resnext101_32x4d', 'fb_swsl_ig1b_ft_in1k'),
            ('resnext101_32x8d', 'fb_swsl_ig1b_ft_in1k'),
            ('resnext101_32x8d', 'fb_wsl_ig1b_ft_in1k'),
            ('resnext101_32x16d', 'fb_swsl_ig1b_ft_in1k'),
            ('resnext101_32x16d', 'fb_wsl_ig1b_ft_in1k'),
            ('resnext101_32x32d', 'fb_wsl_ig1b_ft_in1k'),
            ('beit_base_patch16_224', 'in22k_ft_in22k_in1k'),
            ('beit_base_patch16_384', 'in22k_ft_in22k_in1k'),
            ('beit_large_patch16_224', 'in22k_ft_in22k_in1k'),
            ('beit_large_patch16_384', 'in22k_ft_in22k_in1k'),
            ('deit3_base_patch16_224', 'fb_in22k_ft_in1k'),
            ('deit3_base_patch16_384', 'fb_in22k_ft_in1k'),
            ('deit3_small_patch16_224', 'fb_in22k_ft_in1k'),
            ('deit3_small_patch16_384', 'fb_in22k_ft_in1k'),
            ('deit3_medium_patch16_224', 'fb_in22k_ft_in1k'),
            ('deit3_large_patch16_224', 'fb_in22k_ft_in1k'),
            ('deit3_large_patch16_384', 'fb_in22k_ft_in1k'),
            ('convnext_tiny', 'fb_in22k_ft_in1k'),
            ('convnext_tiny', 'fb_in22k_ft_in1k_384'),
            ('convnext_small', 'fb_in22k_ft_in1k'),
            ('convnext_small', 'fb_in22k_ft_in1k_384'),
            ('convnext_base', 'fb_in22k_ft_in1k'),
            ('convnext_base', 'fb_in22k_ft_in1k_384'),
            ('convnext_base', 'clip_laion2b_augreg_ft_in1k'),
            ('convnext_base', 'clip_laion2b_augreg_ft_in12k_in1k'),
            ('convnext_base', 'clip_laion2b_augreg_ft_in12k_in1k_384'),
            ('convnext_base', 'clip_laiona_augreg_ft_in1k_384'),
            ('convnext_large', 'fb_in22k_ft_in1k'),
            ('convnext_base', 'clip_laion2b_augreg_ft_in12k_in1k'),
            ('convnext_large_mlp', 'clip_laion2b_augreg_ft_in1k'),
            ('convnext_large_mlp', 'clip_laion2b_augreg_ft_in1k_384'),
            ('convnext_large_mlp', 'clip_laion2b_soup_ft_in12k_in1k_320'),
            ('convnext_large_mlp', 'clip_laion2b_soup_ft_in12k_in1k_384'),
            ('convnext_xxlarge', 'clip_laion2b_soup_ft_in1k'),
            ('vit_base_patch32_clip_224', 'openai_ft_in1k'),
            ('vit_base_patch32_clip_224', 'laion2b_ft_in12k_in1k'),
            ('vit_base_patch32_clip_384', 'openai_ft_in12k_in1k'),
            ('vit_base_patch32_clip_384', 'laion2b_ft_in12k_in1k'),
            ('vit_base_patch32_clip_448', 'laion2b_ft_in12k_in1k'),
            ('vit_base_patch16_clip_224', 'openai_ft_in1k'),
            ('vit_base_patch16_clip_224', 'laion2b_ft_in12k_in1k'),
            ('vit_base_patch16_clip_384', 'openai_ft_in12k_in1k'),
            ('vit_base_patch16_clip_384', 'laion2b_ft_in12k_in1k'),
            ('vit_large_patch14_clip_224', 'openai_ft_in1k'),
            ('vit_large_patch14_clip_224', 'openai_ft_in12k_in1k'),
            ('vit_large_patch14_clip_224', 'laion2b_ft_in1k'),
            ('vit_large_patch14_clip_224', 'laion2b_ft_in12k_in1k'),
            ('vit_large_patch14_clip_336', 'openai_ft_in12k_in1k'),
            ('vit_large_patch14_clip_336', 'laion2b_ft_in1k'),
            ('vit_large_patch14_clip_336', 'laion2b_ft_in12k_in1k'),
            ('vit_huge_patch14_clip_224', 'laion2b_ft_in1k'),
            ('vit_huge_patch14_clip_224', 'laion2b_ft_in12k_in1k'),
            ('vit_huge_patch14_clip_336', 'laion2b_ft_in12k_in1k'),
        ]
        model_managers = {}
        for model_name, pretrained_cfg in timm_model_names:
            config = {
                "model": {
                    "model_name": f"timm_{model_name}",
                    "pretrained": pretrained_cfg,
                    "resize": -1,
                }
            }
            model_managers[f"pretrained_{model_name}_{pretrained_cfg}"] = model_manager.TimmModelManager(config, group="Pre-trained")
        return model_managers

    def _make_model_managers(self):
        train_ffcv_dataset = self.get_ffcv_dataset("source_train")
        train_indices = self.get_indices("source_train")
        baseline_model_managers = self._make_baseline_model_managers(train_ffcv_dataset, train_indices)
        pretrained_model_managers = self._make_pretrained_model_managers(train_ffcv_dataset, train_indices)
        return {**baseline_model_managers, **pretrained_model_managers}


if __name__ == "__main__":
    experiment_type = sys.argv[1]
    index = int(sys.argv[2])

    manager = SplittingImagenetShiftsExperimentManager(paths.STORE_PATH / "splitting_shifts", estimate_support=False)

    if "estimate_support" in experiment_type:
        target_split = sys.argv[3]
        density_ratio_estimation_manager = manager.density_ratio_estimation_managers[target_split]
        if "tune" in experiment_type:
            density_ratio_estimation_manager.compute_auc_for_config_index(index)
        else:
            density_ratio_estimation_manager.get_density_ratios(fold=index)
    else:
        assert experiment_type == "evaluate"
        model_name = manager.model_names[index]
        manager.get_preds(model_name, "source_val")
        for target_split in manager.target_splits:
            manager.get_preds(model_name, target_split)
