import numpy as np
import torch as ch
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
from filelock import FileLock
from collections import defaultdict

import src.modeling as modeling
import src.experiment_manager.model_manager as model_manager
from src.experiments.utils import generate_configs


def calibrate(logits, labels, lr=1.0, num_iters=1000):
    logits = ch.tensor(logits).cuda()
    labels = ch.tensor(labels).cuda()
    alpha = ch.nn.Parameter(data=ch.tensor(1.0))
    optimizer = ch.optim.SGD([alpha], lr=lr)
    losses = []
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = ch.nn.functional.binary_cross_entropy_with_logits(logits * alpha, labels)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    return alpha.item()


class DensityRatioEstimationManager:
    def __init__(self, save_path, ffcv_dataset, source_indices, target_indices, configs, linear_probe_config=None, num_folds=10, target_groups=None):
        self.save_path = Path(save_path)
        self.configs = configs
        self.linear_probe_config = linear_probe_config
        self.ffcv_dataset = ffcv_dataset
        self.source_indices = source_indices
        self.target_indices = target_indices
        self.num_folds = num_folds
        self.target_groups = target_groups

        self.aucs_path = self.save_path / "tuning" / "aucs.npy"
        self.aucs_path.parent.mkdir(exist_ok=True, parents=True)
        self.aucs_lock = FileLock(self.save_path / "tuning" / "aucs.npy.lock")

    def _get_results(self, config, model_path, train_indices, eval_indices, results_path=None):
        if results_path is not None and results_path.exists():
            return ch.load(results_path)
        else:
            if "clip" in config["model"]["model_name"]:
                manager = model_manager.CLIPFinetunedModelManager(
                    self.ffcv_dataset, train_indices, config, zero_shot_init=False, verbose_epochs=True, linear_probe_config=self.linear_probe_config, checkpoint_every=1
                )
            else:
                manager = model_manager.SimpleModelManager(self.ffcv_dataset, train_indices, config, checkpoint_every=1)
            model = manager.train_and_load(model_path)
            modeling.populate_config(config)
            loader = modeling.make_loader(
                self.ffcv_dataset,
                indices=eval_indices,
                train=False,
                image_dtype="float32",
                normalization_params=(0.0, 1.0),
            )
            results = modeling.evaluate(model, loader, autocast_dtype=ch.float32)
            if results_path is not None:
                ch.save(results, results_path)
            return results

    def get_aucs(self):
        if self.aucs_path.exists():
            return np.load(self.aucs_path)
        else:
            return np.zeros(len(self.configs))

    def update_aucs(self, index, value):
        with self.aucs_lock:
            aucs = self.get_aucs()
            aucs[index] = value
            np.save(self.aucs_path, aucs)

    def compute_auc_for_config_index(self, index):
        if self.get_aucs()[index] == 0:
            config = self.configs[index]
            model_path = self.save_path / "tuning" / f"model_index={index}"
            # We sample target indices so that the tuning dataset distribution matches the distribution used later
            indices = np.concatenate([self.source_indices, self.target_indices])
            train_indices, test_indices = train_test_split(indices, train_size=(self.num_folds - 1) / self.num_folds, random_state=0)
            test_results = self._get_results(config, model_path, train_indices, test_indices)
            auc = roc_auc_score(test_results["labels"].cpu(), test_results["outputs"][:, 1].cpu())
            self.update_aucs(index, auc or -1)

    @property
    def selected_config(self):
        if len(self.configs) == 1:
            return self.configs[0]
        for index in range(len(self.configs)):
            self.compute_auc_for_config_index(index)
        return self.configs[np.argmax(self.get_aucs())]

    def get_density_ratios_for_fold(self, fold, train_indices, evaluation_indices):
        model_path = self.save_path / "folds" / f"model_fold={fold}"
        evaluation_results_path = self.save_path / "folds" / f"evaluation_results_fold={fold}.pt"
        evaluation_results = self._get_results(
            self.selected_config,
            model_path,
            train_indices,
            evaluation_indices,
            results_path=evaluation_results_path,
        )
        logits = evaluation_results["outputs"]
        return logits[:, 1] - logits[:, 0]

    def get_density_ratios(self, fold=None):
        indices = np.concatenate([self.source_indices, self.target_indices])
        if self.target_groups is None:
            kf = KFold(n_splits=self.num_folds, random_state=0, shuffle=True)
            groups = None
        else:
            kf = GroupKFold(n_splits=self.num_folds)
            source_groups = -(np.arange(len(self.source_indices)) + 1)
            groups = np.concatenate([source_groups, self.target_groups])
        num_source_samples = len(self.source_indices)
        num_target_samples = len(self.target_indices)
        logits = np.zeros(len(indices))
        for cur_fold, (train_indices, evaluation_indices) in enumerate(kf.split(np.arange(len(indices)), groups=groups)):
            if fold is not None and cur_fold != fold:
                continue
            logits[evaluation_indices] = self.get_density_ratios_for_fold(cur_fold, indices[train_indices], indices[evaluation_indices])
        labels = np.concatenate([np.zeros(num_source_samples), np.ones(num_target_samples)])
        alpha = calibrate(logits, labels)
        probs = 1 / (1 + np.exp(-logits * alpha))
        ratios = (probs / (1 - probs)) * (num_source_samples / num_target_samples)
        return {
            "source_logits": logits[:num_source_samples],
            "target_logits": logits[num_source_samples:],
            "source_probs": probs[:num_source_samples],
            "target_probs": probs[num_source_samples:],
            "source_density_ratios": ratios[:num_source_samples],
            "target_density_ratios": ratios[num_source_samples:],
        }

    @property
    def computed(self):
        computed = True
        for fold in range(self.num_folds):
            computed = computed and (self.save_path / "folds" / f"evaluation_results_fold={fold}.pt").exists()
        return computed
