import pickle
import numpy as np
import torch as ch
from filelock import FileLock
from tqdm.notebook import tqdm
from torch.cuda.amp import autocast
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import sklearn.metrics as metrics

from src.robustness_utils import get_effective_robustness


def worst_group_accuracy_fn(preds, labels, num_classes, groups=None, **kwargs):
    assert groups is not None
    corrects = preds.argmax(dim=1) == labels
    worst_group_accuracy = 1
    for group in np.unique(groups):
        if group >= 0:
            group_accuracy = (corrects[groups==group].sum() / (groups == group).sum()).item()
            worst_group_accuracy = min(worst_group_accuracy, group_accuracy)
    return worst_group_accuracy


def balanced_accuracy_fn(preds, labels, num_classes, **kwargs):
    return metrics.balanced_accuracy_score(labels.cpu(), preds.argmax(dim=1).cpu())


def accuracy_fn(preds, labels, num_classes, **kwargs):
    return ((preds.argmax(dim=1) == labels).sum() / len(labels)).item()


def macro_f1_fn(preds, labels, num_classes, **kwargs):
    return metrics.f1_score(labels.cpu(), preds.argmax(dim=1).cpu(), average="macro", labels=ch.unique(labels).cpu())


def auroc_fn(preds, labels, num_classes, **kwargs):
    assert preds.shape[1] == 2
    return 2 * metrics.roc_auc_score(labels.cpu(), preds[:, 1].cpu()) - 1


metric_fns = {
    "balanced_accuracy": balanced_accuracy_fn,
    "accuracy": accuracy_fn,
    "worst_group_accuracy": worst_group_accuracy_fn,
    "macro_f1": macro_f1_fn,
    "auroc": auroc_fn,
}


def get_lock_path(path):
    return f"{path}.lock"


class ExperimentManager:
    def __init__(self, path):
        self.path = Path(path)
        self._loaders = {}
        self._custom_model_paths = {}

        self.tuning_model_managers = {}
        self.model_managers = {}
        self.model_managers.update(self._make_tuning_model_managers())
        self.model_managers.update(self._make_model_managers())

    def _make_model_managers(self):
        raise NotImplementedError

    def get_ffcv_dataset(self, split):
        raise NotImplementedError

    def get_indices(self, split):
        raise NotImplementedError

    def get_groups(self, split):
        return None

    def get_sub_indices(self, split, sub_split):
        raise NotImplementedError

    def get_loader(self, split):
        raise NotImplementedError

    def get_loader_from_cache(self, split):
        if split in self._loaders:
            loader = self._loaders[split]
        else:
            loader = self.get_loader(split)
            self._loaders[split] = loader
        return loader

    @property
    def splits(self):
        raise NotImplementedError

    def get_model(self, model_name, index=None, retrain=False, ignore_untrained=False):
        path = self.get_model_path(model_name, index=index)
        manager = self.model_managers[model_name]
        if manager.num_copies is None:
            assert index is None
        else:
            assert index is not None
        if not manager.trained(path) and ignore_untrained:
            return None
        return manager.train_and_load(path, overwrite=retrain, seed=index)

    @property
    def model_names(self):
        return list(self.model_managers.keys())

    @property
    def model_names_with_copies(self):
        model_names = []
        for model_name, manager in self.model_managers.items():
            if manager.num_copies is None:
                model_names.append((model_name, None))
            else:
                for index in range(manager.num_copies):
                    model_names.append((model_name, index))
        return model_names

    @property
    def model_groups(self):
        return {model_name: manager.group for model_name, manager in self.model_managers.items()}

    def get_model_names(self, group=None):
        if group is None:
            return self.model_names
        else:
            return [model_name for model_name in self.model_names if self.model_groups[model_name] == group]

    def get_model_path(self, model_name, index=None):
        if (model_name, index) in self._custom_model_paths:
            return self._custom_model_paths[(model_name, index)]
        return self.path / "models" / (model_name if index is None else f"{model_name}_index={index}")

    def get_preds(self, model_name, split_name, index=None, verbose=False, repredict=False, retrain=False, ignore_untrained=False, ignore_unpredicted=False):
        preds_path = self.path / "preds" / split_name / (f"{model_name}_preds.pt" if index is None else f"{model_name}_index={index}_preds.pt")
        if ignore_unpredicted and not preds_path.exists():
            return None
        if preds_path.exists() and not repredict:
            try:
                return ch.load(preds_path).cuda()
            except EOFError:
                print(f"Corrupted predictions file {preds_path}, repredicting...")
        model = self.get_model(model_name, index=index, retrain=retrain, ignore_untrained=ignore_untrained)
        if model is None:
            return None
        model.eval()
        loader = self.get_loader_from_cache(split_name)
        preds = []
        # Some loaders apply a random augmentation/sampling; we would like to fix this when evaluating
        ch.random.manual_seed(0)
        np.random.seed(0)
        for example in tqdm(loader, disable=not verbose):
            x = example[0]
            # Occasional issues with ch.float16, so we use ch.float32 instead
            with ch.no_grad(), autocast(dtype=ch.float32):
                preds.append(model(x))
        preds = ch.cat(preds)
        preds_path.parent.mkdir(exist_ok=True, parents=True)
        ch.save(preds, preds_path)
        return preds

    def get_labels(self, split_name):
        labels_path = self.path / split_name / "labels.pt"
        if labels_path.exists():
            try:
                return ch.load(labels_path).cuda()
            except EOFError:
                print(f"Corrupted labels file {labels_path}, recreating...")
        loader = self.get_loader_from_cache(split_name)
        labels = []
        # Some loaders apply a random augmentation/sampling; we would like to fix this when evaluating
        ch.random.manual_seed(0)
        np.random.seed(0)
        for example in loader:
            y = deepcopy(example[1])
            labels.append(y)
        labels = ch.cat(labels)
        labels_path.parent.mkdir(exist_ok=True, parents=True)
        # with FileLock(get_lock_path(labels_path)):
        ch.save(labels, labels_path)
        return labels

    def _postprocess_preds(self, preds, split_name, sub_split_name):
        return preds

    def _process_labels(self, labels, split_name, sub_split_name):
        return labels

    def _compute_metric(self, model_name, index, split_name, sub_split_name, metric_fn, verbose, repredict, retrain, ignore_untrained, ignore_unpredicted):
         preds = self.get_preds(model_name, split_name, index=index, verbose=verbose, repredict=repredict, retrain=retrain, ignore_untrained=ignore_untrained, ignore_unpredicted=ignore_unpredicted)
         if preds is None:
            return None
         else:
             labels = self.get_labels(split_name)
             groups = self.get_groups(split_name)
             if sub_split_name is not None:
                 sub_indices = self.get_sub_indices(split_name, sub_split_name)
                 preds = preds[sub_indices]
                 labels = labels[sub_indices]
                 if groups is not None:
                    groups = groups[sub_indices]
             _, num_classes = preds.shape
             preds = self._postprocess_preds(preds, split_name, sub_split_name)
             labels = self._process_labels(labels, split_name, sub_split_name)
             return metric_fn(preds, labels, num_classes, groups=groups)

    @staticmethod
    def _aggregate_metric_values(metric_values):
        if len(metric_values) == 1:
            error = None
        else:
            error = 1.96 * np.std(metric_values) / np.sqrt(len(metric_values))
        return np.mean(metric_values), error

    def compute_copies_metrics(self, model_name, split_name, sub_split_name=None, metric_name="accuracy", custom_metric_fn=None, verbose=False, repredict=False, retrain=False, ignore_untrained=True, ignore_unpredicted=True):
        if metric_name in metric_fns:
            metric_fn = metric_fns[metric_name]
        else:
            metric_fn = custom_metric_fn
        metric_value_list = []
        num_copies = self.model_managers[model_name].num_copies
        assert num_copies is not None
        for index in range(num_copies):
            metric_value = self._compute_metric(model_name, index, split_name, sub_split_name, metric_fn, verbose, repredict, retrain, ignore_untrained, ignore_unpredicted)
            if metric_value is not None:
                metric_value_list.append(metric_value)
        return metric_value_list

    def get_metrics(self, split_name, sub_split_name=None, model_names=None, metric_name="accuracy", custom_metric_fn=None, verbose=False, recompute=False, repredict=False, retrain=False, group=False, model_groups=None, ignore_untrained=True, ignore_unpredicted=True, return_errors=False):
        if metric_name in metric_fns:
            metric_fn = metric_fns[metric_name]
        else:
            metric_fn = custom_metric_fn

        if sub_split_name is None:
            metrics_split_name = split_name
        else:
            metrics_split_name = f"{split_name}_{sub_split_name}"

        metrics = {}
        metrics_path = self.path / "metrics.pkl"
        if metrics_path.exists():
            with FileLock(self.path / "metrics.pkl.lock"):
                with open(metrics_path, "rb") as f:
                    try:
                        metrics.update(pickle.load(f))
                    except EOFError:
                        pass

        metric_values = {}
        metric_errors = {}
        model_names = self.model_names if model_names is None else model_names
        for model_name in tqdm(model_names, disable=not verbose):
            if metrics_split_name not in metrics:
                metrics[metrics_split_name] = {}
            if model_name not in metrics[metrics_split_name]:
                metrics[metrics_split_name][model_name] = {}

            if metric_name in metrics[metrics_split_name][model_name] and not recompute:
                metric_values[model_name], metric_errors[model_name] = metrics[metrics_split_name][model_name][metric_name]
            else:
                num_copies = self.model_managers[model_name].num_copies
                error = None
                if num_copies is None:
                    metric_value = self._compute_metric(model_name, None, split_name, sub_split_name, metric_fn, verbose, repredict, retrain, ignore_untrained, ignore_unpredicted)
                else:
                    metric_value_list = self.compute_copies_metrics(model_name, split_name, sub_split_name=sub_split_name, metric_name=metric_name, custom_metric_fn=custom_metric_fn, verbose=verbose, repredict=repredict, retrain=retrain, ignore_untrained=ignore_untrained, ignore_unpredicted=ignore_unpredicted)
                    if len(metric_value_list) > 0:
                        metric_value, error = self._aggregate_metric_values(metric_value_list)
                    else:
                        metric_value = None
                if metric_value is not None:
                    metrics[metrics_split_name][model_name][metric_name] = metric_value, error
                    with FileLock(self.path / "metrics.pkl.lock"):
                        with open(metrics_path, "wb") as f:
                            pickle.dump(metrics, f)
                    metric_values[model_name] = metric_value
                    metric_errors[model_name] = error
        if group:
            model_groups = self.model_groups if model_groups is None else model_groups
            group_metric_values = defaultdict(list)
            for model_name, metric_value in metric_values.items():
                group_metric_values[model_groups[model_name]].append(metric_value)
            group_metric_errors = defaultdict(list)
            for model_name, error in metric_errors.items():
                group_metric_errors[model_groups[model_name]].append(error)
            for group, group_errors in group_metric_errors.items():
                if len(group_errors) == 0 or group_errors[0] is None:
                    group_metric_errors[group] = None
            if return_errors:
                return group_metric_values, group_metric_errors
            else:
                return group_metric_values
        else:
            if return_errors:
                return metric_values, metric_errors
            else:
                return metric_values

    def get_effective_robustness(self, ref_split_name, shift_split_name, baseline_group=None, baseline_model_names=None, ref_sub_split_name=None, shift_sub_split_name=None, ref_metric_name="accuracy", ref_custom_metric_fn=None, shift_metric_name="accuracy", shift_custom_metric_fn=None, scale="probit", with_confidence_interval=False, **kwargs):
        if baseline_model_names is None:
            assert baseline_group is not None
            baseline_model_names = [model_name for model_name in self.model_names if self.model_groups[model_name] == baseline_group]
        baseline_kwargs = kwargs.copy()
        if "model_names" in baseline_kwargs:
            del baseline_kwargs["model_names"]
        baseline_ref_values = self.get_metrics(ref_split_name, sub_split_name=ref_sub_split_name, model_names=baseline_model_names, metric_name=ref_metric_name, custom_metric_fn=ref_custom_metric_fn, **baseline_kwargs)
        baseline_shift_values = self.get_metrics(shift_split_name, sub_split_name=shift_sub_split_name, model_names=baseline_model_names, metric_name=shift_metric_name, custom_metric_fn=shift_custom_metric_fn, **baseline_kwargs)
        ref_values = self.get_metrics(ref_split_name, sub_split_name=ref_sub_split_name, metric_name=ref_metric_name, custom_metric_fn=ref_custom_metric_fn, **kwargs)
        shift_values = self.get_metrics(shift_split_name, sub_split_name=shift_sub_split_name, metric_name=shift_metric_name, custom_metric_fn=shift_custom_metric_fn, **kwargs)
        return get_effective_robustness(baseline_ref_values, baseline_shift_values, ref_values, shift_values, scale=scale, with_confidence_interval=with_confidence_interval)

    @property
    def _tuning_split(self):
        return "source_val"

    @property
    def _tuning_metric(self):
        return "accuracy"

    def _get_tuning_specs(self):
        return []

    @staticmethod
    def _get_tuning_config_description(index, config):
        return f"index={index}"

    def _create_tuning_model_managers(self, model_name, manager_cls, kwargs, configs):
        tuning_model_managers = {}
        for index, config in enumerate(configs):
            current_kwargs = kwargs.copy()
            current_kwargs["config"] = config
            description = self._get_tuning_config_description(index, config)
            tuning_model_managers[f"{model_name}_config_{description}"] = manager_cls(**current_kwargs)
        self.tuning_model_managers[model_name] = manager_cls, kwargs, tuning_model_managers
        return tuning_model_managers

    def _make_tuning_model_managers(self):
        tuning_specs = self._get_tuning_specs()
        model_managers = {}
        for model_name, manager_cls, kwargs, configs in tuning_specs:
            model_managers.update(self._create_tuning_model_managers(model_name, manager_cls, kwargs, configs))
        return model_managers

    def _get_selected_specs(self, model_name):
        manager_cls, kwargs, tuning_model_managers = self.tuning_model_managers[model_name]
        source_metrics = self.get_metrics(
            self._tuning_split,
            model_names=list(tuning_model_managers.keys()),
            metric_name=self._tuning_metric,
            ignore_untrained=True,
            recompute=True,
        )
        if len(source_metrics) < len(tuning_model_managers):
            selected_config = None
        else:
            best_model_name = max(source_metrics, key=source_metrics.get)
            selected_config = tuning_model_managers[best_model_name].config
        selected_kwargs = kwargs.copy()
        selected_kwargs["config"] = selected_config
        return manager_cls, selected_kwargs
