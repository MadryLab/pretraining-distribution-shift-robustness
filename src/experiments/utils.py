import numpy as np
from itertools import product
from copy import deepcopy


def generate_configs(param_options, default_config):
    if isinstance(param_options, list):
        configs = []
        for sub_param_options in param_options:
            configs += generate_configs(sub_param_options, default_config)
        return configs
    param_values_options = list(param_options.values())
    config_options = product(*param_values_options)
    if default_config is None:
        return [None for _ in config_options]
    configs = []
    for param_values in config_options:
        config = deepcopy(default_config)
        for keys, values in zip(param_options, param_values):
            if isinstance(keys[0], str):
                keys = [keys]
                values = [values]
            for (section, param_name), value in zip(keys, values):
                config[section][param_name] = value
        configs.append(config)
    return configs



def sample_indices(labels, class_sampling_rates=None, class_sample_sizes=None):
    indices = np.arange(len(labels))
    sampled_indices = []
    if class_sample_sizes is None:
        assert class_sampling_rates is not None
        class_sample_sizes = [int(sampling_rate * (labels == label).sum()) for label, sampling_rate in enumerate(class_sampling_rates)]
    for label, sample_size in enumerate(class_sample_sizes):
        label_indices = indices[labels == label]
        cur_sampled_indices = np.random.RandomState(label).choice(label_indices, size=sample_size, replace=False)
        sampled_indices.append(cur_sampled_indices)
    return np.concatenate(sampled_indices)
