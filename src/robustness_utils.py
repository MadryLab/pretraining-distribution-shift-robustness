import numpy as np
from scipy.stats import norm, linregress, t
from scipy.special import logit, expit


TRANSFORMS = {
    "probit": (norm.ppf, norm.cdf),
    "logit": (logit, expit),
}


def _dict_to_array(d):
    if isinstance(d, dict):
        return np.array(list(d.values())), list(d.keys())
    else:
        return d, None


def _array_to_dict(a, keys):
    if keys is None:
        return a
    else:
        return dict(zip(keys, a))


# https://real-statistics.com/regression/confidence-and-prediction-intervals
def get_line_with_confidence_interval(x, y, scale="probit", p=0.05):
    transform, inverse_transform = TRANSFORMS[scale]
    t_x = transform(x)
    t_y = transform(y)
    n = len(x)
    result = linregress(t_x, t_y)
    standard_error_estimate = np.std(t_y) * np.sqrt((1 - result.rvalue ** 2) * (n - 1) / (n - 2))
    squared_deviation = np.std(t_x) ** 2 * n
    t_crit = t(n - 2).ppf(1 - p / 2)
    def get_prediction_with_confidence_interval(x_test):
        t_x_test = transform(x_test)
        t_y_hat = t_x_test * result.slope + result.intercept
        t_error = t_crit * standard_error_estimate * np.sqrt(1 / n + (t_x_test - np.mean(t_x)) ** 2 / squared_deviation)
        return (
            inverse_transform(t_y_hat - t_error),
            inverse_transform(t_y_hat),
            inverse_transform(t_y_hat + t_error),
        )
    return get_prediction_with_confidence_interval


def get_effective_robustness_from_line(
    get_prediction_with_confidence_interval,
    ref_values,
    shift_values,
    with_confidence_interval=False,
):
    ref_values, keys = _dict_to_array(ref_values)
    shift_values, _ = _dict_to_array(shift_values)
    lower, predicted_shift_values, upper = get_prediction_with_confidence_interval(ref_values)
    if with_confidence_interval:
        effective_robustness = np.stack([
            shift_values - upper,
            shift_values - predicted_shift_values,
            shift_values - lower,
        ], axis=1)
    else:
        effective_robustness = shift_values - predicted_shift_values
    return _array_to_dict(effective_robustness, keys)


def get_effective_robustness(
    baseline_ref_values,
    baseline_shift_values,
    ref_values,
    shift_values,
    scale="probit",
    with_confidence_interval=False,
):
    baseline_ref_values, _ = _dict_to_array(baseline_ref_values)
    baseline_shift_values, _ = _dict_to_array(baseline_shift_values)
    get_prediction_with_confidence_interval = get_line_with_confidence_interval(
        baseline_ref_values,
        baseline_shift_values,
        scale=scale,
    )
    return get_effective_robustness_from_line(
        get_prediction_with_confidence_interval, 
        ref_values, 
        shift_values,
        with_confidence_interval=with_confidence_interval,
    )
