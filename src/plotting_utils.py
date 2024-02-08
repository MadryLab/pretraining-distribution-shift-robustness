import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logit, expit

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FuncFormatter

from src.robustness_utils import get_line_with_confidence_interval

class ProbitScale(mscale.ScaleBase):
    name = 'probit'

    def get_transform(self):
        return self.ProbitTransform()

    def set_default_locators_and_formatters(self, axis):
        # fmt = FuncFormatter(lambda x, pos=None: f"{round(x * 100):.0f}")
        def display_percent(x):
            percent = x * 100
            if np.abs(round(percent) - percent) < 1e-6:
                return round(percent)
            else:
                return "{:.6g}".format(percent)
        fmt = FuncFormatter(lambda x, pos=None: str(display_percent(x)))
        axis.set(major_formatter=fmt, minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 0.0), min(vmax, 1.0)

    class ProbitTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, a):
            return norm.ppf(a)

        def inverted(self):
            return ProbitScale.InvertedProbitTransform()

    class InvertedProbitTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, a):
            return norm.cdf(a)

        def inverted(self):
            return ProbitScale.ProbitTransform()
        
mscale.register_scale(ProbitScale)


class LogitScale(mscale.ScaleBase):
    name = 'logit'

    def get_transform(self):
        return self.LogitTransform()

    def set_default_locators_and_formatters(self, axis):
        def display_percent(x):
            percent = x * 100
            if np.abs(round(percent) - percent) < 1e-6:
                return round(percent)
            else:
                return "{:.6g}".format(percent)
        fmt = FuncFormatter(lambda x, pos=None: str(display_percent(x)))
        axis.set(major_formatter=fmt, minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 0.0), min(vmax, 1.0)

    class LogitTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, a):
            return logit(a)

        def inverted(self):
            return LogitScale.InvertedLogitTransform()

    class InvertedLogitTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, a):
            return expit(a)

        def inverted(self):
            return LogitScale.LogitTransform()
        
mscale.register_scale(LogitScale)


metric_display_names = {
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy": "Accuracy",
    "worst_group_accuracy": "Worst-Group Accuracy",
    "macro_f1": "Macro F1",
    "auroc": "AUROC",
}


def scatter(x, y, label, x_errors=None, y_errors=None, plot_line=False, scatter_alpha=0.5, line_alpha=0.25, marker=None, s=20, c=None, scale="probit", ax=None):
    ax = ax or plt.gca()
    if x_errors is None:
        p = ax.plot(x, y, alpha=scatter_alpha, markersize=s/4, label=label, marker=marker or 'o', linestyle="None", c=c)
    else:
        p = ax.errorbar(x, y, xerr=x_errors, yerr=y_errors, alpha=scatter_alpha, markersize=s/4, label=label, marker=marker, fmt='o', capsize=2, c=c)

    if plot_line:
        reference = np.linspace(0.01, 0.99, 991)
        get_prediction_with_confidence_interval = get_line_with_confidence_interval(x, y, scale=scale)
        lower, y_hat, upper = get_prediction_with_confidence_interval(reference)
        x1, x2 = reference[[0, -1]]
        y1, y2 = y_hat[[0, -1]]
        ax.axline((x1, y1), (x2, y2), alpha=scatter_alpha, label=f"{label} linear fit", color=p[0].get_color())
        ax.fill_between(reference, lower, upper, alpha=line_alpha, label=None, color=p[0].get_color())


def plot_on_the_line(
    manager,
    model_names=None,
    model_groups=None,
    recompute=False,
    repredict=False,
    retrain=False,
    ignore_untrained=True,
    ignore_unpredicted=True,
    plot_lines_groups=["Baseline"],
    plot_without_error_groups=(),
    source_metric_name="accuracy",
    source_metric_display_name=None,
    source_custom_metric_fn=None,
    source_split="source_val",
    source_sub_split=None,
    target_split="target_val",
    target_sub_split=None,
    target_metric_name="accuracy",
    target_metric_display_name=None,
    target_custom_metric_fn=None,
    title=None,
    verbose=False,
    scatter_alpha=0.5,
    line_alpha=0.25,
    include_legend=True,
    legend_fontsize=8,
    legend_loc=0,
    scale="probit",
    ax=None,
    custom_styles={},
):
    ax = ax or plt.gca()

    source_values, source_errors = manager.get_metrics(source_split, sub_split_name=source_sub_split, model_names=model_names, metric_name=source_metric_name, custom_metric_fn=source_custom_metric_fn, recompute=recompute, repredict=repredict, retrain=retrain, ignore_untrained=ignore_untrained, ignore_unpredicted=ignore_unpredicted, group=True, model_groups=model_groups, verbose=verbose, return_errors=True)
    target_values, target_errors = manager.get_metrics(target_split, sub_split_name=target_sub_split, model_names=model_names, metric_name=target_metric_name, custom_metric_fn=target_custom_metric_fn, recompute=recompute, repredict=repredict, retrain=retrain, ignore_untrained=ignore_untrained, ignore_unpredicted=ignore_unpredicted, group=True, model_groups=model_groups, verbose=verbose, return_errors=True)

    for group in plot_without_error_groups:
        source_errors[group] = None
        target_errors[group] = None

    for group in source_values:
        scatter(
            source_values[group],
            target_values[group],
            group,
            x_errors=source_errors[group],
            y_errors=target_errors[group],
            plot_line=group in plot_lines_groups,
            scatter_alpha=scatter_alpha,
            line_alpha=line_alpha,
            scale=scale,
            **custom_styles.get(group, {}),
            ax=ax
        )

    source_metric_display_name = source_metric_display_name or metric_display_names[source_metric_name]
    target_metric_display_name = target_metric_display_name or metric_display_names[target_metric_name]
    
    all_source_values = np.concatenate([values for values in source_values.values()])
    all_target_values = np.concatenate([values for values in target_values.values()])
    source_min, source_max = all_source_values.min(), all_source_values.max()
    target_min, target_max = all_target_values.min(), all_target_values.max()
    target_min = min(target_min, source_min)
    target_max = max(target_max, source_max)

    def get_lim(min_, max_):
        transform = mscale.scale_factory(scale, 0).get_transform().transform_non_affine
        inverse_transform = mscale.scale_factory(scale, 0).get_transform().inverted().transform_non_affine
        t_min_, t_max_ = transform(min_), transform(max_)
        padding = (t_max_ - t_min_) * 0.05
        return inverse_transform(t_min_ - padding), inverse_transform(t_max_ + padding)
    ax.axline((source_min, source_min), (source_max, source_max), color="dimgrey", linestyle="--", label=r"$y=x$")
    ax.set_xlim(get_lim(source_min, source_max))
    ax.set_ylim(get_lim(target_min, target_max))
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlabel(f"Reference {source_metric_display_name}")
    ax.set_ylabel(f"Shifted {target_metric_display_name}")
    ax.grid(linewidth=0.5, alpha=0.8)
    if include_legend:
        ax.legend(prop={"size": legend_fontsize}, loc=legend_loc)
    ax.set_title(title)