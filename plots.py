"""Reusable exploratory-data-analysis plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig_or_ax, save_path: str | Path | None):
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None, save_path=None):
    """Boxplot (with mean) stacked above a histogram for a numeric feature."""
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins)
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    _save(f2, save_path)
    return f2


def labeled_barplot(data, feature, perc=False, n=None, save_path=None):
    """Count/percentage bar plot with value labels on each bar."""
    total = len(data[feature])
    count = data[feature].nunique()
    plt.figure(figsize=((count if n is None else n) + 1, 5))
    plt.xticks(rotation=90, fontsize=12)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )
    for p in ax.patches:
        label = f"{100 * p.get_height() / total:.1f}%" if perc else int(p.get_height())
        ax.annotate(
            label,
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="center",
            size=11,
            xytext=(0, 5),
            textcoords="offset points",
        )
    _save(ax, save_path)
    return ax


def stacked_barplot(data, predictor, target, save_path=None):
    """Normalised stacked bar plot of ``target`` across ``predictor`` levels.

    Also returns the raw crosstab (counts) for reference.
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    counts = pd.crosstab(data[predictor], data[target], margins=True).sort_values(by=sorter, ascending=False)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(by=sorter, ascending=False)
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylabel("proportion")
    _save(None, save_path)
    return counts


def distribution_plot_wrt_target(data, predictor, target, save_path=None):
    """Distribution and boxplots of a numeric predictor split by target class."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    target_uniq = data[target].unique()

    axs[0, 0].set_title(f"Distribution of {predictor} for {target}={target_uniq[0]}")
    sns.histplot(
        data=data[data[target] == target_uniq[0]], x=predictor, kde=True, ax=axs[0, 0], color="teal", stat="density"
    )
    axs[0, 1].set_title(f"Distribution of {predictor} for {target}={target_uniq[1]}")
    sns.histplot(
        data=data[data[target] == target_uniq[1]], x=predictor, kde=True, ax=axs[0, 1], color="orange", stat="density"
    )
    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 1], showfliers=False, palette="gist_rainbow")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def feature_importance_plot(importances: pd.Series, top_n: int | None = None, save_path=None):
    """Horizontal bar plot of feature importances (already sorted descending)."""
    if top_n:
        importances = importances.head(top_n)
    importances = importances.sort_values()
    plt.figure(figsize=(10, max(6, len(importances) * 0.4)))
    plt.title("Feature importances")
    plt.barh(range(len(importances)), importances.values, color="violet", align="center")
    plt.yticks(range(len(importances)), importances.index)
    plt.xlabel("Relative importance")
    plt.tight_layout()
    _save(None, save_path)
    return plt.gca()
