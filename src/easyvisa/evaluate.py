"""Model evaluation helpers: metrics and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(model, X, y) -> pd.DataFrame:
    """Return accuracy, recall, precision and F1 for ``model`` on ``(X, y)``.

    Recall is the primary metric for this project: the costliest error is a
    false negative (a genuinely qualified applicant predicted as denied).
    """
    pred = model.predict(X)
    return pd.DataFrame(
        {
            "Accuracy": [accuracy_score(y, pred)],
            "Recall": [recall_score(y, pred)],
            "Precision": [precision_score(y, pred)],
            "F1": [f1_score(y, pred)],
        }
    )


def metrics_dict(model, X, y) -> dict[str, float]:
    """Same metrics as :func:`classification_metrics`, as a plain dict."""
    return {k: round(float(v[0]), 4) for k, v in classification_metrics(model, X, y).to_dict("list").items()}


def plot_confusion_matrix(model, X, y, title: str = "Confusion matrix", save_path: str | Path | None = None):
    """Plot a labelled confusion matrix (counts and percentages)."""
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    labels = np.asarray(
        [
            f"{count:0.0f}\n{count / cm.sum():.2%}"
            for count in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gca()


def feature_importance(model, feature_names) -> pd.Series:
    """Return feature importances as a sorted (descending) Series."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)
