"""End-to-end training pipeline for the EasyVisa approval model.

Runs the full study on real data and writes reproducible artifacts:

* ``reports/metrics.json``  -- verified metrics for every stage
* ``reports/figures/*.png`` -- model-comparison and feature-importance charts
* ``models/*.joblib``       -- the persisted best model

Usage
-----
    python scripts/train.py                 # uses config.yaml
    python scripts/train.py --n-iter 50     # full tuning budget
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# Make the src/ package importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from easyvisa import data, evaluate, models, preprocess  # noqa: E402
from easyvisa.plots import feature_importance_plot  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EasyVisa approval model.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--n-iter", type=int, default=None, help="Override tuning n_iter.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    n_iter = args.n_iter or cfg["tuning"]["n_iter"]
    rs = cfg["split"]["random_state"]

    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["paths"]["model"]).parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # 1. Load & clean --------------------------------------------------------
    df = data.load_clean(ROOT / cfg["paths"]["data"])
    results["n_rows"], results["n_cols"] = df.shape
    print(f"[1/6] Loaded and cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Preprocess & split --------------------------------------------------
    X, y = preprocess.build_features(df)
    splits = preprocess.make_splits(
        X, y,
        test_size=cfg["split"]["test_size"],
        holdout_size=cfg["split"]["holdout_size"],
        random_state=rs,
    )
    results["split_shapes"] = {
        "train": list(splits.X_train.shape),
        "validation": list(splits.X_val.shape),
        "test": list(splits.X_test.shape),
    }
    print(f"[2/6] Split -> train {splits.X_train.shape}, val {splits.X_val.shape}, test {splits.X_test.shape}")

    # 3. Model spot-check across sampling strategies -------------------------
    print("[3/6] Cross-validating six models on three sampling strategies (this takes a moment)...")
    cv_original = models.cross_validate_zoo(splits.X_train, splits.y_train, rs)
    X_over, y_over = models.oversample(splits.X_train, splits.y_train, rs)
    cv_over = models.cross_validate_zoo(X_over, y_over, rs)
    X_under, y_under = models.undersample(splits.X_train, splits.y_train, rs)
    cv_under = models.cross_validate_zoo(X_under, y_under, rs)

    results["cv_recall"] = {
        "original": cv_original.set_index("model")["cv_recall_mean"].round(4).to_dict(),
        "oversampled": cv_over.set_index("model")["cv_recall_mean"].round(4).to_dict(),
        "undersampled": cv_under.set_index("model")["cv_recall_mean"].round(4).to_dict(),
    }

    # 4. Tune the top performers --------------------------------------------
    print(f"[4/6] Tuning AdaBoost (oversampled) and Gradient Boosting (undersampled), n_iter={n_iter}...")
    ada = models.tune_adaboost(X_over, y_over, n_iter=n_iter, random_state=rs)
    gbm = models.tune_gradient_boosting(X_under, y_under, n_iter=n_iter, random_state=rs)

    results["tuning"] = {
        "adaboost_oversampled": {
            "best_cv_recall": round(float(ada.best_score_), 4),
            "best_params": {k: str(v) for k, v in ada.best_params_.items()},
        },
        "gradient_boosting_undersampled": {
            "best_cv_recall": round(float(gbm.best_score_), 4),
            "best_params": {k: str(v) for k, v in gbm.best_params_.items()},
        },
    }

    ada_best = ada.best_estimator_
    gbm_best = gbm.best_estimator_

    # 5. Validation & test evaluation ---------------------------------------
    print("[5/6] Evaluating tuned models on validation and test sets...")
    results["validation"] = {
        "adaboost_oversampled": evaluate.metrics_dict(ada_best, splits.X_val, splits.y_val),
        "gradient_boosting_undersampled": evaluate.metrics_dict(gbm_best, splits.X_val, splits.y_val),
    }
    results["test_final_model"] = evaluate.metrics_dict(gbm_best, splits.X_test, splits.y_test)
    print(f"       Final model (GBM) test metrics: {results['test_final_model']}")

    # 6. Artifacts: figures, feature importance, saved model -----------------
    print("[6/6] Writing figures, feature importances and the persisted model...")
    evaluate.plot_confusion_matrix(
        gbm_best, splits.X_test, splits.y_test,
        title="Final model - confusion matrix (test set)",
        save_path=fig_dir / "confusion_matrix_test.png",
    )
    plt.close("all")

    importances = evaluate.feature_importance(gbm_best, splits.feature_names)
    results["feature_importance_top10"] = importances.head(10).round(4).to_dict()
    feature_importance_plot(importances, save_path=fig_dir / "feature_importance.png")
    plt.close("all")

    # model-comparison chart (validation recall, undersampled tuning family)
    ax = cv_under.set_index("model")["cv_recall_mean"].sort_values().plot(
        kind="barh", figsize=(8, 5), color="mediumpurple", title="Cross-validated recall (undersampled)"
    )
    ax.set_xlabel("Mean CV recall")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_recall.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    joblib.dump(gbm_best, ROOT / cfg["paths"]["model"])

    with open(ROOT / cfg["paths"]["metrics"], "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Artifacts written to reports/ and models/.")
    print(f"Final model test recall: {results['test_final_model']['Recall']:.3f}")


if __name__ == "__main__":
    main()
