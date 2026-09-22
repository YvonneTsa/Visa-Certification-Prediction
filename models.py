"""Model zoo, cross-validation and hyperparameter tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import RANDOM_STATE

# Recall is the tuning/selection objective (minimise false negatives).
RECALL_SCORER = metrics.make_scorer(metrics.recall_score)


def model_zoo(random_state: int = RANDOM_STATE) -> list[tuple[str, object]]:
    """Return the list of (name, estimator) pairs compared in the project."""
    return [
        ("Bagging", BaggingClassifier(random_state=random_state)),
        ("Random Forest", RandomForestClassifier(random_state=random_state)),
        ("GBM", GradientBoostingClassifier(random_state=random_state)),
        ("AdaBoost", AdaBoostClassifier(random_state=random_state)),
        ("XGBoost", XGBClassifier(random_state=random_state, eval_metric="logloss")),
        ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
    ]


def oversample(X, y, random_state: int = RANDOM_STATE):
    """Balance the classes with SMOTE (synthetic minority oversampling)."""
    return SMOTE(sampling_strategy=1, k_neighbors=5, random_state=random_state).fit_resample(X, y)


def undersample(X, y, random_state: int = RANDOM_STATE):
    """Balance the classes by randomly undersampling the majority class."""
    return RandomUnderSampler(sampling_strategy=1, random_state=random_state).fit_resample(X, y)


def cross_validate_zoo(X_train, y_train, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """5-fold stratified cross-validated recall for every model in the zoo."""
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []
    for name, model in model_zoo(random_state):
        scores = cross_val_score(model, X_train, y_train, scoring=RECALL_SCORER, cv=kfold, n_jobs=-1)
        rows.append({"model": name, "cv_recall_mean": scores.mean(), "cv_recall_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_recall_mean", ascending=False).reset_index(drop=True)


def tune_adaboost(X, y, n_iter: int = 50, random_state: int = RANDOM_STATE) -> RandomizedSearchCV:
    """Recall-scored RandomizedSearchCV for AdaBoost."""
    param_grid = {
        "n_estimators": [50, 75, 100, 125],
        "learning_rate": [1.0, 0.5, 0.1, 0.01],
        "estimator": [
            DecisionTreeClassifier(max_depth=1, random_state=random_state),
            DecisionTreeClassifier(max_depth=2, random_state=random_state),
            DecisionTreeClassifier(max_depth=3, random_state=random_state),
        ],
    }
    search = RandomizedSearchCV(
        estimator=AdaBoostClassifier(random_state=random_state),
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=RECALL_SCORER,
        cv=5,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def tune_gradient_boosting(X, y, n_iter: int = 50, random_state: int = RANDOM_STATE) -> RandomizedSearchCV:
    """Recall-scored RandomizedSearchCV for Gradient Boosting."""
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "init": [None, DecisionTreeClassifier(max_depth=1, random_state=random_state)],
    }
    search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=RECALL_SCORER,
        cv=5,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search
