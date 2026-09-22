"""Feature encoding and train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from . import RANDOM_STATE, TARGET


@dataclass
class DataSplits:
    """Container for the encoded feature matrix and its splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``case_status`` to 1 (Certified) / 0 (Denied)."""
    df = df.copy()
    df[TARGET] = (df[TARGET] == "Certified").astype(int)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode predictors and separate the target.

    Uses ``drop_first=True`` to avoid the dummy-variable trap.
    """
    df = encode_target(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    holdout_size: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Create stratified train / validation / test splits.

    Mirrors the project design: a 70/30 train--validation split, then a
    further 90/10 split of the validation set into validation and test.
    All splits are stratified on the target to preserve class balance.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=holdout_size, random_state=random_state, stratify=y_val
    )
    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=list(X.columns),
    )
