"""Data loading and cleaning for the EasyVisa dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw EasyVisa CSV.

    Parameters
    ----------
    path : str or Path
        Path to ``EasyVisa.csv``.

    Returns
    -------
    pandas.DataFrame
        The raw dataframe, unmodified.
    """
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw EasyVisa dataframe.

    Applies the data-quality fixes established during EDA:

    * ``no_of_employees`` contains a handful of negative values that are
      data-entry sign errors; they are converted to their absolute value.
    * ``case_id`` is a unique identifier with no predictive value and is
      dropped.

    The input is not mutated; a cleaned copy is returned.
    """
    df = df.copy()
    df["no_of_employees"] = df["no_of_employees"].abs()
    if "case_id" in df.columns:
        df = df.drop(columns=["case_id"])
    return df


def load_clean(path: str | Path) -> pd.DataFrame:
    """Convenience wrapper: :func:`load_raw` followed by :func:`clean`."""
    return clean(load_raw(path))
