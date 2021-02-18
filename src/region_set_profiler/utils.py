import pickle
import re

import pandas as pd


def parquet(fp, suffix=".p"):
    return re.sub(f"{suffix}$", ".parquet", fp)


def p(fp, suffix=".tsv"):
    return re.sub(f"{suffix}$", ".p", fp)


def assert_granges_are_sorted(df: pd.DataFrame):
    """

    Args:
        df: must have columns ['Chromosome', 'Start', 'End']

    Raises:
        AssertionError if df is not sorted on ['Chromosome', 'Start', 'End']

    """
    assert df["Chromosome"].is_monotonic_increasing
    assert (
        df.groupby("Chromosome")[["Start", "End"]]
        .agg(lambda ser: ser.is_monotonic_increasing)
        .all()
        .all()
    )


def to_pickle(obj, fp, protocol=4):
    with open(fp, "wb") as fout:
        pickle.dump(obj, fout, protocol=protocol)


def from_pickle(fp):
    with open(fp, "rb") as fin:
        return pickle.load(fin)
