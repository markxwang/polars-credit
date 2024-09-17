from __future__ import annotations

import polars as pl


def _get_cols(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    elif isinstance(df, pl.DataFrame):
        return df.columns
    else:
        msg = "Input must be a Polars DataFrame or LazyFrame"
        raise TypeError(msg)
