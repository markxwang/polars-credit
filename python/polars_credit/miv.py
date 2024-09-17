from __future__ import annotations

import polars as pl

from polars_credit.util.polars import _get_cols


def cal_miv(df: pl.DataFrame, y: str, y_pred: str, x: str) -> pl.DataFrame:
    """Calculate MIV."""
    df_miv = (
        df.group_by(x)
        .agg(
            (1 - pl.col(y)).sum().alias("good_actual"),
            pl.col(y).sum().alias("bad_actual"),
            (1 - pl.col(y_pred)).sum().alias("good_pred"),
            pl.col(y_pred).sum().alias("bad_pred"),
        )
        .with_columns(
            pl.col("good_actual", "bad_actual")
            / pl.col("good_actual", "bad_actual").sum(),
            pl.col("good_pred", "bad_pred") / pl.col("good_pred", "bad_pred").sum(),
        )
        .with_columns(
            (pl.col("bad_actual") / pl.col("good_actual")).log().alias("woe_actual"),
            (pl.col("bad_pred") / pl.col("good_pred")).log().alias("woe_pred"),
        )
        .with_columns(
            miv=(pl.col("woe_actual") - pl.col("woe_pred"))
            * (pl.col("bad_actual") - pl.col("good_actual"))
        )
    )

    return df_miv


def cal_multiple_miv(
    df: pl.DataFrame | pl.LazyFrame, y: str, y_pred: str
) -> pl.DataFrame:
    """Calculate MIV for multiple variables."""
    df_lazy = df.lazy()
    cols = _get_cols(df_lazy)

    ls_miv = [
        cal_miv(df_lazy, y=y, y_pred=y_pred, x=x) for x in cols if x not in (y, y_pred)
    ]

    df_miv = pl.concat(ls_miv).collect()

    return df_miv
