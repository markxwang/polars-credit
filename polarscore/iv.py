import polars as pl
import polars.selectors as cs
from typing import Union


def get_woe(
    df: Union[pl.DataFrame, pl.LazyFrame], y: str, x: str
) -> Union[pl.DataFrame, pl.LazyFrame]:

    df_woe = (
        df.group_by(x)
        .agg(
            pl.col(y).eq(0).sum().alias("good"),
            pl.col(y).eq(1).sum().alias("bad"),
        )
        .with_columns(pl.col("good", "bad") / pl.col("good", "bad").sum())
        .with_columns((pl.col("bad") / pl.col("good")).log().alias("woe"))
        .with_columns((pl.col("woe") * (pl.col("bad") - pl.col("good"))).alias("iv"))
    )

    return df_woe


def calculate_iv(
    df: Union[pl.DataFrame, pl.LazyFrame], y: str
) -> Union[pl.DataFrame, pl.LazyFrame]:
    cols_x = cs.expand_selector(df, cs.exclude(y))

    ls_iv = [
        get_woe(df, x, y).select(pl.lit(x).alias("var"), pl.col("iv").sum())
        for x in cols_x
    ]
    df_iv = pl.concat(ls_iv)
    return df_iv
