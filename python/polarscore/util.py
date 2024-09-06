from __future__ import annotations

import polars as pl


def _jeffrey_divergence(
    df: pl.DataFrame | pl.LazyFrame, x: str, y: str
) -> pl.DataFrame | pl.LazyFrame:
    """
    Calculate the Jeffrey divergence between two distributions.

    This function computes the Jeffrey divergence, which is a symmetric version of the
    Kullback-Leibler divergence, between two distributions defined by the columns x and
    y in the input DataFrame or LazyFrame.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The input DataFrame or LazyFrame containing the data.
    x : str
        The name of the column representing the feature or category.
    y : str
        The name of the column representing the binary target variable.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        A DataFrame or LazyFrame containing a single row with two columns:
        - 'var': The name of the feature (x)
        - 'val': The calculated Jeffrey divergence

    Raises
    ------
    ValueError
        If the target variable 'y' does not contain exactly two unique values.

    Notes
    -----
    The function works with both eager (DataFrame) and lazy (LazyFrame) execution.
    The type of the output matches the input type.
    """
    y_val = df[y].unique()

    if len(y_val) != 2:
        msg = f"Expected 2 unique values in '{y}', but found {len(y_val)}"
        raise ValueError(msg)

    v1 = y_val[0]
    v2 = y_val[1]

    v1_s = f"{v1}"
    v2_s = f"{v2}"

    df_divergence = (
        df.group_by(x)
        .agg(
            pl.col(y).eq(v1).sum().alias(v1_s),
            pl.col(y).eq(v2).sum().alias(v2_s),
        )
        .select(pl.col(v1_s, v2_s) / pl.col(v1_s, v2_s).sum())
        .select(
            var=pl.lit(x),
            val=(
                (pl.col(v1_s) - pl.col(v2_s)) * (pl.col(v1_s) / pl.col(v2_s)).log()
            ).sum(),
        )
    )

    return df_divergence
