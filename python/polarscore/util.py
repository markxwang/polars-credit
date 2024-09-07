from __future__ import annotations

import polars as pl


def _jeffrey_divergence(
    df: pl.DataFrame | pl.LazyFrame,
    x: str,
    y: str,
    benchmark=None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Calculate the Jeffrey divergence between two categorical variables.

    This function computes the Jeffrey divergence, which is a symmetric version of the
    Kullback-Leibler divergence, between the distribution of a categorical variable 'x'
    and a benchmark category in another categorical variable 'y'.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The input dataframe containing the variables to be analyzed.
    x : str
        The name of the categorical variable for which to calculate the divergence.
    y : str
        The name of the categorical variable containing the benchmark category.
    benchmark : str, optional
        The benchmark category in 'y' to compare against. If None, the first unique
        value in 'y' is used as the benchmark.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        A dataframe containing the Jeffrey divergence for each category in 'x'.
        The result has two columns:
        - 'var': The name of the variable 'x'.
        - 'val': The calculated Jeffrey divergence value.

    Raises
    ------
    ValueError
        If the specified benchmark value is not found in the unique values of 'y'.

    Notes
    -----
    The Jeffrey divergence is calculated as the maximum of the sum of the divergences
    in both directions between each category in 'x' and the benchmark category in 'y'.
    """
    y_unique = df.select(pl.col(y)).unique().sort(y)
    if isinstance(df, pl.LazyFrame):
        y_unique = y_unique.collect()
    y_unique = y_unique.to_series().to_list()

    if benchmark is None:
        benchmark = y_unique[0]

    if benchmark not in y_unique:
        msg = f"Benchmark value '{benchmark}' not found in unique values of '{y}'"
        raise ValueError(msg)

    df_divergence = (
        df.group_by(x)
        .agg(pl.col(y).eq(y_val).sum().alias(f"{y_val}") for y_val in y_unique)
        .drop(x)
        .select(pl.all() / pl.all().sum())
        .select(
            (pl.all() - pl.col(f"{benchmark}"))
            * (pl.all() / pl.col(f"{benchmark}")).log()
        )
        .select(var=pl.lit(x), val=pl.max_horizontal(pl.all().sum()))
    )

    return df_divergence
