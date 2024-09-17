from __future__ import annotations

import polars as pl
from polars_credit.util.polars import _get_cols


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


def _multi_jeffrey_divergence(df: pl.DataFrame | pl.LazyFrame, y: str):
    """
    Calculate Jeffrey divergence for multiple variables against a target variable.

    This function computes the Jeffrey divergence for each variable in the DataFrame
    (except the target variable) against the specified target variable.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The input DataFrame or LazyFrame containing the variables to analyze.
    y : str
        The name of the target variable column.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the Jeffrey divergence for each variable.
        The result has two columns:
        - 'var': The name of the variable.
        - 'val': The calculated Jeffrey divergence value.

    Notes
    -----
    This function uses the _jeffrey_divergence function to calculate the divergence
    for each variable. It excludes the target variable from the calculation.
    The result is collected into a single DataFrame.
    """
    df_lazy = df.lazy()
    cols = _get_cols(df_lazy)

    ls_iv = [_jeffrey_divergence(df_lazy, x=x, y=y) for x in cols if x != y]

    df_iv = pl.concat(ls_iv).collect()

    return df_iv


def cal_iv(df: pl.DataFrame | pl.LazyFrame, y: str):
    """
    Calculate Information Value (IV) for multiple variables against a target variable.

    This function computes the Information Value for each variable in the DataFrame
    (except the target variable) against the specified target variable. The IV is
    calculated using Jeffrey divergence.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The input DataFrame or LazyFrame containing the variables to analyze.
    y : str
        The name of the target variable column.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the Information Value for each variable.
        The result has two columns:
        - 'var': The name of the variable.
        - 'iv': The calculated Information Value.

    Notes
    -----
    This function uses the _multi_jeffrey_divergence function to calculate the
    divergence for each variable. The resulting values are interpreted as Information
    Values.
    """
    df_iv = _multi_jeffrey_divergence(df, y).rename({"val": "iv"})
    return df_iv


def cal_psi(df: pl.DataFrame | pl.LazyFrame, t: str):
    """
    Calculate Population Stability Index for multiple variables against a time var.

    This function computes the Population Stability Index for each variable in the
    DataFrame (except the time variable) against the specified time variable. The PSI is
    calculated using Jeffrey divergence.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The input DataFrame or LazyFrame containing the variables to analyze.
    t : str
        The name of the time variable column.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the Population Stability Index for each variable.
        The result has two columns:
        - 'var': The name of the variable.
        - 'psi': The calculated Population Stability Index.

    Notes
    -----
    This function uses the _multi_jeffrey_divergence function to calculate the
    divergence for each variable. The resulting values are interpreted as Population
    Stability Indices.
    PSI is used to measure the stability of a variable's distribution over time.
    """
    df_iv = _multi_jeffrey_divergence(df, t).rename({"val": "psi"})
    return df_iv
