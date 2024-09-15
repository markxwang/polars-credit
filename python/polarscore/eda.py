import polars as pl
from polars_credit import cal_iv


def _eda_long_format(df, operation, *args, **kwargs):
    _eda_expr = getattr(pl.all().eda, operation)(*args, **kwargs)
    return df.select(_eda_expr).unpivot(variable_name="var", value_name=operation)


@pl.api.register_expr_namespace("eda")
class EdaExpr:
    """A class for exploratory data analysis."""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def null_count(self) -> pl.Expr:
        """Return the count of null values in the expression."""
        return self._expr.null_count()

    def null_ratio(self) -> pl.Expr:
        """Return the ratio of null values in the expression."""
        return self._expr.null_count() / self._expr.len()

    def n_unique(self) -> pl.Expr:
        """Return the number of unique values in the expression."""
        return self._expr.n_unique()

    def identical_ratio(self, *, ignore_nulls: bool = True) -> pl.Expr:
        """Return the ratio of identical values in the expression."""
        expr_mode = self._expr.drop_nulls().mode().first()

        if ignore_nulls:
            expr = self._expr.eq(expr_mode)
        else:
            expr = self._expr.eq_missing(expr_mode)

        return expr.mean()

    def iv(self, y: str) -> pl.Expr:
        """Return the information value for the expression."""
        return cal_iv(x=self._expr.exclude(y), y=y)


@pl.api.register_dataframe_namespace("eda")
class EdaFrame:
    """A class for exploratory data analysis on DataFrames."""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def null_count(self) -> pl.DataFrame:
        """Return a DataFrame with the count of null values for each column."""
        return _eda_long_format(self._df, "null_count")

    def null_ratio(self) -> pl.DataFrame:
        """Return a DataFrame with the ratio of null values for each column."""
        return _eda_long_format(self._df, "null_ratio")

    def identical_ratio(self) -> pl.DataFrame:
        """Return a DataFrame with the ratio of identical values for each column."""
        return _eda_long_format(self._df, "identical_ratio")

    def n_unique(self) -> pl.DataFrame:
        """Return a DataFrame with the number of unique values for each column."""
        return _eda_long_format(self._df, "n_unique")

    def iv(self, y: str) -> pl.DataFrame:
        """Return a DataFrame with the information value for each column."""
        return _eda_long_format(self._df, "iv", y=y)
