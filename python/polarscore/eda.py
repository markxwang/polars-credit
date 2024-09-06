import polars as pl


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
            expr = pl.all().eq(expr_mode)
        else:
            expr = pl.all().eq_missing(expr_mode)

        return expr.mean()


@pl.api.register_dataframe_namespace("eda")
class EdaFrame:
    """A class for exploratory data analysis on DataFrames."""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def null_count(self) -> pl.DataFrame:
        """Return a DataFrame with the count of null values for each column."""
        return self._df.select(pl.all().eda.null_count()).unpivot(
            variable_name="var", value_name="null_count"
        )

    def null_ratio(self) -> pl.DataFrame:
        """Return a DataFrame with the ratio of null values for each column."""
        return self._df.select(pl.all().eda.null_ratio()).unpivot(
            variable_name="var", value_name="null_ratio"
        )
