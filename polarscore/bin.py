import polars as pl
import polars.selectors as cs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def get_qcut_breaks_expr(col: str, q: int, allow_duplicates: bool = True):
    """
    Generate an expression to compute quantile cut breakpoints for a column.

    This function creates a Polars expression that calculates quantile breakpoints
    for the specified column, removing infinite values and returning a unique list
    of breakpoints.

    Parameters:
    -----------
    col : str
        The name of the column to compute breakpoints for.
    q : int
        The number of quantiles to compute.
    allow_duplicates : bool, optional
        Whether to allow duplicate breakpoints. Default is True.

    Returns:
    --------
    pl.Expr
        A Polars expression that, when evaluated, returns a list of unique
        breakpoints for the specified column.
    """

    expr = (
        pl.col(col)
        .qcut(q, include_breaks=True, allow_duplicates=allow_duplicates)
        .struct.field("breakpoint")
        .unique()
    )

    expr_rm_inf = (
        pl.when(~expr.is_infinite()).then(expr).drop_nulls().implode().alias(col)
    )

    return expr_rm_inf


class QuantileBinner(BaseEstimator, TransformerMixin):
    """
    A transformer that bins numeric columns into quantiles.

    This class implements a quantile-based binning strategy for numeric columns
    in a Polars DataFrame. It calculates breakpoints based on quantiles during
    the fit phase and uses these breakpoints to transform the data into bins.

    Parameters:
    -----------
    q : int
        The number of quantiles to use for binning.
    allow_duplicates : bool, optional
        Whether to allow duplicate breakpoints. Default is True.

    Attributes:
    -----------
    breakpoints_ : dict
        A dictionary containing the breakpoints for each numeric column,
        calculated during the fit phase.

    Methods:
    --------
    fit(X, y=None)
        Compute the quantile breakpoints on the input DataFrame X.
    transform(X)
        Bin the values in X according to the computed breakpoints.

    Examples:
    ---------
    >>> import polars as pl
    >>> from polarscore.bin import QuantileBinner
    >>> df = pl.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
    >>> binner = QuantileBinner(q=3)
    >>> binner.fit(df)
    >>> binned_df = binner.transform(df)
    """

    def __init__(self, q: int, allow_duplicates: bool = True):
        self.q = q
        self.allow_duplicates = allow_duplicates

    def fit(self, X: pl.DataFrame, y=None):
        numeric_columns = cs.expand_selector(X, cs.numeric())

        if not numeric_columns:
            raise ValueError("Input DataFrame contains no numeric columns")

        self.breakpoints_ = X.select(
            get_qcut_breaks_expr(x, q=self.q, allow_duplicates=self.allow_duplicates)
            for x in numeric_columns
        ).row(0, named=True)

        return self

    def transform(self, X: pl.DataFrame):
        check_is_fitted(self)

        X_cut = X.with_columns(
            pl.col(col).cut(self.breakpoints_[col])
            for col in X.columns
            if col in self.breakpoints_
        )

        return X_cut
