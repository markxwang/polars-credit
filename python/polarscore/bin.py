import polars as pl
import polars.selectors as cs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def get_qcut_breaks_expr(col: str, q: int, *, allow_duplicates: bool = True):
    """
    Generate an expression to compute quantile cut breakpoints for a column.

    This function creates a Polars expression that calculates quantile breakpoints
    for the specified column, removing infinite values and returning a unique list
    of breakpoints.

    Parameters
    ----------
    col : str
        The name of the column to compute breakpoints for.
    q : int
        The number of quantiles to compute.
    allow_duplicates : bool, optional
        Whether to allow duplicate breakpoints. Default is True.

    Returns
    -------
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


class BinnerMixin(BaseEstimator, TransformerMixin):
    """
    Base class for binning transformers in Polarscore.

    This abstract base class defines the interface for binning transformers
    that work with Polars DataFrames. It extends scikit-learn's BaseEstimator
    and TransformerMixin to provide a consistent API for binning operations.

    Attributes
    ----------
    breakpoints_ : dict
        A dictionary containing the breakpoints for each numeric column,
        calculated during the fit phase.

    Methods
    -------
    fit(X, y=None)
        Abstract method to be implemented by subclasses.
        Should compute the breakpoints for binning.
    transform(X)
        Bin the values in X according to the computed breakpoints.

    Notes
    -----
    Subclasses must implement the `fit` method to define
    the specific binning criteria and compute breakpoints.
    """

    def transform(self, X: pl.DataFrame):
        """
        Transform the input DataFrame by binning numeric columns.

        This method applies the binning transformation to the input DataFrame
        using the breakpoints computed during the fit phase. It bins the values
        in each numeric column according to the corresponding breakpoints.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pl.DataFrame
            A new DataFrame with the numeric columns binned according to
            the computed breakpoints.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.

        Notes
        -----
        This method only transforms columns that were present during the fit
        phase and have computed breakpoints. Other columns remain unchanged.

        """
        check_is_fitted(self)

        X_cut = X.with_columns(
            pl.col(col).cut(self.breakpoints_[col])
            for col in X.columns
            if col in self.breakpoints_
        )

        return X_cut


class QuantileBinner(BinnerMixin):
    """
    A transformer that bins numeric columns into quantiles.

    This class implements a quantile-based binning strategy for numeric columns
    in a Polars DataFrame. It calculates breakpoints based on quantiles during
    the fit phase and uses these breakpoints to transform the data into bins.

    Parameters
    ----------
    q : int
        The number of quantiles to use for binning.
    allow_duplicates : bool, optional
        Whether to allow duplicate breakpoints. Default is True.

    Attributes
    ----------
    breakpoints_ : dict
        A dictionary containing the breakpoints for each numeric column,
        calculated during the fit phase.

    Methods
    -------
    fit(X, y=None)
        Compute the quantile breakpoints on the input DataFrame X.
    transform(X)
        Bin the values in X according to the computed breakpoints.

    Examples
    --------
    >>> import polars as pl
    >>> from polarscore.bin import QuantileBinner
    >>> df = pl.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    >>> binner = QuantileBinner(q=3)
    >>> binner.fit(df)
    >>> binned_df = binner.transform(df)

    """

    def __init__(self, q: int, *, allow_duplicates: bool = True):
        self.q = q
        self.allow_duplicates = allow_duplicates

    def fit(self, X: pl.DataFrame, y=None):
        """
        Compute the quantile breakpoints for each numeric column in the input DataFrame.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame containing numeric columns to be binned.
        y : None
            Ignored. Kept for compatibility with scikit-learn API.

        Returns
        -------
        self : QuantileBinner
            Returns the instance itself.
        """
        numeric_columns = cs.expand_selector(X, cs.numeric())

        if not numeric_columns:
            msg = "Input DataFrame contains no numeric columns"
            raise ValueError(msg)

        self.breakpoints_ = X.select(
            get_qcut_breaks_expr(x, q=self.q, allow_duplicates=self.allow_duplicates)
            for x in numeric_columns
        ).row(0, named=True)

        return self


class CustomBinner(BinnerMixin):
    """
    A binner that uses custom-defined breakpoints for binning.

    This class allows users to specify custom breakpoints for each column
    to be binned, providing more control over the binning process.

    Parameters
    ----------
    breakpoints : dict
        A dictionary where keys are column names and values are lists of
        breakpoints for that column. Each list should contain the lower
        and upper bounds of the bins, in ascending order.

    Attributes
    ----------
    breakpoints_ : dict
        The validated and stored breakpoints used for binning.

    Methods
    -------
    fit(X, y=None)
        Validate and store the custom breakpoints.
    transform(X)
        Bin the values in X according to the stored breakpoints.

    Examples
    --------
    >>> import polars as pl
    >>> from polarscore.bin import CustomBinner
    >>> df = pl.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    >>> breakpoints = {"A": [0, 2, 4, 6], "B": [0, 25, 50, 75]}
    >>> binner = CustomBinner(breakpoints)
    >>> binner.fit(df)
    >>> binned_df = binner.transform(df)

    """

    def __init__(self, breakpoints: dict):
        self.breakpoints = breakpoints

    def fit(self, X: pl.DataFrame, y=None):
        """
        Validate and store the custom breakpoints.

        This method checks if the provided breakpoints are valid for the input
        DataFrame and stores them for use in the transformation step.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame containing the features to be binned.
        y : None
            Ignored. Kept for compatibility with scikit-learn API.

        Returns
        -------
        self : CustomBinner
            Returns the instance itself.

        Raises
        ------
        ValueError
            If breakpoints are not provided for all columns in X,
            or if the breakpoints for any column are not in ascending order.

        Notes
        -----
        This method performs the following validations:
        1. Ensures breakpoints are provided for all columns in X.
        2. Checks that breakpoints for each column are in ascending order.
        3. Verifies that the number of breakpoints is appropriate for binning.

        The validated breakpoints are stored in the `breakpoints_` attribute.
        """
        # needs to add validation logic later
        self.breakpoints_ = self.breakpoints
        return self
