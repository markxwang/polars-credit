import polars as pl
from sklearn.base import BaseEstimator

import polarscore as ps
from polarscore.base import PolarSelectorMixin
from polarscore.woe import calculate_iv


class NullRatioThreshold(PolarSelectorMixin, BaseEstimator):
    """
    A transformer that removes columns with a high proportion of null values.

    This class implements a feature selection strategy based on the ratio of null
    values in each column. Columns with a null ratio equal to or higher than the
    specified threshold are removed during the transformation phase.

    Parameters
    ----------
    threshold : float, optional (default=0.95)
        The threshold for the null ratio. Columns with a null ratio equal to or
        higher than this value will be removed.

    Attributes
    ----------
    cols_to_drop_ : list
        A list of column names that have been identified for removal during the
        fit phase.

    Methods
    -------
    fit(X, y=None)
        Identify the columns to be dropped based on their null ratio.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples
    --------
    >>> import polars as pl
    >>> from polarscore.feature_selection import NullRatioThreshold
    >>> df = pl.DataFrame(
    ...     {
    ...         "A": [1, None, 3, None, 5],
    ...         "B": [None, None, None, None, 1],
    ...         "C": [1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> selector = NullRatioThreshold(threshold=0.6)
    >>> selector.fit(df)
    >>> df_transformed = selector.transform(df)
    >>> print(df_transformed.columns)
    ['A', 'C']

    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def get_cols_to_drop(self):
        """Get the columns to drop."""
        return self.cols_to_drop_

    def fit(self, X: pl.DataFrame, y=None):
        """Fit the null ratio threshold."""
        X_null_ratio_above_tr = (X.null_count() / X.height) >= self.threshold

        self.cols_to_drop_ = [col.name for col in X_null_ratio_above_tr if col.item()]

        return self


class IdenticalRatioThreshold(PolarSelectorMixin, BaseEstimator):
    """
    A feature selector that removes columns with a high ratio of identical values.

    This selector computes the ratio of the most frequent value (mode) for each column
    and removes columns where this ratio exceeds a specified threshold.

    Parameters
    ----------
    threshold : float, optional (default=0.95)
        The threshold for the identical value ratio. Columns with a ratio equal to or
        higher than this value will be removed.
    ignore_nulls : bool, optional (default=True)
        If True, null values are ignored when calculating the ratio. If False, null
        values are included in the ratio calculation.

    Attributes
    ----------
    cols_to_drop_ : list
        A list of column names that have been identified for removal during the
        fit phase.

    Methods
    -------
    fit(X, y=None)
        Identify the columns to be dropped based on their identical value ratio.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples
    --------
    >>> import polars as pl
    >>> from polarscore.feature_selection import IdenticalRatioThreshold
    >>> df = pl.DataFrame(
    ...     {"A": [1, 1, 1, 2, 1], "B": [1, 1, 1, 1, 1], "C": [1, 2, 3, 4, 5]}
    ... )
    >>> selector = IdenticalRatioThreshold(threshold=0.8)
    >>> selector.fit(df)
    >>> df_transformed = selector.transform(df)
    >>> print(df_transformed.columns)
    ['A', 'C']

    """

    def __init__(self, threshold: float = 0.95, *, ignore_nulls: bool = True):
        self.threshold = threshold
        self.ignore_nulls = ignore_nulls

    def get_cols_to_drop(self):
        """Get the columns to drop."""
        return self.cols_to_drop_

    def fit(self, X: pl.DataFrame, y=None):
        """Fit the identical ratio threshold."""
        expr_mode = pl.all().drop_nulls().mode().first()

        if self.ignore_nulls:
            expr = pl.all().eq(expr_mode)
        else:
            expr = pl.all().eq_missing(expr_mode)

        X_mode_ratio_above_tr = X.select(expr.mean() >= self.threshold)

        self.cols_to_drop_ = [col.name for col in X_mode_ratio_above_tr if col.item()]

        return self


class IVThreshold(PolarSelectorMixin, BaseEstimator):
    """
    A feature selector that removes features based on their Information Value (IV).

    This class implements a feature selection strategy that calculates the Information
    Value for each feature with respect to a target variable and removes features with
    IV below a specified threshold.

    Parameters
    ----------
    threshold : float, optional (default=0.02)
        The threshold for Information Value. Features with IV less than or equal to
        this threshold will be removed.

    Attributes
    ----------
    cols_to_drop_ : list
        A list of column names identified for removal during the fit phase.

    Methods
    -------
    fit(X, y)
        Calculate the Information Value for each feature and identify columns to be
        dropped.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples
    --------
    >>> import polars as pl
    >>> from polarscore.feature_selection import IVThreshold
    >>> X = pl.DataFrame(
    ...     {"A": [1, 2, 1, 2, 1], "B": [1, 1, 1, 1, 1], "C": [1, 2, 3, 4, 5]}
    ... )
    >>> y = pl.Series([0, 1, 0, 1, 0])
    >>> selector = IVThreshold(threshold=0.1)
    >>> selector.fit(X, y)
    >>> X_transformed = selector.transform(X)
    >>> print(X_transformed.columns)
    ['A', 'C']

    """

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def get_cols_to_drop(self):
        """Get the columns to drop."""
        return self.cols_to_drop_

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """Fit the IV threshold."""
        df_iv_filter = (
            X.with_columns(y)
            .select(ps.iv(x=pl.exclude(y.name).cast(pl.String), y=pl.col(y.name)))
            .unpivot(variable_name="var", value_name="iv")
            .filter(pl.col("iv") <= self.threshold)
        )
        df_iv_filter = calculate_iv(X, y).filter(pl.col("iv") <= self.threshold)
        self.cols_to_drop_ = df_iv_filter["var"].to_list()

        return self
