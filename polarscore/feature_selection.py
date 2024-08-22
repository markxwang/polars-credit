import polars as pl
from .base import PolarSelectorMixin
from sklearn.base import BaseEstimator


class NullRatioThreshold(PolarSelectorMixin, BaseEstimator):
    """
    A transformer that removes columns with a high proportion of null values.

    This class implements a feature selection strategy based on the ratio of null
    values in each column. Columns with a null ratio equal to or higher than the
    specified threshold are removed during the transformation phase.

    Parameters:
    -----------
    threshold : float, optional (default=0.95)
        The threshold for the null ratio. Columns with a null ratio equal to or
        higher than this value will be removed.

    Attributes:
    -----------
    cols_to_drop_ : list
        A list of column names that have been identified for removal during the
        fit phase.

    Methods:
    --------
    fit(X, y=None)
        Identify the columns to be dropped based on their null ratio.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples:
    ---------
    >>> import polars as pl
    >>> from polarscore.feature_selection import NullRatioThreshold
    >>> df = pl.DataFrame({
    ...     'A': [1, None, 3, None, 5],
    ...     'B': [None, None, None, None, 1],
    ...     'C': [1, 2, 3, 4, 5]
    ... })
    >>> selector = NullRatioThreshold(threshold=0.6)
    >>> selector.fit(df)
    >>> df_transformed = selector.transform(df)
    >>> print(df_transformed.columns)
    ['A', 'C']
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def get_cols_to_drop(self):
        return self.cols_to_drop_

    def fit(self, X: pl.DataFrame, y=None):
        X_null_ratio_above_tr = (X.null_count() / X.height) >= self.threshold

        self.cols_to_drop_ = [col.name for col in X_null_ratio_above_tr if col.item()]

        return self


class ModeRatioThreshold(PolarSelectorMixin, BaseEstimator):
    """
    A transformer that removes columns based on th e ratio of the mode value.

    This class implements a feature selection strategy that removes columns
    where the ratio of the most frequent value (mode) exceeds a specified threshold.

    Parameters:
    -----------
    threshold : float, optional (default=0.95)
        The threshold for the mode ratio. Columns with a mode ratio equal to or
        higher than this value will be removed.

    Attributes:
    -----------
    cols_to_drop_ : list
        A list of column names that have been identified for removal during the
        fit phase.

    Methods:
    --------
    fit(X, y=None)
        Identify the columns to be dropped based on their mode ratio.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples:
    ---------
    >>> import polars as pl
    >>> from polarscore.feature_selection import ModeRatioThreshold
    >>> df = pl.DataFrame({
    ...     'A': [1, 1, 1, 2, 3],
    ...     'B': [1, 1, 1, 1, 1],
    ...     'C': [1, 2, 3, 4, 5]
    ... })
    >>> selector = ModeRatioThreshold(threshold=0.8)
    >>> selector.fit(df)
    >>> df_transformed = selector.transform(df)
    >>> print(df_transformed.columns)
    ['A', 'C']
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def get_cols_to_drop(self):
        return self.cols_to_drop_

    def fit(self, X: pl.DataFrame, y=None):
        X_mode_ratio_above_tr = X.select(
            pl.all().eq(pl.all().mode().first()).mean() > self.threshold
        )

        self.cols_to_drop_ = [col.name for col in X_mode_ratio_above_tr if col.item()]

        return self
