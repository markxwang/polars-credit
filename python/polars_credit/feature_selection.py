import polars as pl
from sklearn.base import BaseEstimator

from polars_credit.base import PolarSelectorMixin
from polars_credit.util import cal_iv, cal_psi


class NullRatioThreshold(PolarSelectorMixin, BaseEstimator):
    """
    A feature selector that removes columns with a high ratio of null values.

    This selector computes the ratio of null values for each column and removes
    columns where this ratio exceeds a specified threshold.

    Parameters
    ----------
    threshold : float, optional (default=0.95)
        The threshold for the null value ratio. Columns with a ratio equal to or
        higher than this value will be removed.

    Attributes
    ----------
    cols_to_drop_ : list
        A list of column names that have been identified for removal during the
        fit phase.

    Methods
    -------
    fit(X, y=None)
        Identify the columns to be dropped based on their null value ratio.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_credit.feature_selection import NullRatioThreshold
    >>> df = pl.DataFrame(
    ...     {
    ...         "A": [1, None, 3, None, 5],
    ...         "B": [None, None, None, None, None],
    ...         "C": [1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> selector = NullRatioThreshold(threshold=0.8)
    >>> selector.fit(df)
    >>> df_transformed = selector.transform(df)
    >>> print(df_transformed.columns)
    ['A', 'C']

    Notes
    -----
    This selector is particularly useful for removing columns with a high proportion
    of missing values, which might not be informative or could potentially bias
    the analysis or model training.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pl.DataFrame, y=None):
        """Fit the null ratio threshold."""
        X_null_ratio_above_tr = (X.null_count() / X.height) >= self.threshold

        self.cols_to_drop_ = [col.name for col in X_null_ratio_above_tr if col.item()]

        return self


class IdenticalRatioThreshold(PolarSelectorMixin, BaseEstimator):
    """
    A feature selector that removes columns based on the ratio of identical values.

    This class implements a feature selection strategy that calculates the ratio of
    identical values for each column and removes columns where this ratio exceeds
    a specified threshold.

    Parameters
    ----------
    threshold : float, optional (default=0.95)
        The threshold for the ratio of identical values. Columns with a ratio
        greater than or equal to this threshold will be removed.
    ignore_nulls : bool, optional (default=True)
        If True, null values are ignored when calculating the ratio of identical values.
        If False, null values are treated as a distinct value.

    Attributes
    ----------
    cols_to_drop_ : list
        A list of column names that have been identified for removal during the
        fit phase.

    Methods
    -------
    fit(X, y=None)
        Identify the columns to be dropped based on their ratio of identical values.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_credit.feature_selection import IdenticalRatioThreshold
    >>> df = pl.DataFrame(
    ...     {"A": [1, 1, 1, 1, 2], "B": [1, 1, 1, 1, 1], "C": [1, 2, 3, 4, 5]}
    ... )
    >>> selector = IdenticalRatioThreshold(threshold=0.8)
    >>> selector.fit(df)
    >>> df_transformed = selector.transform(df)
    >>> print(df_transformed.columns)
    ['A', 'C']

    Notes
    -----
    This selector is particularly useful for removing columns with a high proportion
    of identical values, which might not provide much information or could potentially
    introduce bias in the analysis or model training.
    """

    def __init__(self, threshold: float = 0.95, *, ignore_nulls: bool = True):
        self.threshold = threshold
        self.ignore_nulls = ignore_nulls

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
    >>> from polars_credit.feature_selection import IVThreshold
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

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """Fit the IV threshold."""
        self.iv_ = X.with_columns(y).pipe(cal_iv, y.name)

        df_iv_filter = self.iv_.filter(pl.col("iv") <= self.threshold)
        self.cols_to_drop_ = df_iv_filter["var"].to_list()

        return self


class PSIThreshold(PolarSelectorMixin, BaseEstimator):
    """
    Population Stability Index (PSI) Threshold Selector.

    This class implements a feature selector that removes features based on their
    Population Stability Index (PSI) values. Features with PSI values less than or
    equal to a specified threshold are removed.

    Parameters
    ----------
    threshold : float, optional (default=0.1)
        The threshold for Population Stability Index. Features with PSI less than or
        equal to this threshold will be removed.

    Attributes
    ----------
    cols_to_drop_ : list
        A list of column names identified for removal during the fit phase.
    psi_ : pl.DataFrame
        A DataFrame containing the PSI values for each feature.

    Methods
    -------
    fit(X, y, t)
        Calculate the Population Stability Index for each feature and identify columns
        to be dropped.
    transform(X)
        Remove the identified columns from the input DataFrame.
    get_cols_to_drop()
        Return the list of columns identified for removal.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_credit.feature_selection import PSIThreshold
    >>> X = pl.DataFrame(
    ...     {"A": [1, 2, 1, 2, 1], "B": [1, 1, 1, 1, 1], "C": [1, 2, 3, 4, 5]}
    ... )
    >>> t = pl.Series([1, 1, 2, 2, 2])  # Time periods
    >>> selector = PSIThreshold(threshold=0.05)
    >>> selector.fit(X, t=t)
    >>> X_transformed = selector.transform(X)
    >>> print(X_transformed.columns)
    ['A', 'C']
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def fit(self, X: pl.DataFrame, y: pl.Series = None, t: pl.Series = None):
        """Fit the PSI threshold."""
        if t is None:
            msg = "t must be provided"
            raise ValueError(msg)

        self.psi_ = X.with_columns(t).pipe(cal_psi, t.name)

        df_psi_filter = self.psi_.filter(pl.col("psi") <= self.threshold)
        self.cols_to_drop_ = df_psi_filter["var"].to_list()
        return self
