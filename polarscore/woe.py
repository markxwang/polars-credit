import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


def get_woe(df: pl.DataFrame, y: str, x: str) -> pl.DataFrame:
    """
    Calculate the Weight of Evidence (WOE) for a binary target variable.

    This function computes the Weight of Evidence (WOE) for each category of a feature
    with respect to a binary target variable.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing both the feature and target variable.
    y : str
        The name of the binary target variable column (0 or 1).
    x : str
        The name of the feature column for which WOE is calculated.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:
        - The original feature column
        - 'good': Count of target=0 for each category
        - 'bad': Count of target=1 for each category
        - Normalized 'good' and 'bad' (as proportions)
        - 'woe': The calculated Weight of Evidence for each category

    Notes
    -----
    WOE is calculated as ln(% of bad / % of good) for each category of the feature.
    The resulting DataFrame is sorted by the feature values.

    This function uses lazy evaluation for efficiency and can handle large datasets.

    """
    df_woe = (
        df.group_by(x)
        .agg(
            pl.col(y).eq(0).sum().alias("good"),
            pl.col(y).eq(1).sum().alias("bad"),
        )
        .with_columns(pl.col("good", "bad") / pl.col("good", "bad").sum())
        .with_columns((pl.col("bad") / pl.col("good")).log().alias("woe"))
        .sort(x)
    )

    return df_woe


def calculate_iv(X: pl.DataFrame, y: pl.Series):
    """
    Calculate the Information Value (IV) for each feature in X with respect to y.

    This function computes the Information Value, a measure of a variable's
    predictive power in relation to a binary target variable.

    Parameters
    ----------
    X : pl.DataFrame
        The input DataFrame containing the features.
    y : pl.Series
        The target variable as a Polars Series. Should be binary (0 or 1).

    Returns
    -------
    pl.DataFrame
        A DataFrame with two columns:
        - 'var': The name of the variable (feature).
        - 'iv': The calculated Information Value for that variable.

    Notes
    -----
    The Information Value is calculated using the Weight of Evidence (WOE)
    transformation. It provides a measure of each feature's predictive power,
    with higher values indicating stronger predictive ability.

    The function uses lazy evaluation for efficiency and can handle large datasets.

    """
    df = X.with_columns(y).lazy()

    ls_iv = []

    for x in X.columns:
        iv = df.pipe(get_woe, y=y.name, x=x).select(
            var=pl.lit(x),
            iv=(pl.col("woe") * (pl.col("bad") - pl.col("good"))).sum(),
        )

        ls_iv.append(iv)

    df_iv = pl.concat(ls_iv).collect()
    return df_iv


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Weight of Evidence (WOE) encoding to features.

    This class implements WOE encoding, a technique that transforms categorical
    variables into continuous variables based on their relationship with a binary
    target variable. It's particularly useful in credit scoring and risk modeling.

    Attributes
    ----------
    woe_maps : dict
        A dictionary storing the WOE mappings for each feature. Keys are feature
        names, and values are DataFrames containing the original values and their
        corresponding WOE values.

    Methods
    -------
    fit(X, y)
        Compute the WOE mappings for each feature in X with respect to y.
    transform(X)
        Transform the input features using the computed WOE mappings.

    Examples
    --------
    >>> import polars as pl
    >>> from polarscore.woe import WOETransformer
    >>> X = pl.DataFrame({"A": ["a", "b", "a", "c"], "B": [1, 2, 1, 3]})
    >>> y = pl.Series([0, 1, 0, 1])
    >>> woe = WOETransformer()
    >>> woe.fit(X, y)
    >>> X_woe = woe.transform(X)

    Notes
    -----
    The WOE transformation is defined as:
    WOE = ln(%of non-events / %of events)

    This transformer uses lazy evaluation for efficiency and can handle large datasets.
    """

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """
        Compute the Weight of Evidence (WOE) mappings for each feature.

        This method calculates the WOE values for each unique value in every feature
        of the input DataFrame X, with respect to the binary target variable y.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame containing the features to be encoded.
        y : pl.Series
            The binary target variable.

        Returns
        -------
        self : WOETransformer
            Returns the instance itself.

        Notes
        -----
        This method populates the `woe_maps` attribute with WOE mappings for each
        feature.
        Each mapping is stored as a DataFrame containing the original feature values
        and their corresponding WOE values.

        The WOE calculation is performed using lazy evaluation for efficiency.
        """
        self.woe_maps = {}

        df = X.with_columns(y).lazy()

        ls_woe_lazy = [
            get_woe(df, y.name, x).select(pl.col(x), pl.col("woe")) for x in X.columns
        ]

        ls_woe = pl.collect_all(ls_woe_lazy)

        self.woe_maps = dict(zip(X.columns, ls_woe))
        return self

    def transform(self, X: pl.DataFrame):
        """
        Transform the input DataFrame using the computed WOE mappings.

        This method applies the Weight of Evidence (WOE) transformation to each feature
        in the input DataFrame, using the WOE mappings computed during the fit phase.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pl.DataFrame
            A new DataFrame with all features transformed to their WOE values.

        Notes
        -----
        This method replaces each value in the input DataFrame with its corresponding
        WOE value, as determined by the mappings in the `woe_maps` attribute.
        If a value is encountered that was not present during the fit phase,
        it will be replaced with a null value.

        The transformation is performed using Polars' efficient column operations.
        """
        X_woe = X.with_columns(
            pl.col(x).replace_strict(self.woe_maps[x][x], self.woe_maps[x]["woe"])
            for x in X.columns
        )

        return X_woe
