import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class FixedValueImputer(TransformerMixin, BaseEstimator):
    """
    Imputer that fills missing values with fixed values for specified columns.

    This transformer allows for imputation of missing values in a DataFrame
    using predefined fixed values for specified columns.

    Parameters
    ----------
    fill_value_dict : dict
        A dictionary where keys are column names and values are the fixed
        values to use for imputation in those columns.

    Attributes
    ----------
    fill_value_dict_ : dict
        The fitted dictionary of column names and their corresponding
        imputation values.

    Methods
    -------
    fit(X, y=None)
        Fit the imputer to the input DataFrame.
    transform(X)
        Impute missing values in the input DataFrame using the fixed values.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_credit.impute import FixedValueImputer
    >>> df = pl.DataFrame({"A": [1, None, 3], "B": ["x", None, "z"]})
    >>> imputer = FixedValueImputer({"A": 0, "B": "unknown"})
    >>> imputer.fit_transform(df)
    shape: (3, 2)
    ┌─────┬─────────┐
    │ A   ┆ B       │
    │ --- ┆ ---     │
    │ i64 ┆ str     │
    ╞═════╪═════════╡
    │ 1   ┆ x       │
    │ 0   ┆ unknown │
    │ 3   ┆ z       │
    └─────┴─────────┘

    Notes
    -----
    This imputer is particularly useful when you have domain knowledge about
    appropriate default values for specific columns in your dataset.
    """

    def __init__(self, fill_value_dict: dict):
        self.fill_value_dict = fill_value_dict

    def fit(self, X: pl.DataFrame, y=None):
        """
        Fit the imputer to the input DataFrame.

        This method validates that all columns specified in the fill_value_dict
        are present in the input DataFrame. It then stores the fill_value_dict
        as an attribute for use in the transform method.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame to fit the imputer on.
        y : None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's transformer interface.

        Returns
        -------
        self : FixedValueImputer
            Returns the instance itself.

        Raises
        ------
        ValueError
            If any column specified in fill_value_dict is not present in X.
        """
        if not set(self.fill_value_dict).issubset(X.columns):
            msg = "Some columns in fill_value_dict are not present in the input DataFrame."
            raise ValueError(msg)

        self.fill_value_dict_ = self.fill_value_dict
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input DataFrame by imputing missing values with fixed values.

        This method applies the fixed value imputation to the specified columns
        in the input DataFrame using the fill values provided during initialization.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            A new DataFrame with missing values imputed in the specified columns.

        Notes
        -----
        This method uses the `fill_value_dict_` attribute set during the fit method
        to determine which columns to impute and what values to use for imputation.
        """
        return X.with_columns(
            pl.col(col).fill_null(value) for col, value in self.fill_value_dict_.items()
        )
