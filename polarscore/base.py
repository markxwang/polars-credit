from abc import ABCMeta, abstractmethod

import polars as pl
from sklearn.base import TransformerMixin


class PolarSelectorMixin(TransformerMixin, metaclass=ABCMeta):
    """
    Base class for feature selectors in Polarscore.

    This abstract base class defines the interface for feature selectors
    that work with Polars DataFrames. It extends scikit-learn's TransformerMixin
    to provide a consistent API for feature selection operations.

    Attributes
    ----------
        None

    Methods
    -------
        get_cols_to_drop: Abstract method to be implemented by subclasses.
            Should return a list of column names to be dropped.
        transform: Applies the feature selection by dropping columns.

    Notes
    -----
        Subclasses must implement the `get_cols_to_drop` method to define
        the specific selection criteria.

    """

    @abstractmethod
    def get_cols_to_drop(self):
        """
        Get the list of columns to be dropped.

        This method should be implemented by subclasses to define the specific
        criteria for selecting columns to be dropped during the feature selection
        process.

        Returns
        -------
        list
            A list of column names that should be dropped from the DataFrame.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Notes
        -----
        This is an abstract method that must be overridden by subclasses.
        The implementation should return a list of column names based on
        the feature selection criteria defined in the subclass.
        """

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Apply the feature selection by dropping columns.

        This method removes the columns identified by the `get_cols_to_drop` method
        from the input DataFrame.

        Parameters
        ----------
        X : pl.DataFrame
            The input DataFrame from which columns will be dropped.

        Returns
        -------
        pl.DataFrame
            A new DataFrame with the selected columns removed.

        Notes
        -----
        This method relies on the `get_cols_to_drop` method to determine which
        columns should be removed. Ensure that the `get_cols_to_drop` method
        is properly implemented in subclasses.
        """
        cols_drop = self.get_cols_to_drop()

        return X.drop(cols_drop)
