from abc import abstractmethod
from abc import ABCMeta
import polars as pl
from sklearn.base import TransformerMixin


class PolarSelectorMixin(TransformerMixin, metaclass=ABCMeta):
    @abstractmethod
    def get_cols_to_drop(self):
        pass

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        cols_drop = self.get_cols_to_drop()

        return X.drop(cols_drop)
