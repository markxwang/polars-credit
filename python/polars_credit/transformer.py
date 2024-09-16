from __future__ import annotations

from typing import Literal

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from polars_credit.utils import _parse_expr


class ParallelFeatureTransformer(TransformerMixin, BaseEstimator):
    """
    Transformer that applies a list of transformers in parallel.

    This transformer allows for applying a list of transformers in parallel
    to a Polars DataFrame. It is useful when you want to apply multiple transformations
    to different subsets of columns in your DataFrame.
    """

    def __init__(self, transformers, *, remainder=Literal["drop", "passthrough"]):
        self.transformers = transformers
        self.remainder = remainder

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None):
        """Fit the transformers."""
        self.transformers_fitted_ = []

        used_cols = set()
        for expr, tf in self.transformers:
            X_sub = X.select(_parse_expr(expr))
            fitted_transformer = tf.fit(X_sub, y)
            self.transformers_fitted_.append((expr, fitted_transformer))

            used_cols |= set(X_sub.columns)

        if self.remainder == "passthrough":
            self.remainder_cols_ = [col for col in X.columns if col not in used_cols]
        elif self.remainder == "drop":
            self.remainder_cols_ = []
        else:
            msg = "remainder must be 'drop' or 'passthrough'"
            raise ValueError(msg)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the data."""
        X_transformed = [
            tf.transform(X.select(_parse_expr(expr))) for expr, tf in self.transformers_
        ]

        X_transformed.append(X.select(self.remainder_cols_))

        X_transformed_concat = pl.concat(X_transformed, how="horizontal")
        return X_transformed_concat
