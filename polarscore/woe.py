import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


def get_woe(df: pl.DataFrame, y: str, x: str) -> pl.DataFrame:
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
    def fit(self, X: pl.DataFrame, y: pl.Series):
        self.woe_maps = {}

        df = X.with_columns(y).lazy()

        ls_woe_lazy = [
            get_woe(df, y.name, x).select(pl.col(x), pl.col("woe")) for x in X.columns
        ]

        ls_woe = pl.collect_all(ls_woe_lazy)

        self.woe_maps = dict(zip(X.columns, ls_woe))
        return self

    def transform(self, X: pl.DataFrame):
        X_woe = X.with_columns(
            pl.col(x).replace_strict(self.woe_maps[x][x], self.woe_maps[x]["woe"])
            for x in X.columns
        )

        return X_woe
