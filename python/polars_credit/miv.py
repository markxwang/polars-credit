import polars as pl


def cal_miv(df: pl.DataFrame, y: str, y_pred: str, x: str) -> pl.DataFrame:
    """Calculate MIV."""
    df_miv = (
        df.group_by(x)
        .agg(
            (1 - pl.col(y)).sum().alias("good_actual"),
            pl.col(y).sum().alias("bad_actual"),
            (1 - pl.col(y_pred)).sum().alias("good_pred"),
            pl.col(y_pred).sum().alias("bad_pred"),
        )
        .with_columns(
            pl.col("good_actual", "bad_actual")
            / pl.col("good_actual", "bad_actual").sum(),
            pl.col("good_pred", "bad_pred") / pl.col("good_pred", "bad_pred").sum(),
        )
        .with_columns(
            (pl.col("bad_actual") / pl.col("good_actual")).log().alias("woe_actual"),
            (pl.col("bad_pred") / pl.col("good_pred")).log().alias("woe_pred"),
        )
        .with_columns(
            miv=(pl.col("woe_actual") - pl.col("woe_pred"))
            * (pl.col("bad_actual") - pl.col("good_actual"))
        )
    )

    return df_miv
