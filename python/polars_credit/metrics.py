import polars as pl


def roc_auc_score(col_true: str, col_pred: str):
    """
    Calculate the ROC AUC score using Polars expressions.

    This function computes the Receiver Operating Characteristic Area Under the Curve
    (ROC AUC) score for binary classification problems using Polars expressions.

    Args:
        col_true (str): Name of the column containing true binary labels.
        col_pred (str): Name of the column containing predicted probabilities.

    Returns
    -------
        pl.Expr: A Polars expression that computes the ROC AUC score.
    """
    sort_true = pl.col(col_true).sort_by(pl.col(col_pred), descending=True)

    cum_tp = sort_true.eq(1).cum_sum()
    cum_fp = sort_true.eq(0).cum_sum()

    pos = pl.col(col_true).eq(1).sum()
    neg = pl.col(col_true).eq(0).sum()

    tpr = cum_tp / pos
    fpr = cum_fp / neg

    roc_auc = ((fpr - fpr.shift(1)) * (tpr + tpr.shift(1)) / 2).sum().alias("roc_auc")

    return roc_auc
