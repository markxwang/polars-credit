import polars as pl


def roc_curve(col_true: str, col_pred: str):
    """Calculate the ROC curve using Polars expressions."""
    sort_true = pl.col(col_true).sort_by(pl.col(col_pred), descending=True)

    cum_tp = sort_true.eq(1).cum_sum()
    cum_fp = sort_true.eq(0).cum_sum()

    pos = pl.col(col_true).eq(1).sum()
    neg = pl.col(col_true).eq(0).sum()

    tpr = cum_tp / pos
    fpr = cum_fp / neg

    return tpr, fpr


def roc_auc_score(col_true: str, col_pred: str):
    """Calculate the ROC AUC score using Polars expressions."""
    tpr, fpr = roc_curve(col_true, col_pred)
    roc_auc = ((fpr - fpr.shift(1)) * (tpr + tpr.shift(1)) / 2).sum().alias("roc_auc")

    return roc_auc


def ks_score(col_true: str, col_pred: str):
    """Calculate the Kolmogorov-Smirnov (KS) score using Polars expressions."""
    tpr, fpr = roc_curve(col_true, col_pred)
    ks_score = (tpr - fpr).abs().max().alias("ks_score")
    return ks_score
