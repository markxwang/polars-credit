from __future__ import annotations

from typing import TYPE_CHECKING

from polars_credit.util.expr import _parse_expr

if TYPE_CHECKING:
    import polars as pl


def roc_curve(true: str | pl.Expr, pred: str | pl.Expr):
    """Calculate the ROC curve using Polars expressions."""
    expr_true = _parse_expr(true)
    expr_pred = _parse_expr(pred)

    sort_true = expr_true.sort_by(expr_pred, descending=True)

    cum_tp = sort_true.eq(1).cum_sum()
    cum_fp = sort_true.eq(0).cum_sum()

    pos = expr_true.eq(1).sum()
    neg = expr_true.eq(0).sum()

    tpr = cum_tp / pos
    fpr = cum_fp / neg

    return tpr, fpr


def roc_auc_score(true: str | pl.Expr, pred: str | pl.Expr):
    """Calculate the ROC AUC score using Polars expressions."""
    expr_true = _parse_expr(true)
    expr_pred = _parse_expr(pred)

    tpr, fpr = roc_curve(expr_true, expr_pred)
    roc_auc = ((fpr - fpr.shift(1)) * (tpr + tpr.shift(1)) / 2).sum().alias("roc_auc")

    return roc_auc


def ks_score(true: str | pl.Expr, pred: str | pl.Expr):
    """Calculate the Kolmogorov-Smirnov (KS) score using Polars expressions."""
    expr_true = _parse_expr(true)
    expr_pred = _parse_expr(pred)

    tpr, fpr = roc_curve(expr_true, expr_pred)
    ks_score = (tpr - fpr).abs().max().alias("ks_score")
    return ks_score


def gini(true: str | pl.Expr, pred: str | pl.Expr):
    """Calculate the Gini coefficient using Polars expressions."""
    return 2 * roc_auc_score(true, pred) - 1
