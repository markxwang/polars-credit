from pathlib import Path

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from polarscore import base, bin, feature_selection, impute, woe

LIB = Path(__file__).parent


def cal_iv(x: IntoExpr, y: IntoExpr) -> pl.Expr:  # noqa: D103
    output = register_plugin_function(
        args=[x, y],
        plugin_path=LIB,
        function_name="pl_iv",
        is_elementwise=False,
        changes_length=True,
        returns_scalar=True,
    )

    return output


def cal_woe(x: IntoExpr, y: IntoExpr) -> pl.Expr:  # noqa: D103
    output = register_plugin_function(
        args=[x, y],
        plugin_path=LIB,
        function_name="pl_woe",
        is_elementwise=False,
        changes_length=True,
        returns_scalar=False,
    )

    return output


__all__ = [
    "cal_iv",
    "cal_woe",
    "woe",
    "feature_selection",
    "bin",
    "base",
    "impute",
]
