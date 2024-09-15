import polars as pl
from polars._typing import IntoExpr
from polars._utils.parse import parse_into_expression


def _parse_expr(expr: IntoExpr, *args, **kwargs) -> pl.Expr:
    return pl.Expr._from_pyexpr(parse_into_expression(expr), *args, **kwargs)
