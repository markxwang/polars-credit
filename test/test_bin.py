import polars as pl
import pytest
from polarscore.bin import get_qcut_breaks_expr


@pytest.mark.parametrize(
    ("data", "q", "allow_duplicates", "expected_len"),
    [
        (range(100), 4, True, 3),
        ([1.0, 2.0, float("inf")], 2, True, 1),
        ([1, 1, 2, 2, 3, 3], 3, True, 2),
        ([1, 2, None, 4, 5], 2, True, 1),
        ([], 4, True, 0),
        ([1, 1, 1, 1], 4, True, 1),
    ],
)
def test_get_qcut_breaks_expr_parametrized(data, q, allow_duplicates, expected_len):
    df = pl.DataFrame({"col": data})

    result = df.select(
        get_qcut_breaks_expr("col", q=q, allow_duplicates=allow_duplicates)
    )

    assert len(result["col"][0]) == expected_len
