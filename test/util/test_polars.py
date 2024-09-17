import polars as pl
import pytest
from polars_credit.util.polars import _get_cols


@pytest.mark.parametrize(
    ("input_data", "expected_cols"),
    [
        (pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["a", "b"]),
        (
            pl.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"], "z": [True, False]}),
            ["x", "y", "z"],
        ),
        (pl.LazyFrame({"c": [1, 2, 3], "d": [4, 5, 6]}), ["c", "d"]),
        (
            pl.LazyFrame({"p": [1.0, 2.0], "q": ["a", "b"], "r": [True, False]}),
            ["p", "q", "r"],
        ),
        (pl.DataFrame(), []),
        (pl.LazyFrame(), []),
    ],
)
def test_get_cols_valid_input(input_data, expected_cols):
    assert _get_cols(input_data) == expected_cols


def test_get_cols_invalid_input():
    with pytest.raises(
        TypeError, match="Input must be a Polars DataFrame or LazyFrame"
    ):
        _get_cols([1, 2, 3])
