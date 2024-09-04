import polars as pl
import pytest
from polarscore.feature_selection import NullRatioThreshold


@pytest.mark.parametrize(
    ("threshold", "expected_columns"),
    [
        (0.5, ["A", "B", "C"]),
        (0.3, ["B", "C"]),
    ],
)
def test_null_ratio_threshold(threshold, expected_columns):
    # Create a sample DataFrame with null values
    df = pl.DataFrame(
        {
            "A": [1, None, 3, None, 5],
            "B": [1, 2, None, 4, 5],
            "C": [1, 2, 3, 4, 5],
            "D": [None, None, None, None, None],
        }
    )

    # Initialize NullRatioThreshold
    nrt = NullRatioThreshold(threshold=threshold)

    # Fit and transform the DataFrame
    result = nrt.fit_transform(df)

    # Check that the correct columns remain
    assert set(result.columns) == set(expected_columns)
