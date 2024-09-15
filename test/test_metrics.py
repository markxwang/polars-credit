from math import isclose

import polars as pl
import pytest
from polars_credit.metrics import roc_auc_score

df1 = pl.DataFrame(
    {
        "true": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        "pred": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.5, 0.3, 0.6, 0.15],
    }
)


@pytest.mark.parametrize(("input", "output"), [(df1, 0.8)])
def test_roc_auc_score(input, output):
    score = input.select(roc_auc_score("true", "pred"))[0, 0]
    assert isclose(score, output)
