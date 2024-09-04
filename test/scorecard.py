from math import log

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ScorecardTransformer(BaseEstimator, ClassifierMixin):
    """
    A transformer that converts classifier probabilities into credit scorecard points.

    This class wraps a classifier and transforms its output probabilities into
    credit scorecard points. The transformation is based on the Points to Double
    the Odds (PDO) and the base score and odds.

    Parameters
    ----------
    cls : object
        The classifier object to be wrapped. It should have fit, predict_proba,
        and predict methods.
    pdo : float, default=20
        Points to Double the Odds. The number of points that doubles the odds.
    rate : float, default=2
        The rate at which the odds double.
    base_score : float, default=600
        The base score, typically representing a neutral credit score.
    base_odds : float, default=50
        The base odds, representing the odds at the base score.

    Attributes
    ----------
    cls_fitted_ : object
        The fitted classifier.
    factor_ : float
        The scaling factor for the log-odds.
    offset_ : float
        The offset added to the scaled log-odds.

    Methods
    -------
    fit(X, y)
        Fit the model according to the given training data.
    predict_proba(X)
        Predict class probabilities for X.
    predict(X)
        Predict class labels for X.
    """

    def __init__(self, cls, pdo=20, rate=2, base_score=600, base_odds=50):
        self.cls = cls
        self.pdo = pdo
        self.rate = rate
        self.base_score = base_score
        self.base_odds = base_odds

    def fit(self, X, y):
        self.factor_ = self.pdo / log(self.rate)
        self.offset_ = self.base_score - self.factor_ * log(self.base_odds)

        self.cls_fitted_ = self.cls.fit(X, y)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        log_proba = self.cls_fitted_.predict_log_proba(X)
        scores = self.offset_ + self.factor_ * (log_proba[:, 1] - log_proba[:, 0])

        return scores

    def predict(self, X):
        check_is_fitted(self)

        return self.cls_fitted_.predict(X)
