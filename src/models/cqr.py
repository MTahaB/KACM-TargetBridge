import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import HistGradientBoostingRegressor


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = scores.shape[0]
    if n == 0:
        raise ValueError("Cannot calibrate CQR intervals with no calibration samples.")
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


@dataclass
class CQR:
    """Conformalized Quantile Regression with HistGradientBoostingRegressor.

    Combines quantile regression with conformal calibration. Coverage should be
    evaluated empirically under the split used for a given experiment.
    """

    alpha: float = 0.1
    lq_: any = None
    uq_: any = None
    qhat_: float | None = None
    qhat_lo_: float | None = None
    qhat_hi_: float | None = None
    q_low: float = 0.05
    q_high: float = 0.95

    def fit(self, X_tr, y_tr, X_cal, y_cal):
        """Fit quantile regressors and calibrate on holdout set."""
        self.q_low = self.alpha / 2.0
        self.q_high = 1.0 - self.alpha / 2.0

        # Train quantile regressors
        self.lq_ = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.q_low,
            max_depth=6,
            learning_rate=0.06,
            random_state=42,
        )
        self.uq_ = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.q_high,
            max_depth=6,
            learning_rate=0.06,
            random_state=42,
        )

        self.lq_.fit(X_tr, y_tr)
        self.uq_.fit(X_tr, y_tr)

        # Conformal calibration
        lo_cal = self.lq_.predict(X_cal)
        hi_cal = self.uq_.predict(X_cal)
        lo_cal, hi_cal = np.minimum(lo_cal, hi_cal), np.maximum(lo_cal, hi_cal)

        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        self.qhat_ = conformal_quantile(scores, self.alpha)
        self.qhat_lo_ = self.qhat_
        self.qhat_hi_ = self.qhat_
        return self

    def predict_interval(self, X):
        """Predict conformalized quantile intervals."""
        if self.qhat_ is None:
            raise ValueError("CQR model has not been fitted and calibrated.")
        lo_raw = self.lq_.predict(X)
        hi_raw = self.uq_.predict(X)
        lo_raw, hi_raw = np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw)
        lo = lo_raw - self.qhat_
        hi = hi_raw + self.qhat_
        mu = 0.5 * (lo + hi)  # Center of interval
        return mu, lo, hi

    def predict(self, X):
        """Point prediction (mean of quantiles)."""
        mu, _, _ = self.predict_interval(X)
        return mu
