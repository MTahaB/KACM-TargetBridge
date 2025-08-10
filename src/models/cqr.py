import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import HistGradientBoostingRegressor

@dataclass
class CQR:
    """Conformalized Quantile Regression with HistGradientBoostingRegressor.
    
    Advanced technique combining quantile regression with conformal prediction
    for sharper intervals while maintaining coverage guarantees.
    """
    alpha: float = 0.1
    lq_: any = None
    uq_: any = None
    qhat_lo_: float | None = None
    qhat_hi_: float | None = None
    q_low: float = 0.05
    q_high: float = 0.95

    def fit(self, X_tr, y_tr, X_cal, y_cal):
        """Fit quantile regressors and calibrate on holdout set."""
        self.q_low  = self.alpha/2.0
        self.q_high = 1.0 - self.alpha/2.0
        
        # Train quantile regressors
        self.lq_ = HistGradientBoostingRegressor(
            loss="quantile", 
            quantile=self.q_low, 
            max_depth=6, 
            learning_rate=0.06,
            random_state=42
        )
        self.uq_ = HistGradientBoostingRegressor(
            loss="quantile", 
            quantile=self.q_high, 
            max_depth=6, 
            learning_rate=0.06,
            random_state=42
        )
        
        self.lq_.fit(X_tr, y_tr)
        self.uq_.fit(X_tr, y_tr)
        
        # Conformal calibration
        lo_cal = self.lq_.predict(X_cal)
        hi_cal = self.uq_.predict(X_cal)
        
        # CQR conformalization scores
        E_lo = lo_cal - y_cal  # Lower miscoverage
        E_hi = y_cal - hi_cal  # Upper miscoverage
        
        n = len(y_cal)
        def qcorr(e):
            e = np.sort(e)
            k = int(np.ceil((n+1)*(1-self.alpha))/1) - 1
            k = min(max(k, 0), n-1)
            return float(e[k])
        
        self.qhat_lo_ = qcorr(E_lo)
        self.qhat_hi_ = qcorr(E_hi)
        return self

    def predict_interval(self, X):
        """Predict conformalized quantile intervals."""
        lo = self.lq_.predict(X) - self.qhat_lo_
        hi = self.uq_.predict(X) + self.qhat_hi_
        mu = 0.5 * (lo + hi)  # Center of interval
        return mu, lo, hi
    
    def predict(self, X):
        """Point prediction (mean of quantiles)."""
        mu, _, _ = self.predict_interval(X)
        return mu
