import numpy as np
from dataclasses import dataclass
from src.featurization.ood import density_score

def quantile_calibration(residuals: np.ndarray, alpha: float) -> float:
    n = residuals.shape[0]
    q = np.quantile(residuals, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    return float(q)

@dataclass
class AdaptiveConformalRegressor:
    """Split conformal avec échelle locale ~ f(densité).
       s(x) = 1 + gamma * (1 - densité_kNN(x)),
       on calibre sur |resid| / s(x_cal), puis à l’inférence on multiplie q̂ par s(x).
    """
    model: any
    alpha: float = 0.1
    gamma: float = 1.5
    k_dens: int = 8
    qhat_: float | None = None
    X_train_: np.ndarray | None = None

    def fit_calibrate(self, X_train, y_train, X_cal, y_cal):
        self.model.fit(X_train, y_train)
        self.X_train_ = X_train
        mu_cal = self.model.predict(X_cal)
        resid = np.abs(y_cal - mu_cal)
        s_cal = 1.0 + self.gamma * (1.0 - np.array([density_score(x, X_train, k=self.k_dens) for x in X_cal]))
        scaled = resid / s_cal
        self.qhat_ = quantile_calibration(scaled, self.alpha)
        return self

    def predict_interval(self, X):
        assert self.qhat_ is not None and self.X_train_ is not None
        mu = self.model.predict(X)
        s = 1.0 + self.gamma * (1.0 - np.array([density_score(x, self.X_train_, k=self.k_dens) for x in X]))
        w = self.qhat_ * s
        lo, hi = mu - w, mu + w
        return mu, lo, hi
