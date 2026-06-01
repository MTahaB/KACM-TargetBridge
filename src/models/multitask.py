"""Experimental target-aware multi-task models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TargetAwareRidge:
    """Ridge regression with target identity as an additional feature block.

    Parameters
    ----------
    alpha:
        Ridge regularization strength.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler(with_mean=False)
        self.model = Ridge(alpha=alpha)

    def _design(self, X: np.ndarray, target_ids: list[str] | np.ndarray) -> np.ndarray:
        target_block = self.encoder.transform(np.asarray(target_ids).reshape(-1, 1))
        X_scaled = self.scaler.transform(X.astype(float))
        return np.hstack([X_scaled, target_block])

    def fit(
        self, X: np.ndarray, y: np.ndarray, target_ids: list[str] | np.ndarray
    ) -> "TargetAwareRidge":
        """Fit the multi-task model.

        Parameters
        ----------
        X:
            Molecular feature matrix.
        y:
            Activity values.
        target_ids:
            Target identifier for each row.

        Returns
        -------
        TargetAwareRidge
            Fitted estimator.
        """
        target_ids = np.asarray(target_ids).reshape(-1, 1)
        self.encoder.fit(target_ids)
        self.scaler.fit(X.astype(float))
        design = self._design(X, target_ids.ravel())
        self.model.fit(design, y)
        return self

    def predict(self, X: np.ndarray, target_ids: list[str] | np.ndarray) -> np.ndarray:
        """Predict activities for molecules and target IDs."""
        return self.model.predict(self._design(X, target_ids))
