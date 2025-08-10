import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.featurization.ood import tanimoto_sim_matrix

@dataclass
class TanimotoKRR:
    alpha: float = 1.0
    X_train_: Optional[np.ndarray] = None
    alpha_vec_: Optional[np.ndarray] = None
    y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = X.astype(bool)
        K = tanimoto_sim_matrix(Xb, Xb)
        n = K.shape[0]
        K_reg = K + self.alpha * np.eye(n)
        self.alpha_vec_ = np.linalg.solve(K_reg, y)
        self.X_train_ = Xb
        self.y_train_ = y.astype(float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.alpha_vec_ is not None and self.X_train_ is not None
        K_star = tanimoto_sim_matrix(X.astype(bool), self.X_train_)
        return K_star @ self.alpha_vec_
