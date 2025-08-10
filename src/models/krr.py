import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from src.featurization.ood import tanimoto_sim_matrix

@dataclass
class TanimotoKRR:
    alpha: float = 1.0
    X_train_: Optional[np.ndarray] = None   # bool matrix (n,d)
    y_train_: Optional[np.ndarray] = None
    L_: Optional[np.ndarray] = None         # Cholesky of (K + alpha I)
    alpha_vec_: Optional[np.ndarray] = None # (K + alpha I)^{-1} y
    k_self_: float = 1.0                    # k(x,x) for Tanimoto on binary bits ~ 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = X.astype(bool)
        K = tanimoto_sim_matrix(Xb, Xb)
        n = K.shape[0]
        K_reg = K + self.alpha * np.eye(n)
        # Cholesky (numériquement plus stable que l'inverse direct)
        L = np.linalg.cholesky(K_reg)
        # Solve (K+αI)α_vec = y via cholesky
        alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y.astype(float)))

        self.X_train_ = Xb
        self.y_train_ = y.astype(float)
        self.L_ = L
        self.alpha_vec_ = alpha_vec
        return self

    def _k_star(self, X: np.ndarray) -> np.ndarray:
        return tanimoto_sim_matrix(X.astype(bool), self.X_train_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        k_star = self._k_star(X)
        return k_star @ self.alpha_vec_

    def predict_mean_var(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne μ(x), σ^2(x) façon GP (bruit = alpha)."""
        k_star = self._k_star(X)              # (m,n)
        # v = L^{-1} k_star^T
        v = np.linalg.solve(self.L_, k_star.T)  # (n,m)
        mu = k_star @ self.alpha_vec_
        # Pour Tanimoto binaire, k(x,x)≈1 (normalisé); sinon clip
        kxx = np.ones(X.shape[0], dtype=float)
        var = kxx - np.sum(v * v, axis=0)
        var = np.clip(var, 1e-9, None)
        return mu, var
