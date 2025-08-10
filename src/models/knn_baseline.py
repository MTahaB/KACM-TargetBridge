import numpy as np
from dataclasses import dataclass
from src.featurization.ood import tanimoto_sim_matrix

@dataclass
class KNNRegressorTanimoto:
    k: int = 5

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train_ = X.astype(bool)
        self.y_train_ = y.astype(float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        sims = tanimoto_sim_matrix(X.astype(bool), self.X_train_)
        idx = np.argsort(-sims, axis=1)[:, : self.k]
        topk_sims = np.take_along_axis(sims, idx, axis=1)
        topk_vals = self.y_train_[idx]
        w = topk_sims + 1e-8
        return (w * topk_vals).sum(axis=1) / w.sum(axis=1)
