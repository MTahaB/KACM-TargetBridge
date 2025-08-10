import numpy as np

def tanimoto_sim_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xb = X.astype(bool)
    Yb = Y.astype(bool)
    inter = Xb @ Yb.T
    a = Xb.sum(axis=1, keepdims=True)
    b = Yb.sum(axis=1, keepdims=True).T
    denom = a + b - inter
    denom[denom == 0] = 1
    return inter / denom

def max_train_similarity(x: np.ndarray, X_train: np.ndarray) -> float:
    sims = tanimoto_sim_matrix(x.reshape(1, -1), X_train)
    return float(sims.max())

def topk_mean_similarity(x: np.ndarray, X_train: np.ndarray, k: int = 8) -> float:
    sims = tanimoto_sim_matrix(x.reshape(1, -1), X_train).ravel()
    if sims.size == 0:
        return 0.0
    idx = np.argsort(-sims)[: max(1, min(k, sims.size))]
    return float(sims[idx].mean())

def novelty_score(x: np.ndarray, X_train: np.ndarray) -> float:
    return 1.0 - max_train_similarity(x, X_train)

def density_score(x: np.ndarray, X_train: np.ndarray, k: int = 8) -> float:
    return topk_mean_similarity(x, X_train, k=k)  # ∈ [0,1]

def ood_composite(x: np.ndarray, X_train: np.ndarray, w_novelty: float = 0.6, k: int = 8) -> float:
    """Retourne un score OOD (↑ = plus OOD). Combinaison nouveauté + faible densité."""
    nov = novelty_score(x, X_train)
    dens = density_score(x, X_train, k=k)
    return float(w_novelty * nov + (1 - w_novelty) * (1 - dens))
