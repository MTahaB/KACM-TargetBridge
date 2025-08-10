import numpy as np
from dataclasses import dataclass
from typing import List, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from src.models.krr import TanimotoKRR
from src.models.knn_baseline import KNNRegressorTanimoto
from src.featurization.ood import tanimoto_sim_matrix, density_score

@dataclass
class StackedEnsemble:
    """Ensemble sophistiqué combinant KRR, RF, et Ridge avec meta-learner adaptatif.
    
    Architecture:
    - Base models: TanimotoKRR + RandomForest + Ridge
    - Meta-learner: Ridge weighted by density score
    - Confidence: Ensemble disagreement + individual uncertainties
    """
    krr_alpha: float = 1.0
    rf_n_estimators: int = 100
    ridge_alpha: float = 1.0
    meta_alpha: float = 0.1
    
    # Fitted components
    krr_model: Optional[TanimotoKRR] = None
    rf_model: Optional[RandomForestRegressor] = None  
    ridge_model: Optional[Ridge] = None
    meta_model: Optional[Ridge] = None
    X_train_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Entraîne l'ensemble avec validation pour le meta-learner."""
        self.X_train_ = X.astype(bool)
        
        # Base models sur données d'entraînement
        self.krr_model = TanimotoKRR(alpha=self.krr_alpha).fit(X, y)
        self.rf_model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators, 
            random_state=42,
            max_depth=8,
            min_samples_split=10
        ).fit(X.astype(float), y)
        self.ridge_model = Ridge(alpha=self.ridge_alpha, random_state=42).fit(X.astype(float), y)
        
        # Prédictions sur validation pour meta-learner
        krr_val = self.krr_model.predict(X_val)
        rf_val = self.rf_model.predict(X_val.astype(float))
        ridge_val = self.ridge_model.predict(X_val.astype(float))
        
        # Features pour meta-learner: [pred1, pred2, pred3, density_score]
        meta_features = []
        for i, x in enumerate(X_val):
            dens = density_score(x, self.X_train_, k=8)
            meta_features.append([krr_val[i], rf_val[i], ridge_val[i], dens])
        
        meta_X = np.array(meta_features)
        self.meta_model = Ridge(alpha=self.meta_alpha, random_state=42).fit(meta_X, y_val)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction ensemble avec meta-learner."""
        assert all([self.krr_model, self.rf_model, self.ridge_model, self.meta_model])
        
        krr_pred = self.krr_model.predict(X)
        rf_pred = self.rf_model.predict(X.astype(float))
        ridge_pred = self.ridge_model.predict(X.astype(float))
        
        # Meta-features avec densité
        meta_features = []
        for i, x in enumerate(X):
            dens = density_score(x, self.X_train_, k=8)
            meta_features.append([krr_pred[i], rf_pred[i], ridge_pred[i], dens])
        
        meta_X = np.array(meta_features)
        return self.meta_model.predict(meta_X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prédiction + incertitude basée sur disagreement des base models."""
        krr_pred = self.krr_model.predict(X)
        rf_pred = self.rf_model.predict(X.astype(float))
        ridge_pred = self.ridge_model.predict(X.astype(float))
        
        # Prédiction finale
        meta_features = []
        for i, x in enumerate(X):
            dens = density_score(x, self.X_train_, k=8)
            meta_features.append([krr_pred[i], rf_pred[i], ridge_pred[i], dens])
        
        meta_X = np.array(meta_features)
        final_pred = self.meta_model.predict(meta_X)
        
        # Incertitude = désaccord + densité
        base_preds = np.column_stack([krr_pred, rf_pred, ridge_pred])
        disagreement = np.std(base_preds, axis=1)
        
        # Ajustement par densité (moins de données = plus d'incertitude)
        densities = np.array([density_score(x, self.X_train_, k=8) for x in X])
        density_penalty = 1.0 + 0.5 * (1.0 - densities)
        
        uncertainty = disagreement * density_penalty
        return final_pred, uncertainty


@dataclass 
class BayesianEnsemble:
    """Ensemble bayésien avec bootstrap aggregating et calibration d'incertitude."""
    base_models: List[Any] = None
    n_bootstrap: int = 10
    bootstrap_ratio: float = 0.8
    
    def __init__(self, n_bootstrap: int = 10):
        self.n_bootstrap = n_bootstrap
        self.base_models = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Bootstrap sampling + entraînement de multiples modèles."""
        rng = np.random.default_rng(42)
        n_samples = X.shape[0]
        sample_size = int(self.bootstrap_ratio * n_samples)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = rng.choice(n_samples, size=sample_size, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Modèle avec bruit aléatoire pour diversité
            alpha = rng.uniform(0.5, 2.0)  # Régularisation aléatoire
            model = TanimotoKRR(alpha=alpha).fit(X_boot, y_boot)
            self.base_models.append(model)
        
        return self
    
    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne moyenne, std, et quantiles de la distribution prédictive."""
        predictions = np.array([model.predict(X) for model in self.base_models])
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        q05 = np.quantile(predictions, 0.05, axis=0)
        q95 = np.quantile(predictions, 0.95, axis=0)
        
        return mean_pred, std_pred, (q05, q95)
