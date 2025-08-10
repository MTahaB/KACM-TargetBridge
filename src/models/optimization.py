import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
from src.models.krr import TanimotoKRR
from src.models.conformal import AdaptiveConformalRegressor
from src.eval.metrics import picp, mpiw

@dataclass
class BayesianOptimizer:
    """Optimisation bayésienne pour hyperparamètres avec acquisition UCB."""
    
    def __init__(self, objective_fn: Callable, bounds: Dict[str, Tuple[float, float]], 
                 n_calls: int = 20, random_state: int = 42):
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.n_calls = n_calls
        self.rng = np.random.default_rng(random_state)
        self.X_obs = []
        self.y_obs = []
    
    def _gaussian_process_surrogate(self, X_candidate: np.ndarray) -> Tuple[float, float]:
        """Approximation GP simple pour acquisition function."""
        if len(self.y_obs) < 2:
            return 0.0, 1.0  # Haute incertitude si peu de données
        
        # Distance pondérée aux points observés
        distances = []
        for x_obs in self.X_obs:
            dist = np.sqrt(np.sum((X_candidate - np.array(x_obs))**2))
            distances.append(dist)
        
        # Moyenne pondérée par distance (kernel RBF simplifié)
        weights = np.exp(-np.array(distances))
        weights /= weights.sum()
        
        mean = np.sum(weights * np.array(self.y_obs))
        var = np.var(self.y_obs) * (1.0 - weights.max())  # Incertitude réduite près des observations
        
        return float(mean), float(np.sqrt(var))
    
    def _ucb_acquisition(self, X_candidate: np.ndarray, beta: float = 2.0) -> float:
        """Upper Confidence Bound pour exploration-exploitation."""
        mean, std = self._gaussian_process_surrogate(X_candidate)
        return mean + beta * std  # Maximise UCB
    
    def optimize(self) -> Dict[str, Any]:
        """Optimisation bayésienne des hyperparamètres."""
        param_names = list(self.bounds.keys())
        
        # Exploration initiale aléatoire
        n_initial = min(5, self.n_calls // 2)
        for _ in range(n_initial):
            params = {}
            for name, (low, high) in self.bounds.items():
                params[name] = self.rng.uniform(low, high)
            
            score = self.objective_fn(params)
            self.X_obs.append([params[name] for name in param_names])
            self.y_obs.append(score)
        
        # Optimisation guidée par acquisition
        for _ in range(n_initial, self.n_calls):
            # Optimise acquisition function
            best_acq = -np.inf
            best_params = None
            
            # Grid search sur acquisition (simplifié)
            for _ in range(50):  # 50 candidats aléatoires
                candidate_params = {}
                candidate_vec = []
                for name, (low, high) in self.bounds.items():
                    val = self.rng.uniform(low, high)
                    candidate_params[name] = val
                    candidate_vec.append(val)
                
                acq_val = self._ucb_acquisition(np.array(candidate_vec))
                if acq_val > best_acq:
                    best_acq = acq_val
                    best_params = candidate_params
            
            # Évalue le meilleur candidat
            if best_params:
                score = self.objective_fn(best_params)
                self.X_obs.append([best_params[name] for name in param_names])
                self.y_obs.append(score)
        
        # Retourne les meilleurs hyperparamètres
        best_idx = np.argmax(self.y_obs)
        best_vec = self.X_obs[best_idx]
        best_params = {name: best_vec[i] for i, name in enumerate(param_names)}
        
        return {
            "best_params": best_params,
            "best_score": self.y_obs[best_idx],
            "optimization_history": list(zip(self.X_obs, self.y_obs))
        }


def optimize_conformal_hyperparams(X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 X_cal: np.ndarray, y_cal: np.ndarray) -> Dict[str, Any]:
    """Optimise alpha, gamma pour conformal prediction via score composite."""
    
    def objective(params: Dict[str, float]) -> float:
        """Score composite: coverage + largeur + regularization."""
        alpha = params['alpha']
        gamma = params['gamma']
        
        try:
            # Entraîne modèle conformal
            base_model = TanimotoKRR(alpha=alpha)
            ac_model = AdaptiveConformalRegressor(
                model=base_model, 
                alpha=0.1, 
                gamma=gamma, 
                k_dens=8
            ).fit_calibrate(X_train, y_train, X_cal, y_cal)
            
            # Prédictions sur validation
            mu, lo, hi = ac_model.predict_interval(X_val)
            
            # Métriques
            coverage = picp(y_val, lo, hi)
            width = mpiw(lo, hi)
            
            # Score composite (target coverage = 0.9)
            coverage_penalty = abs(coverage - 0.9) * 10  # Pénalité si loin de 90%
            width_penalty = width  # Minimise largeur
            
            # Score à maximiser (minimise pénalités)
            score = -(coverage_penalty + 0.5 * width_penalty)
            return float(score)
            
        except Exception:
            return -1000.0  # Pénalité forte si échec
    
    # Optimisation bayésienne
    bounds = {
        'alpha': (0.1, 3.0),    # Régularisation KRR
        'gamma': (0.5, 3.0)     # Amplification adaptative
    }
    
    optimizer = BayesianOptimizer(objective, bounds, n_calls=25)
    return optimizer.optimize()


def multi_objective_optimization(X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               X_cal: np.ndarray, y_cal: np.ndarray) -> Dict[str, Any]:
    """Optimisation multi-objectifs: Coverage vs Width (front de Pareto)."""
    
    def evaluate_objectives(params: Dict[str, float]) -> Tuple[float, float]:
        """Retourne (coverage, -width) pour front de Pareto."""
        alpha, gamma = params['alpha'], params['gamma']
        
        try:
            base_model = TanimotoKRR(alpha=alpha)
            ac_model = AdaptiveConformalRegressor(
                model=base_model, alpha=0.1, gamma=gamma, k_dens=8
            ).fit_calibrate(X_train, y_train, X_cal, y_cal)
            
            mu, lo, hi = ac_model.predict_interval(X_val)
            coverage = picp(y_val, lo, hi)
            width = mpiw(lo, hi)
            
            return coverage, -width  # Maximise coverage, minimise width
            
        except Exception:
            return 0.0, -10.0  # Mauvais scores si échec
    
    # Grid search pour front de Pareto
    param_grid = {
        'alpha': np.logspace(-1, 1, 10),  # 0.1 à 10
        'gamma': np.linspace(0.5, 3.0, 10)
    }
    
    pareto_front = []
    all_results = []
    
    for params in ParameterGrid(param_grid):
        coverage, neg_width = evaluate_objectives(params)
        all_results.append({
            'params': dict(params),
            'coverage': coverage,
            'width': -neg_width,
            'objectives': (coverage, neg_width)
        })
    
    # Identifie front de Pareto
    for i, result_i in enumerate(all_results):
        is_dominated = False
        for j, result_j in enumerate(all_results):
            if i != j:
                # result_j domine result_i si meilleur sur tous les objectifs
                if (result_j['objectives'][0] >= result_i['objectives'][0] and 
                    result_j['objectives'][1] >= result_i['objectives'][1] and
                    (result_j['objectives'][0] > result_i['objectives'][0] or 
                     result_j['objectives'][1] > result_i['objectives'][1])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_front.append(result_i)
    
    # Sélectionne solution équilibrée (plus proche de coverage=0.9)
    best_solution = min(pareto_front, key=lambda x: abs(x['coverage'] - 0.9))
    
    return {
        'best_solution': best_solution,
        'pareto_front': pareto_front,
        'all_evaluations': all_results
    }
