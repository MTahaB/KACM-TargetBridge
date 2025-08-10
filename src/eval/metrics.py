import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any

def picp(y_true, lo, hi):
    """Prediction Interval Coverage Probability."""
    return float(((y_true >= lo) & (y_true <= hi)).mean())

def mpiw(lo, hi):
    """Mean Prediction Interval Width."""
    return float(np.mean(hi - lo))

def spearman_rho(x, y):
    """Spearman rank correlation coefficient."""
    rho, _ = spearmanr(x, y)
    return float(rho)

def pearson_r(x, y):
    """Pearson correlation coefficient."""
    r, _ = pearsonr(x, y)
    return float(r)

def coverage_width_ratio(y_true, lo, hi):
    """Coverage-Width Ratio: High coverage with low width is better."""
    coverage = picp(y_true, lo, hi)
    width = mpiw(lo, hi)
    return float(coverage / (width + 1e-8))

def prediction_efficiency(y_true, lo, hi):
    """Prediction Efficiency: Coverage normalized by width."""
    coverage = picp(y_true, lo, hi)
    width = mpiw(lo, hi)
    # Penalize if coverage < 0.85
    penalty = max(0, 0.85 - coverage) * 10
    return float(coverage - 0.1 * width - penalty)

def conditional_coverage_by_uncertainty(y_true, y_pred, lo, hi, n_bins=5):
    """Coverage conditioned on prediction uncertainty (width)."""
    widths = hi - lo
    bin_edges = np.quantile(widths, np.linspace(0, 1, n_bins + 1))
    
    coverages = []
    for i in range(n_bins):
        mask = (widths >= bin_edges[i]) & (widths < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_coverage = picp(y_true[mask], lo[mask], hi[mask])
            coverages.append(bin_coverage)
        else:
            coverages.append(np.nan)
    
    return np.array(coverages)

def reliability_diagram(y_true, y_pred, uncertainties, n_bins=10):
    """Reliability diagram for uncertainty calibration."""
    # Bin by predicted uncertainty
    bin_edges = np.quantile(uncertainties, np.linspace(0, 1, n_bins + 1))
    
    bin_centers = []
    empirical_errors = []
    
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_uncertainty = uncertainties[mask].mean()
            bin_error = np.abs(y_true[mask] - y_pred[mask]).mean()
            
            bin_centers.append(bin_uncertainty)
            empirical_errors.append(bin_error)
    
    return np.array(bin_centers), np.array(empirical_errors)

def sharpness_score(lo, hi, y_train):
    """Sharpness: How tight are intervals relative to data scale."""
    data_range = np.max(y_train) - np.min(y_train)
    avg_width = mpiw(lo, hi)
    return float(1.0 - avg_width / data_range)

def coverage_deviation(y_true, lo, hi, target_coverage=0.9):
    """Absolute deviation from target coverage."""
    actual_coverage = picp(y_true, lo, hi)
    return float(abs(actual_coverage - target_coverage))

def interval_score(y_true, lo, hi, alpha=0.1):
    """Interval Score (lower is better) - proper scoring rule."""
    width = hi - lo
    below = y_true < lo
    above = y_true > hi
    
    score = (width + 
             (2.0 / alpha) * (lo - y_true) * below +
             (2.0 / alpha) * (y_true - hi) * above)
    
    return float(np.mean(score))

def prediction_quality_index(y_true, y_pred, lo, hi):
    """Composite index: Accuracy + Calibration + Sharpness."""
    # Accuracy component
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman = spearman_rho(y_true, y_pred)
    
    # Calibration component  
    coverage = picp(y_true, lo, hi)
    coverage_penalty = abs(coverage - 0.9) * 2
    
    # Sharpness component
    width = mpiw(lo, hi)
    data_range = np.max(y_true) - np.min(y_true)
    normalized_width = width / data_range
    
    # Composite score (0-1, higher is better)
    accuracy_score = max(0, spearman)  # 0-1
    calibration_score = max(0, 1.0 - coverage_penalty)  # 0-1
    sharpness_score = max(0, 1.0 - normalized_width)  # 0-1
    
    # Weighted combination
    pqi = 0.4 * accuracy_score + 0.4 * calibration_score + 0.2 * sharpness_score
    return float(pqi)

def comprehensive_evaluation(y_true, y_pred, lo, hi, y_train=None) -> Dict[str, Any]:
    """Complete evaluation suite for conformal prediction."""
    
    results = {
        # Basic metrics
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'spearman_rho': spearman_rho(y_true, y_pred),
        'pearson_r': pearson_r(y_true, y_pred),
        
        # Conformal prediction metrics
        'picp': picp(y_true, lo, hi),
        'mpiw': mpiw(lo, hi),
        'coverage_width_ratio': coverage_width_ratio(y_true, lo, hi),
        'prediction_efficiency': prediction_efficiency(y_true, lo, hi),
        'coverage_deviation': coverage_deviation(y_true, lo, hi),
        'interval_score': interval_score(y_true, lo, hi),
        
        # Advanced metrics
        'prediction_quality_index': prediction_quality_index(y_true, y_pred, lo, hi),
    }
    
    # Add sharpness if training data available
    if y_train is not None:
        results['sharpness'] = sharpness_score(lo, hi, y_train)
    
    # Conditional coverage analysis
    results['conditional_coverage'] = conditional_coverage_by_uncertainty(
        y_true, y_pred, lo, hi, n_bins=5
    ).tolist()
    
    # Uncertainty calibration
    uncertainties = (hi - lo) / 2  # Half-width as uncertainty proxy
    bin_centers, empirical_errors = reliability_diagram(y_true, y_pred, uncertainties)
    results['reliability_diagram'] = {
        'predicted_uncertainties': bin_centers.tolist(),
        'empirical_errors': empirical_errors.tolist()
    }
    
    return results

def ranking_metrics(y_true, y_pred, top_k=10):
    """Metrics for molecular ranking tasks."""
    # Sort by predicted values (descending)
    sorted_indices = np.argsort(-y_pred)
    
    # Top-k precision in terms of true high-value compounds
    true_threshold = np.quantile(y_true, 0.8)  # Top 20% of true values
    top_k_true = y_true[sorted_indices[:top_k]]
    precision_at_k = (top_k_true >= true_threshold).mean()
    
    # Enrichment factor
    total_actives = (y_true >= true_threshold).sum()
    expected_random = (top_k / len(y_true)) * total_actives
    enrichment_factor = (top_k_true >= true_threshold).sum() / max(1, expected_random)
    
    return {
        'precision_at_k': float(precision_at_k),
        'enrichment_factor': float(enrichment_factor),
        'top_k_mean_true_value': float(np.mean(top_k_true))
    }
