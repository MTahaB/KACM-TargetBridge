import numpy as np
from scipy.stats import spearmanr

def picp(y_true, lo, hi):
    return float(((y_true >= lo) & (y_true <= hi)).mean())

def mpiw(lo, hi):
    return float(np.mean(hi - lo))

def spearman_rho(x, y):
    rho, _ = spearmanr(x, y)
    return float(rho)
