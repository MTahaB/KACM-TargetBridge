import numpy as np

from src.featurization.advanced_features import (
    atom_pair_fingerprint,
    layered_fingerprint,
    pharmacophore_fingerprint,
    rdkit_2d_descriptors,
    topological_torsion_fp,
)
from src.featurization.ood import density_score, tanimoto_sim_matrix
from src.models.conformal import quantile_calibration
from src.models.cqr import conformal_quantile


def test_tanimoto_uses_intersection_counts():
    x = np.array([[1, 1, 0]], dtype=np.uint8)
    y = np.array([[1, 1, 1]], dtype=np.uint8)

    sim = tanimoto_sim_matrix(x, y)

    assert sim.shape == (1, 1)
    assert sim[0, 0] == 2 / 3


def test_density_score_uses_true_tanimoto_values():
    x = np.array([1, 1, 0], dtype=np.uint8)
    train = np.array([[1, 1, 1], [1, 0, 0]], dtype=np.uint8)

    assert density_score(x, train, k=2) == np.mean([2 / 3, 1 / 2])


def test_conformal_quantiles_clamp_small_calibration_sets():
    residuals = np.array([0.1, 0.2, 0.3])

    assert quantile_calibration(residuals, alpha=0.1) == 0.3
    assert conformal_quantile(residuals, alpha=0.1) == 0.3


def test_advanced_rdkit_features_have_stable_shapes():
    smiles = "CCO"

    assert rdkit_2d_descriptors(smiles).shape == (26,)
    assert pharmacophore_fingerprint(smiles, n_bits=128).shape == (128,)
    assert atom_pair_fingerprint(smiles, n_bits=128).shape == (128,)
    assert topological_torsion_fp(smiles, n_bits=128).shape == (128,)
    assert layered_fingerprint(smiles, n_bits=128).shape == (128,)
