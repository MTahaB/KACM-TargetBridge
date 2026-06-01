import numpy as np
import pandas as pd

from scripts.download_chembl import activity_to_pvalue, prepare_endpoint_table
from src.data.split import scaffold_split
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


def test_download_preparation_keeps_endpoints_separate():
    raw = pd.DataFrame(
        {
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL1"],
            "smiles": ["CCO", "CCO"],
            "value": [100.0, 1000.0],
            "units": ["nM", "nM"],
            "activity_type": ["IC50", "IC50"],
            "relation": ["=", "="],
        }
    )

    table = prepare_endpoint_table(raw, "CHEMBL203", "IC50")

    assert list(table.columns) == [
        "smiles",
        "pIC50",
        "target_chembl_id",
        "endpoint",
        "n_measurements",
    ]
    assert table.loc[0, "pIC50"] == np.median([7.0, 6.0])
    assert table.loc[0, "endpoint"] == "pIC50"


def test_activity_conversion_handles_micro_molar_units():
    assert activity_to_pvalue(1.0, "uM") == 6.0
    assert activity_to_pvalue(1.0, "\u00b5M") == 6.0


def test_scaffold_split_is_reproducible_for_noscaffold_molecules():
    df = pd.DataFrame({"smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"]})

    train_a, test_a = scaffold_split(df, seed=7)
    train_b, test_b = scaffold_split(df, seed=7)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)
