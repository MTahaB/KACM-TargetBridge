"""Streamlit interface for TargetBridge prediction artifacts."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

try:
    import py3Dmol

    HAVE_3D = True
except ImportError:
    py3Dmol = None
    HAVE_3D = False

from src.featurization.fingerprints import featurize_smiles_list, morgan_fp
from src.featurization.ood import density_score, ood_composite, tanimoto_sim_matrix

ART_DIR = Path("artifacts")
RESULTS_PATH = Path("results/benchmark_results.csv")


def artifact_targets() -> list[str]:
    """Return ChEMBL IDs that have at least one local artifact."""
    if not ART_DIR.exists():
        return []
    return sorted(p.name for p in ART_DIR.iterdir() if p.is_dir())


def parse_targets(raw: str) -> list[str]:
    """Parse user-entered ChEMBL IDs."""
    return [
        token.strip().upper()
        for token in raw.replace("\n", ",").split(",")
        if token.strip()
    ]


@st.cache_resource(show_spinner=False)
def load_pack(target: str, method: str) -> dict[str, Any] | None:
    """Load one prediction artifact."""
    file_by_method = {
        "krr": "model.joblib",
        "cqr": "model_cqr.joblib",
        "ensemble": "model_ensemble.joblib",
    }
    path = ART_DIR / target / file_by_method[method]
    return joblib.load(path) if path.exists() else None


def depict(smiles: str):
    """Return a 2D RDKit depiction."""
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(320, 220)) if mol else None


def show_3d(smiles: str, height: int = 360) -> None:
    """Render a py3Dmol view for a SMILES string."""
    if not HAVE_3D:
        st.info("Install py3Dmol to enable 3D visualization.")
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning("Invalid SMILES for 3D rendering.")
        return
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        st.warning("3D conformer generation failed.")
        return
    AllChem.MMFFOptimizeMolecule(mol)
    viewer = py3Dmol.view(width=480, height=height)
    viewer.addModel(Chem.MolToMolBlock(mol), "sdf")
    viewer.setStyle({"stick": {"colorscheme": "default"}})
    viewer.zoomTo()
    components.html(viewer._make_html(), height=height, scrolling=False)


def explain_knn(
    x: np.ndarray, pack: dict[str, Any], k: int = 3
) -> list[dict[str, Any]]:
    """Return nearest neighbors from artifact metadata."""
    if "neighbors" not in pack:
        return []
    Xtr = pack["neighbors"]["X"].astype(bool)
    sims = tanimoto_sim_matrix(x.reshape(1, -1).astype(bool), Xtr).ravel()
    idx = np.argsort(-sims)[:k]
    return [
        {
            "smiles": pack["neighbors"]["smiles"][i],
            "activity": float(pack["neighbors"]["y"][i]),
            "sim": float(sims[i]),
        }
        for i in idx
    ]


def model_features(smiles: str, pack: dict[str, Any]) -> np.ndarray | None:
    """Build the feature vector expected by a saved artifact."""
    X, keep = featurize_smiles_list(
        [smiles],
        n_bits=pack["bits"],
        radius=pack["radius"],
        feature_set=pack.get("feature_set", "morgan"),
    )
    return None if len(keep) == 0 else X[0]


def predict_krr(smiles: str, pack: dict[str, Any], beta: float, mode: str):
    """Predict with a KRR-conformal artifact."""
    x = morgan_fp(smiles, n_bits=pack["bits"], radius=pack["radius"])
    if x is None:
        return None
    model = pack["model"]
    mu = float(model.predict(x.reshape(1, -1))[0])
    dens = density_score(x, pack["X_train"], k=8)
    width = float(pack["qhat"] * (1.0 + pack.get("gamma", 1.5) * (1.0 - dens)))
    lo, hi = mu - width, mu + width
    sigma_gp = None
    if hasattr(model, "predict_mean_var"):
        _, var = model.predict_mean_var(x.reshape(1, -1))
        sigma_gp = float(np.sqrt(var[0]))
    ood = ood_composite(x, pack["X_train"], w_novelty=0.6, k=8)
    explore = sigma_gp if mode == "ucb_gp" and sigma_gp is not None else hi - mu
    score = mu if mode == "prudent" else mu + beta * explore
    return {
        "mu": mu,
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
        "ood": float(ood),
        "in_domain": bool(ood < pack["ood_tau"]),
        "score": float(score),
        "sigma_gp": sigma_gp,
        "knn": explain_knn(x, pack),
    }


def predict_cqr(smiles: str, pack: dict[str, Any], beta: float, mode: str):
    """Predict with a CQR artifact."""
    x_ood = morgan_fp(smiles, n_bits=pack["bits"], radius=pack["radius"])
    x_model = model_features(smiles, pack)
    if x_ood is None or x_model is None:
        return None
    xf = x_model.reshape(1, -1).astype(float)
    lo_raw = float(pack["model"]["lq"].predict(xf)[0])
    hi_raw = float(pack["model"]["uq"].predict(xf)[0])
    lo_raw, hi_raw = min(lo_raw, hi_raw), max(lo_raw, hi_raw)
    qhat = float(pack.get("qhat", pack.get("qhat_hi", 0.0)))
    lo, hi = lo_raw - qhat, hi_raw + qhat
    mu = 0.5 * (lo + hi)
    ood = ood_composite(x_ood, pack["X_train"], w_novelty=0.6, k=8)
    score = mu if mode == "prudent" else mu + beta * (hi - mu)
    return {
        "mu": float(mu),
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
        "ood": float(ood),
        "in_domain": bool(ood < pack["ood_tau"]),
        "score": float(score),
        "sigma_gp": np.nan,
        "knn": explain_knn(x_ood, pack),
    }


def predict_ensemble(smiles: str, pack: dict[str, Any], beta: float, mode: str):
    """Predict with a stacked ensemble artifact."""
    x_ood = morgan_fp(smiles, n_bits=pack["bits"], radius=pack["radius"])
    x_model = model_features(smiles, pack)
    if x_ood is None or x_model is None:
        return None
    pred, unc = pack["model"].predict_with_uncertainty(x_model.reshape(1, -1))
    mu = float(pred[0])
    half_width = float(pack.get("ensemble_z", 1.64) * unc[0])
    lo, hi = mu - half_width, mu + half_width
    ood = ood_composite(x_ood, pack["X_train"], w_novelty=0.6, k=8)
    score = mu if mode == "prudent" else mu + beta * half_width
    return {
        "mu": mu,
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
        "ood": float(ood),
        "in_domain": bool(ood < pack["ood_tau"]),
        "score": float(score),
        "sigma_gp": np.nan,
        "knn": explain_knn(x_ood, pack),
    }


def predict_one(smiles: str, pack: dict[str, Any], method: str, beta: float, mode: str):
    """Dispatch prediction to the selected method."""
    if method == "krr":
        return predict_krr(smiles, pack, beta, mode)
    if method == "cqr":
        return predict_cqr(smiles, pack, beta, mode)
    return predict_ensemble(smiles, pack, beta, mode)


def show_neighbor_molecules(neighbors: list[dict[str, Any]]) -> None:
    """Display nearest training molecules for one prediction."""
    if not neighbors:
        st.info("No nearest-neighbor metadata is available for this artifact.")
        return
    columns = st.columns(min(3, len(neighbors)))
    for column, neighbor in zip(columns, neighbors):
        with column:
            img = depict(neighbor["smiles"])
            if img:
                st.image(img, use_container_width=True)
            st.metric("Similarity", f"{neighbor['sim']:.3f}")
            st.write(f"Activity: {neighbor['activity']:.3f}")
            st.code(neighbor["smiles"], language=None)


def show_prediction_detail(
    row: pd.Series, prediction_detail: dict[str, Any], show_3d_panel: bool
) -> None:
    """Display molecule views and explanation for a selected prediction."""
    st.subheader("Top-ranked molecule")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction", f"{row['mu']:.3f}")
    m2.metric("Interval", f"{row['lo']:.3f} - {row['hi']:.3f}")
    m3.metric("OOD", f"{row['ood']:.3f}")
    m4.metric("Score", f"{row['score']:.3f}")

    tab_2d, tab_3d, tab_neighbors, tab_record = st.tabs(
        ["2D molecule", "3D molecule", "Nearest molecules", "Record"]
    )
    with tab_2d:
        img = depict(str(row["smiles"]))
        if img:
            st.image(img)
        st.code(str(row["smiles"]), language=None)
    with tab_3d:
        if show_3d_panel:
            show_3d(str(row["smiles"]))
        elif HAVE_3D:
            st.info("Enable the 3D panel in the sidebar.")
        else:
            st.info("py3Dmol is not installed in this environment.")
    with tab_neighbors:
        show_neighbor_molecules(prediction_detail.get("knn", []))
    with tab_record:
        st.json(row.drop(labels=["prediction_id"]).to_dict())


def results_to_sdf(results: pd.DataFrame) -> bytes:
    """Convert prediction results to SDF bytes with prediction properties."""
    writer_buffer = StringIO()
    writer = Chem.SDWriter(writer_buffer)
    for _, row in results.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        for col, value in row.items():
            if col != "smiles":
                mol.SetProp(str(col), str(value))
        writer.write(mol)
    writer.close()
    return writer_buffer.getvalue().encode("utf-8")


def show_reliability_diagram() -> None:
    """Display empirical PICP values from the benchmark CSV."""
    if not RESULTS_PATH.exists():
        return
    df = pd.read_csv(RESULTS_PATH).dropna(subset=["picp", "mpiw"])
    if df.empty:
        return
    st.subheader("Empirical Interval Coverage")
    fig = px.bar(
        df,
        x="target_chembl_id",
        y="picp",
        color="model",
        barmode="group",
        labels={"picp": "PICP", "target_chembl_id": "Target"},
        hover_data=["mpiw", "n_test"],
    )
    fig.add_hline(y=0.9, line_dash="dash", annotation_text="Nominal 0.90")
    st.plotly_chart(fig, width="stretch")


st.set_page_config(page_title="KACM-TargetBridge", layout="wide")
st.title("KACM-TargetBridge")
st.caption("Ligand-based molecular activity prediction with empirical intervals.")

with st.sidebar:
    st.header("Prediction Settings")
    method_label = st.selectbox(
        "Method",
        ["krr", "cqr", "ensemble"],
        format_func={"krr": "KRR-Conformal", "cqr": "CQR", "ensemble": "Ensemble"}.get,
    )
    mode = st.selectbox(
        "Ranking mode",
        ["prudent", "ucb_gp", "ucb_width"],
        format_func={
            "prudent": "Prudent (mean)",
            "ucb_gp": "UCB with GP sigma",
            "ucb_width": "UCB with interval width",
        }.get,
    )
    beta = st.slider("UCB beta", 0.0, 2.0, 0.6, 0.1)
    abstain_w = st.slider("Abstain if interval width >=", 0.5, 6.0, 1.2, 0.1)
    show_3d_flag = st.checkbox("3D molecule panel", value=True, disabled=not HAVE_3D)

    discovered = artifact_targets()
    extra_targets = parse_targets(st.text_area("Additional ChEMBL IDs", ""))
    target_options = sorted(set(discovered + extra_targets))
    selected_targets = st.multiselect("Targets", target_options, default=discovered)

left, right = st.columns([1, 1])
with left:
    smiles = st.text_area("SMILES", "CC1=CC=C(C=C1)C(C(=O)O)N")
with right:
    upload = st.file_uploader("CSV with a smiles column", type=["csv"])

if st.button("Predict", type="primary"):
    if upload is not None:
        uploaded = pd.read_csv(upload)
        if "smiles" not in uploaded.columns:
            st.error("Uploaded CSV must contain a smiles column.")
            st.stop()
        input_df = uploaded[["smiles"]].dropna()
    else:
        input_df = pd.DataFrame({"smiles": [smiles]})

    rows: list[dict[str, Any]] = []
    prediction_details: dict[int, dict[str, Any]] = {}
    missing: list[str] = []
    for target in selected_targets:
        pack = load_pack(target, method_label)
        if pack is None:
            missing.append(target)
            continue
        for smi in input_df["smiles"]:
            pred = predict_one(str(smi), pack, method_label, beta, mode)
            if pred is None:
                continue
            prediction_id = len(rows)
            rows.append(
                {
                    "prediction_id": prediction_id,
                    "smiles": smi,
                    "target": target,
                    "endpoint": pack.get("endpoint", "pIC50"),
                    "method": method_label,
                    **{k: v for k, v in pred.items() if k != "knn"},
                    "abstain": (not pred["in_domain"]) or pred["width"] >= abstain_w,
                }
            )
            prediction_details[prediction_id] = pred

    if missing:
        st.warning(
            "No local artifact for: "
            + ", ".join(missing)
            + ". Train with scripts/train_per_target.py first."
        )

    if not rows:
        st.error("No predictions were produced.")
        st.stop()

    res = pd.DataFrame(rows).sort_values(["smiles", "score"], ascending=[True, False])
    st.success(f"{len(res)} predictions")
    display_res = res.drop(columns=["prediction_id"])
    st.dataframe(display_res, width="stretch")

    csv_bytes = display_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export CSV",
        data=csv_bytes,
        file_name="targetbridge_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Export SDF",
        data=results_to_sdf(display_res),
        file_name="targetbridge_predictions.sdf",
        mime="chemical/x-mdl-sdfile",
    )

    first = res.iloc[0]
    first_detail = prediction_details[int(first["prediction_id"])]
    show_prediction_detail(first, first_detail, show_3d_flag)

    with st.expander("Nearest molecules for all predictions"):
        for _, result_row in res.head(12).iterrows():
            detail = prediction_details[int(result_row["prediction_id"])]
            st.markdown(
                f"**{result_row['target']}** - score {result_row['score']:.3f}"
            )
            show_neighbor_molecules(detail.get("knn", []))

show_reliability_diagram()
