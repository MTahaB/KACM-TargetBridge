# app.py — TargetBridge (Lite) with molecule drawings

import numpy as np
import pandas as pd
import streamlit as st

# RDKit quiet + drawing
try:
    from rdkit import RDLogger, Chem
    from rdkit.Chem import Draw
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

from core import (
    TARGETS, load_tables, morgan_fp, ecfp_matrix, fit_krr,
    split_conformal_q, nearest_ligand_info, ood_score_against_target
)

st.set_page_config(page_title="TargetBridge (Lite)", page_icon="🎯", layout="wide")

st.title("TargetBridge (Lite) 🎯")
st.caption(
    "Paste a SMILES (or pick a demo) → ranked targets with calibrated intervals, "
    "OOD flag, and a short “why”."
)

# ---------- cache ----------
@st.cache_data(show_spinner=False)
def get_tables():
    return load_tables()

@st.cache_data(show_spinner=False)
def render_png(smi: str, w: int = 320, h: int = 240):
    """Return PNG bytes for a SMILES (or None). Cached by SMILES+size."""
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        from io import BytesIO
        img = Draw.MolToImage(mol, size=(w, h))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None

tables = get_tables()
if not tables:
    st.warning(
        "No ligand tables found. Generate them first:\n\n"
        "```bash\npython scripts/fetch_chembl.py\nls -lh data/ligands_*.csv  # NOT 0B\n```"
    )
    st.stop()

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    coverage = st.slider("Conformal coverage", 0.50, 0.99, 0.90, 0.01)
    alpha = 1.0 - coverage
    kappa = st.slider("Ranking aggressiveness (κ for UCB)", 0.5, 2.0, 1.0, 0.05)
    ood_threshold = st.slider("OOD threshold (1 − max Tanimoto)", 0.10, 0.90, 0.30, 0.01)
    widen_ood = st.checkbox("Widen interval if OOD (×1.3)", value=True)

    st.divider()
    st.subheader("Demo molecules")
    DEMOS = {
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Imatinib":  "CC1=CC(NC(C)=O)=C(N2CCN(CC2)C3=NC4=CC=CC=C4N=C3N)C=C1",
        "Gefitinib": "COC1=CC2=CC(NC3=NC(=O)N(C)N=C3N)=C(C=C2C=C1)OCCN",
        "Dopamine":  "NCCc1ccc(O)c(O)c1",
        "Celecoxib": "NS(=O)(=O)c1ccc(cc1)C(F)(F)F"
    }
    demo_choice = st.selectbox("Or pick a demo", ["—"] + list(DEMOS.keys()), index=0)

# ---------- main input ----------
smiles = st.text_input(
    "Enter a molecule as SMILES",
    value=DEMOS.get(demo_choice, "") if demo_choice != "—" else "",
    placeholder="e.g., CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
)

# Show query molecule immediately (if parsable)
if smiles.strip():
    png = render_png(smiles.strip(), 360, 270)
    if png:
        st.image(png, caption="Query molecule")

col_run, col_clear = st.columns(2)
run_clicked = col_run.button("Run")
if col_clear.button("Clear"):
    st.experimental_rerun()

# ---------- core ----------
def predict_for_query(query_smiles: str) -> pd.DataFrame:
    q_fp = morgan_fp(query_smiles)
    if q_fp is None:
        raise ValueError("Invalid SMILES. Please check your input.")

    results = []
    for tid, df in tables.items():
        X_all = ecfp_matrix(df["smiles"].tolist())
        if X_all is None or X_all.shape[0] == 0:
            continue
        y_all = df["pIC50"].values
        if len(y_all) != X_all.shape[0]:
            y_all = y_all[: X_all.shape[0]]

        # train on all + quick split-conformal using last 20% as calibration
        model = fit_krr(X_all, y_all, alpha=1.0)
        n = X_all.shape[0]
        n_cal = max(1, int(0.2 * n))
        X_cal, y_cal = X_all[-n_cal:], y_all[-n_cal:]
        q = split_conformal_q(model, X_cal, y_cal, alpha=alpha)

        mu = float(model.predict(q_fp.reshape(1, -1).astype(np.float32))[0])

        # OOD: 1 − max Tanimoto to training
        ood = ood_score_against_target(q_fp, df)
        q_adj = q * (1.3 if (widen_ood and ood >= ood_threshold) else 1.0)

        lo, hi = mu - q_adj, mu + q_adj

        # nearest ligand for "why"
        w_smiles, w_pic50, w_sim = nearest_ligand_info(q_fp, df)
        why = f"{w_smiles} (pIC50={w_pic50:.2f}) | sim={w_sim:.2f}" if w_smiles else "—"
        score = mu + kappa * q_adj

        results.append((
            tid, mu, lo, hi, "OOD" if ood >= ood_threshold else "", score, why,
            w_smiles, w_pic50, w_sim
        ))

    cols = ["Target", "pIC50_pred", "CI_lo", "CI_hi", "OOD", "score", "why",
            "nearest_smiles", "nearest_pIC50", "nearest_sim"]
    return (pd.DataFrame(results, columns=cols)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))

# ---------- run & display ----------
if run_clicked:
    if not smiles.strip():
        st.warning("Please enter a SMILES or pick a demo, then press **Run**.")
        st.stop()
    with st.spinner("Scouting targets..."):
        try:
            df = predict_for_query(smiles.strip())
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.stop()

    if df.empty:
        st.info("No results to show. Try another molecule.")
        st.stop()

    st.subheader("Results")
    st.dataframe(
        df[["Target", "pIC50_pred", "CI_lo", "CI_hi", "OOD", "score", "why"]],
        hide_index=True
    )

    # Nearest known ligand gallery (top 4)
    st.subheader("Nearest known ligand (per target)")
    top = df.head(4)
    cols = st.columns(len(top)) if len(top) > 0 else []
    for c, (_, row) in zip(cols, top.iterrows()):
        with c:
            pic50_txt = f"{row['nearest_pIC50']:.2f}" if pd.notna(row["nearest_pIC50"]) else "NA"
            st.markdown(f"**{row['Target']}** — pIC50={pic50_txt}")
            img = render_png(row["nearest_smiles"], 320, 240)
            if img:
                st.image(img)
            else:
                st.caption("No image available.")

    # Download
    csv = df[["Target", "pIC50_pred", "CI_lo", "CI_hi", "OOD", "score", "why"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="targetbridge_results.csv", mime="text/csv")

    st.caption(
        f"Conformal coverage set to **{int(coverage*100)}%**. "
        "Intervals adapt to calibration residuals; OOD widens intervals if enabled."
    )

st.divider()
with st.expander("What am I looking at?"):
    st.markdown("""
**pIC50_pred** is the predicted potency (higher is stronger).
**CI_lo / CI_hi** is a split-conformal interval (default 90%).
**OOD** appears when the query is dissimilar to training ligands for that target.
**why** shows the nearest known ligand (by Tanimoto) and its measured pIC50.
Drawings are 2D depictions generated with RDKit from SMILES.
    """)
