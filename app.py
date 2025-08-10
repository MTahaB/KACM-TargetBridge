import os, gc, math, json
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO
from typing import Dict, List
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------
# Theme / Page config
# ----------------------------
st.set_page_config(page_title="TargetBridge (Lite) – Mini Affinity",
                   page_icon="🧪", layout="wide")

# ----------------------------
# RDKit (optionnel)
# ----------------------------
FEATURE_SIZE = 2048
USE_RDKIT = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    USE_RDKIT = False

# ----------------------------
# Helpers: featurisation
# ----------------------------
def smiles_to_ecfp4(smiles: str, n_bits: int = FEATURE_SIZE) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    if not USE_RDKIT:
        return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    for b in fp.GetOnBits():
        arr[b] = 1.0
    return arr

def smiles_to_hash_ngrams(smiles: str, n_bits: int = FEATURE_SIZE, n_min=2, n_max=3) -> np.ndarray:
    s = f"^{smiles}$"
    vec = np.zeros(n_bits, dtype=np.float32)
    for n in range(n_min, n_max+1):
        for i in range(len(s)-n+1):
            vec[hash(s[i:i+n]) % n_bits] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def featurize_smiles_list(smiles_list):
    if USE_RDKIT:
        return np.vstack([smiles_to_ecfp4(s) for s in smiles_list]).astype(np.float32)
    else:
        return np.vstack([smiles_to_hash_ngrams(s) for s in smiles_list]).astype(np.float32)

# ----------------------------
# Helpers: conformal & OOD
# ----------------------------
def conformal_q90(y_true, y_pred):
    return float(np.quantile(np.abs(y_true - y_pred), 0.9))

def tanimoto_max_to_train(x, X_train):
    x = x.astype(bool)
    Xb = X_train.astype(bool)
    inter = (Xb & x).sum(axis=1).astype(float)
    union = (Xb | x).sum(axis=1).astype(float)
    sim = np.where(union > 0, inter/union, 0.0)
    return float(sim.max()) if sim.size else 0.0

def cosine_max_to_train(x, X_train):
    denom = (np.linalg.norm(x) * np.linalg.norm(X_train, axis=1) + 1e-8)
    sims = X_train @ x / denom
    return float(sims.max()) if sims.size else 0.0

# ----------------------------
# Cache: data & training
# ----------------------------
@st.cache_data(show_spinner=False)
def load_dataframe_from_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(file_bytes.decode("utf-8")))

@st.cache_data(show_spinner=False)
def clean_df(df: pd.DataFrame, dedupe: str, pmin: float, pmax: float, min_per_target: int) -> pd.DataFrame:
    assert {"smiles","target","pIC50"}.issubset(df.columns), "CSV must contain smiles,target,pIC50"
    df = df[["smiles","target","pIC50"]].dropna()
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["pIC50"]).reset_index(drop=True)
    # clamp pIC50 to a reasonable window
    df = df[(df["pIC50"] >= pmin) & (df["pIC50"] <= pmax)].reset_index(drop=True)
    # dedupe on (smiles,target)
    if dedupe in ("median","mean"):
        agg = {"pIC50": "median" if dedupe=="median" else "mean"}
        df = df.groupby(["smiles","target"], as_index=False).agg(agg)
    # keep targets with enough samples
    if min_per_target > 0:
        counts = df["target"].value_counts()
        keep = counts[counts >= min_per_target].index
        df = df[df["target"].isin(keep)].reset_index(drop=True)
    return df

@st.cache_data(show_spinner=True)
def build_features(df: pd.DataFrame) -> np.ndarray:
    return featurize_smiles_list(df["smiles"].tolist())

@st.cache_resource(show_spinner=True)
def train_per_target(df: pd.DataFrame, X: np.ndarray, model_type: str, n_jobs: int, random_state: int = 42):
    # index by target
    y = df["pIC50"].values.astype(float)
    target_to_idxs = {tgt: np.where(df["target"].values == tgt)[0] for tgt in df["target"].unique()}
    targets = list(target_to_idxs.keys())

    def train_one(tgt):
        idxs = target_to_idxs[tgt]
        X_tr, X_te, y_tr, y_te = train_test_split(X[idxs], y[idxs], test_size=0.2, random_state=random_state)
        if model_type == "Kernel Ridge (linear)":
            base = KernelRidge(alpha=1.0, kernel="linear").fit(X_tr, y_tr)
        else:
            base = RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1).fit(X_tr, y_tr)
        y_pred = base.predict(X_te)
        rmse = float(mean_squared_error(y_te, y_pred) ** 0.5)
        mae  = float(mean_absolute_error(y_te, y_pred))
        r2v  = float(r2_score(y_te, y_pred))

        # conformal split (q90)
        X_tr2, X_cal, y_tr2, y_cal = train_test_split(X[idxs], y[idxs], test_size=0.3, random_state=random_state+1)
        if model_type == "Kernel Ridge (linear)":
            mdl2 = KernelRidge(alpha=1.0, kernel="linear").fit(X_tr2, y_tr2)
        else:
            mdl2 = RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1).fit(X_tr2, y_tr2)
        q90 = conformal_q90(y_cal, mdl2.predict(X_cal))

        return {
            "target": tgt,
            "n": int(len(idxs)),
            "rmse": rmse, "mae": mae, "r2": r2v,
            "q90": float(q90),
            "model": mdl2,
            "X_train": X_tr2,
        }

    trained = Parallel(n_jobs=n_jobs, verbose=0)(delayed(train_one)(tgt) for tgt in targets)
    rep = pd.DataFrame([{k:v for k,v in t.items() if k in ("target","n","rmse","mae","r2","q90")} for t in trained])
    rep = rep.sort_values(["n","r2"], ascending=[False, False]).reset_index(drop=True)
    return trained, rep

# ----------------------------
# UI — Sidebar
# ----------------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.write(f"RDKit disponible : **{USE_RDKIT}**")
uploaded = st.sidebar.file_uploader("Uploader un CSV (colonnes: smiles,target,pIC50)", type=["csv"])

col_sidebar = st.sidebar
dedupe = col_sidebar.selectbox("Dédoublonnage (smiles,target)", ["median","mean","none"], index=0)
pmin, pmax = col_sidebar.slider("Fenêtre pIC50 conservée", 3.0, 12.0, (4.0, 10.0), 0.1)
min_per_target = col_sidebar.number_input("Min exemples/cible", min_value=50, max_value=5000, value=300, step=50)
model_type = col_sidebar.selectbox("Modèle", ["Kernel Ridge (linear)","Random Forest"], index=1)
n_jobs = col_sidebar.slider("Parallélisme (n_jobs)", 1, 16, 4)
demo_smiles = col_sidebar.text_input("SMILES pour la démo", "CC(C)Cc1ccc(cc1)C(C)C(=O)O")
run_train = col_sidebar.button("🚀 Entraîner / Réentraîner", type="primary")

# ----------------------------
# Main layout (tabs)
# ----------------------------
st.title("🧪 TargetBridge (Lite): Uncertainty‑aware target scouting")
tabs = st.tabs(["Train & Evaluate", "Predict", "About"])

with tabs[0]:
    st.subheader("1) Données & entraînement")
    if uploaded is None:
        st.info("Charge un CSV (smiles,target,pIC50) via la sidebar pour démarrer.")
        st.stop()

    # Load & clean
    raw_df = load_dataframe_from_csv(uploaded.getvalue())
    st.write("Aperçu des données brutes :")
    st.dataframe(raw_df.head(10), use_container_width=True)

    df = clean_df(raw_df, dedupe=dedupe, pmin=pmin, pmax=pmax, min_per_target=min_per_target)
    st.success(f"Nettoyage OK — lignes: {len(df)} | cibles: {df['target'].nunique()}")
    st.write(df.head(10))

    # Features
    with st.spinner("Featurisation en cours…"):
        X = build_features(df)
    st.caption(f"Featurisation: matrice {X.shape} | RDKit={USE_RDKIT}")

    # Train
    if run_train or "trained_state" not in st.session_state:
        with st.spinner("Entraînement des modèles par cible…"):
            trained, rep = train_per_target(df, X, model_type=model_type, n_jobs=n_jobs)
        st.session_state["trained_state"] = {"df": df, "X": X, "trained": trained, "rep": rep, "model_type": model_type}
    else:
        trained = st.session_state["trained_state"]["trained"]
        rep = st.session_state["trained_state"]["rep"]
        df = st.session_state["trained_state"]["df"]
        X = st.session_state["trained_state"]["X"]

    st.subheader("2) Rapport par cible")
    st.dataframe(rep, use_container_width=True)
    csv_rep = rep.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Télécharger le rapport (CSV)", csv_rep, file_name="report_per_target.csv")

    # Plots (Top-6 cibles par n)
    st.subheader("3) Plots rapides (Prédit vs Vrai)")
    top_targets = rep["target"].head(6).tolist()
    cols = st.columns(3)
    for i, tgt in enumerate(top_targets):
        with cols[i % 3]:
            idxs = np.where(df["target"].values == tgt)[0]
            X_tr, X_te, y_tr, y_te = train_test_split(X[idxs], df["pIC50"].values[idxs],
                                                      test_size=0.2, random_state=0)
            if model_type == "Kernel Ridge (linear)":
                mdl = KernelRidge(alpha=1.0, kernel="linear").fit(X_tr, y_tr)
            else:
                mdl = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)
            rmse = (mean_squared_error(y_te, y_pred))**0.5
            r2v  = r2_score(y_te, y_pred)

            fig = plt.figure()
            plt.scatter(y_te, y_pred, s=10)
            lo, hi = float(min(y_te.min(), y_pred.min())), float(max(y_te.max(), y_pred.max()))
            plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel("pIC50 (vrai)"); plt.ylabel("pIC50 (prévu)")
            plt.title(f"{tgt}\nRMSE={rmse:.2f} | R²={r2v:.2f}")
            plt.grid(True)
            st.pyplot(fig)

with tabs[1]:
    st.subheader("Prédire pour un nouveau SMILES")
    if "trained_state" not in st.session_state:
        st.info("Entraîne d'abord les modèles dans l'onglet **Train & Evaluate**.")
    else:
        df = st.session_state["trained_state"]["df"]
        trained = st.session_state["trained_state"]["trained"]

        def predict_for_smiles(smiles, trained, widen_on_ood=True):
            x = featurize_smiles_list([smiles])[0]
            rows = []
            for t in trained:
                mdl = t["model"]; Xtr = t["X_train"]; tgt = t["target"]; q90 = t["q90"]
                y_hat = float(mdl.predict(x.reshape(1,-1))[0])
                sim = tanimoto_max_to_train(x, Xtr) if USE_RDKIT else cosine_max_to_train(x, Xtr)
                ood = 1.0 - sim
                half = q90 * (1.3 if (widen_on_ood and ood > 0.5) else 1.0)
                rows.append({
                    "target": tgt,
                    "pIC50_pred": y_hat,
                    "ci90_lo": y_hat - half,
                    "ci90_hi": y_hat + half,
                    "width": half,
                    "ood": ood,
                    "OOD_badge": "⚠️ OOD" if ood > 0.5 else "✅ In‑dist",
                })
            out = pd.DataFrame(rows).sort_values("pIC50_pred", ascending=False).reset_index(drop=True)
            return out

        st.write("Entre un SMILES et clique **Prédire**.")
        smi = st.text_input("SMILES", demo_smiles)
        if st.button("🔮 Prédire"):
            ranked = predict_for_smiles(smi, trained, widen_on_ood=True)
            st.dataframe(ranked, use_container_width=True)
            csv_ranked = ranked.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Télécharger les prédictions (CSV)", csv_ranked, file_name="ranked_predictions.csv")

            # Graphique Top‑5
            top5 = ranked.head(5).copy()
            fig = plt.figure()
            plt.errorbar(top5["target"], top5["pIC50_pred"], yerr=top5["width"], fmt='o', capsize=4)
            plt.title("Top‑5 cibles – pIC50 prédite (± CI90)")
            plt.ylabel("pIC50 (plus haut = meilleure affinité)")
            plt.xlabel("Cible")
            plt.grid(True)
            st.pyplot(fig)

with tabs[2]:
    st.subheader("À propos")
    st.markdown("""
**TargetBridge (Lite)** – *uncertainty‑aware target scouting*

- **Entrée**: SMILES (ligand).  
- **Sortie**: classement des cibles par **pIC50** prédit, avec **intervalle de confiance (CI90)** et **badge OOD** (hors‑distribution).  
- **Pourquoi c’est crédible ?** On annonce **l’incertitude** (conformal) et on **élargit** l’intervalle si la molécule est hors‑manifold.  
- **Tech**: ECFP4 (ou hashing fallback) → petit modèle par cible (**Kernel Ridge** ou **Random Forest**).  
- **Jury‑ready**: UX rapide, CSV exports, plots “prédit vs vrai”.

*Note:* démonstrateur pour exploration **hypothèse‑génératrice** — pas un avis clinique.
""")
