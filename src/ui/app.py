import streamlit as st
import numpy as np, pandas as pd, joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from src.featurization.fingerprints import morgan_fp
from src.featurization.ood import ood_composite, tanimoto_sim_matrix

st.set_page_config(page_title="🎯 TargetBridge (Lite)", layout="wide")
st.title("🎯 TargetBridge (Lite) — CPU-only demo")
st.write("Mini outil de **classement de cibles** avec **incertitude adaptative**, **détection OOD** et **explication par voisins**.")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Ranking mode", ["Prudent (μ)", "Explorateur (UCB)"])
beta = st.sidebar.slider("UCB β (exploration)", 0.0, 2.0, 0.6, 0.1)
abstain_w = st.sidebar.slider("Seuil largeur d'intervalle pour abstention", 0.5, 3.0, 1.2, 0.1)

art_dir = Path("artifacts")
targets = sorted([p.name for p in art_dir.iterdir() if p.is_dir()]) if art_dir.exists() else []
use_targets = st.sidebar.multiselect("Targets", options=targets, default=targets)

def load_pack(t):
    return joblib.load(art_dir / t / "model.joblib")

def depict(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Draw.MolToImage(mol, size=(260, 180)) if mol else None
    except:
        return None

def explain_knn(x, pack, k=3):
    Xtr = pack["neighbors"]["X"].astype(bool)
    sims = tanimoto_sim_matrix(x.reshape(1,-1).astype(bool), Xtr).ravel()
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        out.append({"smiles": pack["neighbors"]["smiles"][i], "pIC50": float(pack["neighbors"]["y"][i]), "sim": float(sims[i])})
    return out

def predict_one(smi, pack, mode, beta):
    x = morgan_fp(smi, n_bits=pack["bits"], radius=pack["radius"])
    if x is None:
        return None
    # μ, intervalle (conformal adaptatif)
    mu = pack["model"].predict(x.reshape(1,-1))[0]
    # Recompute width via qhat & density (same formula as in training)
    from src.featurization.ood import density_score
    dens = density_score(x, pack["X_train"], k=8)
    s = 1.0 + pack.get("gamma", 1.5) * (1.0 - dens)
    w = pack["qhat"] * s
    lo, hi = mu - w, mu + w

    # OOD composite
    ood = ood_composite(x, pack["X_train"], w_novelty=0.6, k=8)
    in_domain = ood < pack["ood_tau"]

    # Score ranking
    score = mu if mode.startswith("Prudent") else (mu + beta * (hi - mu))
    knn = explain_knn(x, pack, k=3)
    return dict(mu=float(mu), lo=float(lo), hi=float(hi), width=float(hi-lo), ood=float(ood), in_domain=bool(in_domain), score=float(score), knn=knn)

# Inputs
colL, colR = st.columns([1,1])
with colL:
    smiles = st.text_area("SMILES", "CC1=CC=C(C=C1)C(C(=O)O)N")  # ibuprofen
with colR:
    up = st.file_uploader("...ou charge un CSV (colonne 'smiles')", type=["csv"])

btn = st.button("Predict")

if btn:
    if not use_targets:
        st.warning("Aucun modèle trouvé dans artifacts/. Entraîne d'abord quelques cibles.")
    else:
        df_in = None
        if up is not None:
            df = pd.read_csv(up)
            if "smiles" in df.columns:
                df_in = df[["smiles"]].dropna()
        else:
            df_in = pd.DataFrame({"smiles":[smiles]})

        if df_in is None or df_in.empty:
            st.error("Aucun SMILES valide.")
        else:
            rows = []
            packs = {t: load_pack(t) for t in use_targets}
            for smi in df_in["smiles"].tolist():
                for t, pack in packs.items():
                    pred = predict_one(smi, pack, mode, beta)
                    if pred is None:
                        continue
                    abstain = (not pred["in_domain"]) or (pred["width"] >= abstain_w)
                    rows.append({
                        "smiles": smi, "target": t, **pred, "abstain": bool(abstain)
                    })
            if not rows:
                st.error("Aucune prédiction produite.")
            else:
                res = pd.DataFrame(rows)
                # Tri par score desc au sein de chaque SMILES
                res = res.sort_values(by=["smiles","score"], ascending=[True, False]).reset_index(drop=True)
                st.success(f"OK — {len(res)} lignes")
                st.dataframe(res[["smiles","target","mu","lo","hi","width","ood","in_domain","abstain","score"]], use_container_width=True)

                # Carte explicative pour le premier SMILES + meilleure cible
                s0 = res.iloc[0]
                st.subheader("🔎 Explication")
                c1, c2 = st.columns([1,1])
                with c1:
                    img = depict(s0["smiles"])
                    if img: st.image(img, caption=s0["smiles"])
                    st.markdown(f"**Cible**: `{s0['target']}`  |  **μ**≈{s0['mu']:.2f}  |  **[{s0['lo']:.2f}, {s0['hi']:.2f}]**  |  **OOD**≈{s0['ood']:.2f}")
                    st.markdown("**Décision**: " + ("✅ *Prédire*" if not s0["abstain"] else "🟡 *S'abstenir*"))
                with c2:
                    st.markdown("**Voisins d'entraînement (k=3)**")
                    for v in s0["knn"]:
                        st.write(f"- sim≈{v['sim']:.2f} | pIC50≈{v['pIC50']:.2f} | `{v['smiles'][:80]}`")
                st.download_button("⬇️ Export CSV", data=res.to_csv(index=False), file_name="targetbridge_predictions.csv", mime="text/csv")
