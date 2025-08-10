import streamlit as st
import numpy as np, pandas as pd, joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import streamlit.components.v1 as components
try:
    import py3Dmol
    import stmol
    HAVE_3D = True
except ImportError:
    HAVE_3D = False
    st.warning("⚠️ Pour la visualisation 3D, installez: pip install py3Dmol stmol")
from src.featurization.fingerprints import morgan_fp
from src.featurization.ood import ood_composite, tanimoto_sim_matrix, density_score

st.set_page_config(page_title="🎯 TargetBridge Advanced", layout="wide")
st.title("🎯 TargetBridge Advanced — Multi-Model Drug Discovery Platform")
st.write("🚀 **Advanced ML Platform** with **Gaussian Process variance**, **Conformalized Quantile Regression**, and **hybrid uncertainty quantification**.")

# 3D capabilities info
if HAVE_3D:
    st.success("🌐 **Visualisation 3D activée** - Modèles moléculaires interactifs disponibles")
else:
    st.info("💡 **Pour activer la 3D**: `pip install py3Dmol stmol` puis redémarrez l'application")

# Sidebar
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.selectbox("Ranking mode", ["Prudent (μ)", "Explorateur (UCB-GP)", "Explorateur (UCB-width)"])
beta = st.sidebar.slider("UCB β (exploration)", 0.0, 2.0, 0.6, 0.1)
abstain_w = st.sidebar.slider("Seuil largeur d'intervalle pour abstention", 0.5, 3.0, 1.2, 0.1)

# Model selection
model_type = st.sidebar.selectbox("Model Type", ["KRR+Conformal", "CQR-HGBR", "Ensemble Average"])

# Visualization options
st.sidebar.header("🎨 Visualisation")
show_3d = st.sidebar.checkbox("Visualisation 3D", value=HAVE_3D, disabled=not HAVE_3D)
show_molecular_gallery = st.sidebar.checkbox("Galerie moléculaire 3D", value=False, disabled=not HAVE_3D)

art_dir = Path("artifacts")
targets = sorted([p.name for p in art_dir.iterdir() if p.is_dir()]) if art_dir.exists() else []
use_targets = st.sidebar.multiselect("Targets", options=targets, default=targets)

def load_pack(target, model_type):
    """Load model pack based on type."""
    if model_type == "CQR-HGBR":
        pack_path = art_dir / target / "model_cqr.joblib"
        if pack_path.exists():
            return joblib.load(pack_path)
    
    # Default to KRR model
    pack_path = art_dir / target / "model.joblib"
    if pack_path.exists():
        return joblib.load(pack_path)
    return None

def depict(smi):
    """Generate 2D molecular depiction."""
    try:
        mol = Chem.MolFromSmiles(smi)
        return Draw.MolToImage(mol, size=(260, 180)) if mol else None
    except:
        return None

def depict_3d(smiles, height=400, width=400):
    """Generate interactive 3D molecular visualization."""
    if not HAVE_3D:
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Convert to SDF format for 3D visualization
        mol_block = Chem.MolToMolBlock(mol)
        
        # Create 3D viewer
        xyzview = py3Dmol.view(width=width, height=height)
        xyzview.addModel(mol_block, 'sdf')
        xyzview.setStyle({'stick': {'colorscheme': 'default'}})
        xyzview.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'lightblue'})
        xyzview.zoomTo()
        
        return xyzview
        
    except Exception as e:
        st.warning(f"Erreur 3D: {str(e)}")
        return None

def compare_molecules_3d(smiles_list, labels=None):
    """Compare multiple molecules in 3D side by side."""
    if not HAVE_3D or not smiles_list:
        return None
        
    try:
        # Create grid view for multiple molecules
        viewer = py3Dmol.view(width=800, height=400)
        
        for i, smiles in enumerate(smiles_list[:4]):  # Max 4 molecules
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42 + i)
            AllChem.MMFFOptimizeMolecule(mol)
            
            mol_block = Chem.MolToMolBlock(mol)
            
            # Position molecules in grid
            x_offset = (i % 2) * 10
            y_offset = (i // 2) * 10
            
            viewer.addModel(mol_block, 'sdf')
            viewer.setStyle({'model': i}, {'stick': {'colorscheme': f'C{i}'}})
            viewer.translate(x_offset, y_offset, 0, {'model': i})
            
        viewer.zoomTo()
        return viewer
        
    except Exception as e:
        st.warning(f"Erreur comparaison 3D: {str(e)}")
        return None

def enhanced_molecular_depiction(smiles, title="Molecule"):
    """Enhanced molecular visualization with both 2D and 3D."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Structure 2D")
        img_2d = depict(smiles)
        if img_2d:
            st.image(img_2d, caption=f"2D: {smiles[:50]}...")
        else:
            st.error("Impossible de générer la structure 2D")
    
    with col2:
        st.subheader("🌐 Modèle 3D Interactif")
        if HAVE_3D:
            viewer_3d = depict_3d(smiles)
            if viewer_3d:
                stmol.showmol(viewer_3d, height=400, width=400)
            else:
                st.warning("Impossible de générer le modèle 3D")
        else:
            st.info("📦 Installez py3Dmol et stmol pour la visualisation 3D")
            # Fallback to larger 2D image
            if img_2d:
                st.image(img_2d, width=400)

def explain_knn(x, pack, k=3):
    """k-NN explanation with Tanimoto similarity."""
    if "neighbors" not in pack:
        return []
    
    Xtr = pack["neighbors"]["X"].astype(bool)
    sims = tanimoto_sim_matrix(x.reshape(1,-1).astype(bool), Xtr).ravel()
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        out.append({
            "smiles": pack["neighbors"]["smiles"][i], 
            "pIC50": float(pack["neighbors"]["y"][i]), 
            "sim": float(sims[i])
        })
    return out

def predict_krr_conformal(smi, pack, mode, beta):
    """Predict with KRR + Adaptive Conformal + GP variance."""
    x = morgan_fp(smi, n_bits=pack["bits"], radius=pack["radius"])
    if x is None:
        return None
    
    # Conformal prediction
    model = pack["model"]
    
    # Get conformal interval
    if hasattr(model, 'predict_interval'):
        # Direct conformal model
        mu_c, lo_c, hi_c = model.predict_interval(x.reshape(1,-1))
        mu_c, lo_c, hi_c = mu_c[0], lo_c[0], hi_c[0]
    else:
        # Reconstruct conformal prediction
        mu_c = model.predict(x.reshape(1,-1))[0]
        # Simplified interval (fallback)
        dens = density_score(x, pack["X_train"], k=8)
        s = 1.0 + pack.get("gamma", 1.5) * (1.0 - dens)
        w = pack["qhat"] * s
        lo_c, hi_c = mu_c - w, mu_c + w
    
    # GP variance if available
    sigma_gp = None
    if hasattr(model, 'predict_mean_var') and hasattr(model, 'L_'):
        try:
            mu_gp, var_gp = model.predict_mean_var(x.reshape(1,-1))
            sigma_gp = float(np.sqrt(var_gp[0]))
        except Exception as e:
            # Fallback if GP variance calculation fails
            sigma_gp = None
    
    # If no GP variance, use interval width as uncertainty proxy
    if sigma_gp is None:
        dens = density_score(x, pack["X_train"], k=8)
        s = 1.0 + pack.get("gamma", 1.5) * (1.0 - dens)
        sigma_gp = pack.get("qhat", 1.0) * s * 0.5  # Approximate uncertainty
    
    # OOD detection
    ood = ood_composite(x, pack["X_train"], w_novelty=0.6, k=8)
    in_domain = ood < pack["ood_tau"]
    
    # Ranking score
    if mode.startswith("Prudent"):
        score = mu_c
    elif mode == "Explorateur (UCB-GP)" and sigma_gp is not None:
        score = mu_c + beta * sigma_gp
    else:  # UCB-width fallback
        score = mu_c + beta * (hi_c - mu_c)
    
    knn = explain_knn(x, pack, k=3)
    
    result = {
        'mu': float(mu_c), 
        'lo': float(lo_c), 
        'hi': float(hi_c), 
        'width': float(hi_c - lo_c),
        'ood': float(ood), 
        'in_domain': bool(in_domain), 
        'score': float(score), 
        'knn': knn
    }
    
    if sigma_gp is not None:
        result['sigma_gp'] = sigma_gp
        
    return result

def predict_cqr(smi, pack, mode, beta):
    """Predict with Conformalized Quantile Regression."""
    x = morgan_fp(smi, n_bits=pack["bits"], radius=pack["radius"])
    if x is None:
        return None
    
    # CQR prediction
    lq_model = pack["model"]["lq"]
    uq_model = pack["model"]["uq"]
    
    x_float = x.reshape(1,-1).astype(float)
    lo_raw = lq_model.predict(x_float)[0]
    hi_raw = uq_model.predict(x_float)[0]
    
    # Apply conformal corrections
    lo = lo_raw - pack["qhat_lo"]
    hi = hi_raw + pack["qhat_hi"]
    mu = 0.5 * (lo + hi)
    
    # OOD detection
    ood = ood_composite(x, pack["X_train"], w_novelty=0.6, k=8)
    in_domain = ood < pack["ood_tau"]
    
    # Ranking score
    if mode.startswith("Prudent"):
        score = mu
    else:  # UCB based on interval width
        score = mu + beta * (hi - mu)
    
    knn = explain_knn(x, pack, k=3)
    
    return {
        'mu': float(mu), 
        'lo': float(lo), 
        'hi': float(hi), 
        'width': float(hi - lo),
        'ood': float(ood), 
        'in_domain': bool(in_domain), 
        'score': float(score), 
        'knn': knn,
        'method': 'CQR'
    }

def predict_ensemble(smi, packs, mode, beta):
    """Ensemble prediction combining available models."""
    predictions = []
    
    for target, pack in packs.items():
        if pack is None:
            continue
            
        # Try both KRR and CQR predictions
        pred_krr = predict_krr_conformal(smi, pack, mode, beta)
        if pred_krr:
            predictions.append(pred_krr)
    
    if not predictions:
        return None
    
    # Simple averaging ensemble
    mu_avg = np.mean([p['mu'] for p in predictions])
    lo_avg = np.mean([p['lo'] for p in predictions])
    hi_avg = np.mean([p['hi'] for p in predictions])
    ood_avg = np.mean([p['ood'] for p in predictions])
    
    # Use most conservative in_domain decision
    in_domain = all(p['in_domain'] for p in predictions)
    
    score = mu_avg + beta * (hi_avg - mu_avg) if not mode.startswith("Prudent") else mu_avg
    
    return {
        'mu': float(mu_avg),
        'lo': float(lo_avg), 
        'hi': float(hi_avg),
        'width': float(hi_avg - lo_avg),
        'ood': float(ood_avg),
        'in_domain': bool(in_domain),
        'score': float(score),
        'knn': predictions[0]['knn'],  # Use first model's kNN
        'method': 'Ensemble',
        'n_models': len(predictions)
    }

def predict_one(smi, packs, mode, beta, model_type):
    """Main prediction function routing to appropriate method."""
    if model_type == "Ensemble Average":
        return predict_ensemble(smi, packs, mode, beta)
    
    # Single model prediction
    for target, pack in packs.items():
        if pack is None:
            continue
            
        if model_type == "CQR-HGBR" and pack.get("type") == "CQR_HGBR":
            return predict_cqr(smi, pack, mode, beta)
        elif model_type == "KRR+Conformal" and "KRR" in pack.get("type", ""):
            return predict_krr_conformal(smi, pack, mode, beta)
    
    return None

# Inputs
colL, colR = st.columns([1,1])
with colL:
    smiles = st.text_area("SMILES", "CC1=CC=C(C=C1)C(C(=O)O)N")  # ibuprofen
with colR:
    up = st.file_uploader("...ou charge un CSV (colonne 'smiles')", type=["csv"])

btn = st.button("🚀 Predict")

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
            packs = {t: load_pack(t, model_type) for t in use_targets}
            
            for smi in df_in["smiles"].tolist():
                for t in use_targets:
                    pack = packs[t]
                    if pack is None:
                        continue
                        
                    pred = predict_one(smi, {t: pack}, mode, beta, model_type)
                    if pred is None:
                        continue
                        
                    abstain = (not pred["in_domain"]) or (pred["width"] >= abstain_w)
                    row = {
                        "smiles": smi, 
                        "target": t, 
                        "method": pred.get("method", model_type),
                        **{k: v for k, v in pred.items() if k not in ["knn", "method"]}, 
                        "abstain": bool(abstain)
                    }
                    rows.append(row)
                    
            if not rows:
                st.error("Aucune prédiction produite.")
            else:
                res = pd.DataFrame(rows)
                # Tri par score desc au sein de chaque SMILES
                res = res.sort_values(by=["smiles","score"], ascending=[True, False]).reset_index(drop=True)
                st.success(f"✅ {len(res)} prédictions avec {model_type}")
                
                # Display results with method info
                display_cols = ["smiles","target","method","mu","lo","hi","width","ood","in_domain","abstain","score"]
                if "sigma_gp" in res.columns:
                    display_cols.insert(-1, "sigma_gp")
                
                st.dataframe(res[display_cols], use_container_width=True)

                # Carte explicative pour le premier SMILES + meilleure cible
                s0_idx = 0
                s0 = res.iloc[s0_idx]
                
                st.subheader("🔎 Explication Détaillée")
                
                # Molecular analysis with input molecule and neighbors
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.markdown("### 🎯 Molécule d'entrée")
                    # Show input molecule once here
                    enhanced_molecular_depiction(s0["smiles"], f"Cible: {s0['target']}")
                    
                    st.markdown("### 📊 Prédictions")
                    st.markdown(f"**Cible**: `{s0['target']}`  |  **Méthode**: `{s0['method']}`")
                    st.markdown(f"**μ** ≈ {s0['mu']:.2f}  |  **Intervalle**: [{s0['lo']:.2f}, {s0['hi']:.2f}]")
                    
                    if "sigma_gp" in s0:
                        st.markdown(f"**σ_GP** ≈ {s0['sigma_gp']:.3f} | **OOD** ≈ {s0['ood']:.2f}")
                    else:
                        st.markdown(f"**OOD** ≈ {s0['ood']:.2f}")
                        
                    decision_text = "✅ *Prédire*" if not s0["abstain"] else "🟡 *S'abstenir*"
                    st.markdown(f"**Décision**: {decision_text}")
                    
                    # Method-specific info
                    if s0["method"] == "CQR":
                        st.info("📊 **CQR**: Régression quantile conformalisée")
                    elif "GP" in str(s0.get("sigma_gp", "")):
                        st.info("🧠 **KRR-GP**: Variance analytique Gaussienne")
                    elif s0["method"] == "Ensemble":
                        st.info(f"🤖 **Ensemble**: Moyenne de {s0.get('n_models', 'N')} modèles")
                
                with col2:
                    st.markdown("### 🔍 Voisins similaires")
                    
                    # Get kNN from original prediction
                    original_pred = predict_one(s0["smiles"], {s0["target"]: packs[s0["target"]]}, mode, beta, model_type)
                    if original_pred and "knn" in original_pred:
                        neighbor_smiles = []
                        for i, v in enumerate(original_pred["knn"]):
                            st.markdown(f"**Voisin {i+1}** - Sim: {v['sim']:.2f} | pIC50: {v['pIC50']:.2f}")
                            
                            # Visual representation of neighbor
                            if HAVE_3D and show_3d:
                                viewer_neighbor = depict_3d(v['smiles'], height=200, width=250)
                                if viewer_neighbor:
                                    stmol.showmol(viewer_neighbor, height=200, width=250)
                                else:
                                    img_neighbor = depict(v['smiles'])
                                    if img_neighbor:
                                        st.image(img_neighbor, width=250)
                            else:
                                img_neighbor = depict(v['smiles'])
                                if img_neighbor:
                                    st.image(img_neighbor, width=250)
                            
                            st.code(f"{v['smiles'][:50]}{'...' if len(v['smiles']) > 50 else ''}", language=None)
                            neighbor_smiles.append(v['smiles'])
                            st.markdown("---")
                
                with col3:
                    st.markdown("### 🌐 Comparaison 3D")
                    
                    if HAVE_3D and len(neighbor_smiles) > 0:
                        molecules_to_compare = [s0["smiles"]] + neighbor_smiles[:2]  # Input + 2 neighbors
                        labels = ["Entrée"] + [f"Voisin {i+1}" for i in range(len(neighbor_smiles[:2]))]
                        
                        viewer_comparison = compare_molecules_3d(molecules_to_compare, labels)
                        if viewer_comparison:
                            stmol.showmol(viewer_comparison, height=350, width=300)
                        else:
                            st.info("Comparaison 3D indisponible")
                    else:
                        st.info("📦 Installez py3Dmol et stmol pour la visualisation 3D")
                
                # Molecular gallery for other predictions (excluding input molecule)
                if show_molecular_gallery and HAVE_3D and len(res) > 1:
                    st.subheader("🖼️ Autres Molécules Prédites")
                    st.write("Autres molécules du dataset avec leurs scores de prédiction")
                    
                    # Get unique molecules excluding the input molecule
                    other_molecules = res[res["smiles"] != s0["smiles"]].drop_duplicates(subset=['smiles']).head(4)
                    
                    if len(other_molecules) > 0:
                        # Create tabs for each other molecule
                        molecule_tabs = st.tabs([f"Mol {i+1} (Score: {row['score']:.1f})" 
                                               for i, (_, row) in enumerate(other_molecules.iterrows())])
                        
                        for tab, (_, mol_data) in zip(molecule_tabs, other_molecules.iterrows()):
                            with tab:
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    if show_3d:
                                        viewer_3d = depict_3d(mol_data["smiles"], height=300, width=350)
                                        if viewer_3d:
                                            stmol.showmol(viewer_3d, height=300, width=350)
                                        else:
                                            img_2d = depict(mol_data["smiles"])
                                            if img_2d:
                                                st.image(img_2d)
                                    else:
                                        img_2d = depict(mol_data["smiles"])
                                        if img_2d:
                                            st.image(img_2d)
                                
                                with col2:
                                    st.markdown(f"**SMILES**: `{mol_data['smiles'][:60]}{'...' if len(mol_data['smiles']) > 60 else ''}`")
                                    st.markdown(f"**Cible**: {mol_data['target']}")
                                    st.markdown(f"**Score**: {mol_data['score']:.2f}")
                                    st.markdown(f"**μ**: {mol_data['mu']:.2f}")
                                    st.markdown(f"**Intervalle**: [{mol_data['lo']:.2f}, {mol_data['hi']:.2f}]")
                                    st.markdown(f"**OOD**: {mol_data['ood']:.3f}")
                                    
                                    # Color-coded decision
                                    if mol_data['abstain']:
                                        st.error("🟡 S'abstenir")
                                    else:
                                        st.success("✅ Prédire")
                    else:
                        st.info("Aucune autre molécule à afficher")
                
                # Advanced metrics display
                if len(res) > 1:
                    st.subheader("📈 Analyse Multi-Cibles")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Score Max", f"{res['score'].max():.2f}")
                    with col2:
                        st.metric("Largeur Moy.", f"{res['width'].mean():.2f}")
                    with col3:
                        st.metric("In-Domain", f"{(res['in_domain']).sum()}/{len(res)}")
                    with col4:
                        st.metric("Abstentions", f"{res['abstain'].sum()}/{len(res)}")
                
                st.download_button(
                    "⬇️ Export CSV", 
                    data=res.to_csv(index=False), 
                    file_name=f"targetbridge_{model_type.lower().replace(' ', '_')}_predictions.csv", 
                    mime="text/csv"
                )
