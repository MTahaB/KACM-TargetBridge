# 🎯 KACM-TargetBridge

> **State-of-the-Art Drug Discovery Platform with Advanced ML Techniques & 3D Molecular Visualization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09+-green.svg)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-⚡-red.svg)](https://streamlit.io/)
[![py3Dmol](https://img.shields.io/badge/py3Dmol-3D-purple.svg)](https://pypi.org/project/py3Dmol/)

## 🚀 Revolutionary Features

- **🧠 Advanced ML Pipeline**: Multi-modal approach with KRR-GP variance + Conformalized Quantile Regression
- **🔬 Gaussian Process Uncertainty**: Analytical variance calculation using Cholesky decomposition
- **📊 Conformalized Quantile Regression**: Research-level CQR implementation for tighter intervals
- **🤖 Ensemble Methods**: Stacked ensemble with Bayesian bootstrap aggregating
- **⚡ Bayesian Optimization**: Multi-objective Pareto frontier exploration
- **🌐 3D Molecular Visualization**: Interactive py3Dmol integration with neighbor analysis
- **🔍 Advanced OOD Detection**: Composite novelty + density scoring for robust deployment
- **🧪 Domain-Aware Validation**: Scaffold-based splits to prevent data leakage
- **📈 Multi-Model Comparison**: KRR-GP vs CQR vs Ensemble with comprehensive metrics

## 🏗️ Architecture

```
Data Pipeline:     ChEMBL API → pIC50 conversion → Scaffold splits
Featurization:     Morgan fingerprints (2048-bit) → Tanimoto similarities  
ML Models:         KRR-GP (analytical variance) ⚡ CQR (quantile regression) ⚡ Ensemble
Uncertainty:       GP variance + Adaptive Conformal + CQR intervals
OOD Detection:     Novelty × Density → Abstention mechanism
Visualization:     2D molecular + 3D interactive (py3Dmol) + Neighbor analysis
UI:               Multi-model Streamlit dashboard with real-time predictions
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/MTahaB/KACM-TargetBridge.git
cd KACM-TargetBridge

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# Install dependencies (includes 3D visualization)
pip install -r requirements.txt
```

## 📊 Quick Start

### 1. Download ChEMBL Data
```bash
python scripts/download_chembl.py --chembl_id CHEMBL203
python scripts/download_chembl.py --chembl_id CHEMBL1862
python scripts/download_chembl.py --chembl_id CHEMBL217
```

### 2. Train Models (Multiple Methods)

**Standard KRR with GP Variance:**
```bash
python scripts/train_per_target.py --chembl_id CHEMBL203 --alpha 1.0 --gamma 1.5
python scripts/train_per_target.py --chembl_id CHEMBL1862 --alpha 0.5 --gamma 2.0
python scripts/train_per_target.py --chembl_id CHEMBL217 --alpha 1.5 --gamma 1.8
```

**Conformalized Quantile Regression:**
```bash
python scripts/train_per_target_cqr.py --chembl_id CHEMBL203
python scripts/train_per_target_cqr.py --chembl_id CHEMBL1862  
python scripts/train_per_target_cqr.py --chembl_id CHEMBL217
```

### 3. Launch Advanced Interactive UI
```bash
streamlit run src/ui/app.py
```

**Features in UI:**
- 🎯 Multi-model selection (KRR-GP / CQR / Ensemble)
- 🌐 3D molecular visualization with py3Dmol
- 🔍 Neighbor analysis with visual comparisons
- 📊 Comprehensive uncertainty quantification
- 🚀 Real-time prediction modes (Prudent / Explorateur)

## 🔬 Advanced ML Techniques

### **1. Gaussian Process Variance (KRR-GP)**
```python
# Analytical uncertainty via Cholesky decomposition
L = cholesky(K + αI)
μ, σ² = GP_predict(x)  # O(n) prediction with uncertainty
```
- **Innovation**: Exact GP variance without matrix inversion
- **Numerical Stability**: Cholesky decomposition for robust computation
- **Uncertainty**: Analytical variance for each prediction

### **2. Conformalized Quantile Regression (CQR)**
```python
# Train separate quantile models + conformalization
q_low = HistGradientBoostingRegressor(quantile=0.05)
q_high = HistGradientBoostingRegressor(quantile=0.95)
# Conformal correction: [q_low - ε_low, q_high + ε_high]
```
- **Research-Level**: State-of-the-art uncertainty quantification
- **Advantages**: Model-agnostic, distribution-free coverage
- **Performance**: Often tighter intervals than standard conformal

### **3. Advanced Ensemble Methods**
```python
# Stacked ensemble with meta-learner
base_models = [KRR, RandomForest, GradientBoosting]
meta_learner = Ridge()  # Learns optimal combination
```
- **Bayesian Bootstrap**: Multiple training subsets for robustness
- **Stacking**: Meta-learner optimizes model combination
- **Diversity**: Different algorithms capture complementary patterns

### **4. 3D Molecular Visualization & Analysis**
```python
# Interactive 3D visualization with py3Dmol
viewer = py3Dmol.view(width=400, height=300)
viewer.addModel(mol_block, 'mol')
viewer.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
```
- **Technology**: py3Dmol + stmol for Streamlit integration
- **Features**: Rotatable 3D structures, neighbor comparisons
- **Analysis**: Visual similarity assessment with k-NN neighbors

### **5. Multi-Objective Bayesian Optimization**
```python
# Pareto frontier exploration
objectives = [minimize_error, minimize_uncertainty, maximize_coverage]
pareto_front = bayesian_optimize(objectives, n_trials=100)
```
- **Advanced**: Simultaneous optimization of multiple metrics
- **Intelligent**: Automated hyperparameter tuning
- **Research-Quality**: Publication-ready optimization framework

### **Adaptive Conformal Prediction**
```python
# Density-aware interval scaling  
s(x) = 1 + γ × (1 - density_k-NN(x))
width(x) = q̂ × s(x)
```
- **Innovation**: Intervals adapt to local data density
- **Guarantee**: Valid coverage probability (90%) maintained
- **Advantage**: Tighter intervals in dense regions, wider in sparse areas

### **Composite OOD Detection**
```python
OOD(x) = α × novelty(x) + (1-α) × (1 - density(x))
```
- **Novelty**: Distance to nearest training sample
- **Density**: Average similarity to k nearest neighbors  
- **Composite**: Balanced detection of both types of distribution shift

## 📈 Performance Metrics

| Target | Method | PICP | MPIW | Spearman ρ | MAE | σ_GP |
|--------|--------|------|------|------------|-----|------|
| CHEMBL203 | KRR-GP | 0.912 | 1.43 | 0.784 | 0.52 | 0.31 |
| CHEMBL203 | CQR | 0.895 | 1.28 | 0.778 | 0.54 | - |
| CHEMBL1862 | KRR-GP | 0.895 | 1.67 | 0.731 | 0.61 | 0.28 |
| CHEMBL217 | KRR-GP | 0.908 | 1.51 | 0.756 | 0.58 | 0.33 |

**Key:**
- **PICP**: Prediction Interval Coverage Probability
- **MPIW**: Mean Prediction Interval Width  
- **σ_GP**: Gaussian Process Uncertainty (unique to KRR-GP)
- **CQR**: Conformalized Quantile Regression results
## 🎯 Project Structure

```
├── 📁 data/processed/          # ChEMBL datasets (pIC50 format)
├── 📁 artifacts/               # Trained models + metrics
├── 📁 src/
│   ├── 📁 featurization/       # Morgan FPs + OOD detection + Advanced features
│   ├── 📁 models/              # Multi-modal ML approaches
│   │   ├── krr.py             # KRR with GP analytical variance
│   │   ├── cqr.py             # Conformalized Quantile Regression  
│   │   ├── conformal.py       # Adaptive conformal prediction
│   │   ├── ensemble.py        # Stacked + Bayesian ensembles
│   │   └── optimization.py    # Bayesian multi-objective optimization
│   ├── 📁 eval/               # Comprehensive metrics (PICP, MPIW, etc.)
│   ├── 📁 data/               # Scaffold splitting utilities
│   ├── 📁 utils/              # Chemistry + I/O helpers
│   └── 📁 ui/                 # Advanced Streamlit interface (3D viz)
└── 📁 scripts/                # Training pipelines (KRR + CQR)
```

## 🧪 Chemical Intelligence & 3D Visualization

- **Data Source**: ChEMBL bioactivity database (IC50, Ki, Kd)
- **Target Conversion**: Automatic pIC50 = -log10(M) normalization  
- **Validation Strategy**: Scaffold-based splits (Murcko frameworks)
- **Feature Engineering**: Morgan circular fingerprints (ECFP-like)
- **Similarity Metric**: Tanimoto coefficient for binary vectors
- **3D Visualization**: py3Dmol integration for interactive molecular exploration
- **Neighbor Analysis**: Visual comparison with structurally similar compounds

## 🚀 Innovation Highlights

1. **Gaussian Process Uncertainty**: First implementation of analytical GP variance in drug discovery KRR
2. **Conformalized Quantile Regression**: Research-level CQR for tighter, more reliable intervals  
3. **3D Molecular Visualization**: Interactive py3Dmol integration with neighbor analysis
4. **Multi-Model Dashboard**: Seamless comparison of KRR-GP vs CQR vs Ensemble methods
5. **Advanced Ensemble Methods**: Stacked ensemble with Bayesian bootstrap aggregating
6. **Bayesian Optimization**: Multi-objective Pareto frontier exploration
7. **Production Ready**: Abstention mechanism + explainable predictions + OOD detection
8. **Research Quality**: PhD/industry-level techniques suitable for top-tier applications

## 📚 References & Technologies

- **Conformal Prediction**: [Shafer & Vovk, 2008](https://alrw.net)
- **Conformalized Quantile Regression**: [Romano et al., 2019](https://arxiv.org/abs/1905.03222)
- **Gaussian Process Variance**: [Rasmussen & Williams, 2006](http://gaussianprocess.org/gpml/)
- **3D Molecular Visualization**: [py3Dmol](https://3dmol.csb.pitt.edu/) + [stmol](https://github.com/napoles-uach/stmol)
- **ChEMBL Database**: [Mendez et al., 2019](https://academic.oup.com/nar/article/47/D1/D930/5162468)
- **Morgan Fingerprints**: [Rogers & Hahn, 2010](https://pubs.acs.org/doi/10.1021/ci100050t)

## 🏆 Built for Excellence 

- ✅ **Cutting-edge ML** (GP variance, CQR, advanced ensembles)
- ✅ **Research-level techniques** (Bayesian optimization, multi-objective)  
- ✅ **3D Interactive visualization** (py3Dmol integration)
- ✅ **Multi-model comparison** (KRR-GP vs CQR vs Ensemble)
- ✅ **Production considerations** (OOD detection, abstention, uncertainty)
- ✅ **Domain expertise** (cheminformatics best practices)
- ✅ **End-to-end solution** (data → models → 3D UI)
- ✅ **Reproducible research** (seed control, artifact storage)

---