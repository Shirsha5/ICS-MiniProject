# Stability and Reliability of Explainable AI (XAI) in ML-Based Network Intrusion Detection Systems

## Overview
This project **evaluates** (demonstrates with evidence) the **stability** and **reliability** of Explainable AI (SHAP, LIME) on ML-based NIDS across models, datasets, and noise levels. It does not claim mathematical “proof” in the formal sense; the report should phrase findings as *we evaluate / we show / evidence suggests* unless you add formal proofs.

---

## Do we have “explanation similarity under perturbation” + statistical rigour?

| What | Where |
|------|--------|
| **Stability = similarity of explanations before vs after perturbation** | **Full_Analysis.ipynb** — Gaussian noise on inputs; cosine, Jaccard, Spearman, sign agreement for SHAP and LIME. |
| **Mean ± std (single run)** | **results/stability_results.csv** (from notebook). |
| **Multi-seed + 95% CI across runs** | **`python stability_multiseed.py`** → **results/stability_multiseed_results.csv** (mean ± CI of mean SHAP cosine per noise level, across seeds 42–44). |
| **Paired statistical tests** | Same script → **results/stability_statistical_tests.csv**: (1) paired *t*-test: SHAP cosine at noise 0.01 vs 0.02 (same samples); (2) paired *t*-test: SHAP vs LIME cosine at noise 0.02 (subset, slower). |

Run **Full_Analysis.ipynb** first (train models), then:

```bash
python stability_multiseed.py
python plot_extended_results.py   # optional summary figure for new components
```

---

## Do we need gradient-based XAI (Integrated Gradients, etc.)?

**No, not required** for the core claim. The project already compares two major explainers (SHAP, LIME) with stability metrics. Gradient-based methods are **optional** if you want an extra subsection (“model-specific vs model-agnostic explainers”).

---

## Do we need a “cross-dataset” analysis?

**You already have two datasets** (UNSW-NB15 and NSL-KDD) in the main notebook — that *is* cross-dataset comparison for model performance and stability. A separate “consistency score” table is **optional**; you can compare F1 and stability side-by-side from existing CSVs/plots.

---

## New scripts — results & visualisation

The **new** pipelines are **not** inside Full_Analysis.ipynb; they are separate scripts. Outputs:

| Script | CSV / other | Figures |
|--------|-------------|---------|
| `multiclass_attack_analysis.py` | per_attack_metrics_multiclass.csv, per_attack_top_shap_features.csv, multiclass_two_stage_comparison.csv | multiclass_confusion_matrix.png (if ≤15 classes) |
| `autoencoder_nids.py` | autoencoder_results.csv | autoencoder_roc.png |
| `ensemble_stacking.py` | ensemble_stacking_results.csv | — |
| `plot_extended_results.py` | — | **extended_components_summary.png** (bars for multiclass/two-stage, AE, stacking) |

Run the three analysis scripts first, then `plot_extended_results.py` for one combined figure.

---

## Datasets
| Dataset | Records | Features | Source |
|---------|---------|----------|--------|
| UNSW-NB15 | 175,341 | 42 | UNSW Sydney |
| NSL-KDD | 125,973 + 22,544 | 41 | UNB Canada |

## ML Models (binary NIDS)
Random Forest, XGBoost, Decision Tree, Gradient Boosting.

## Deep Learning (NSL-KDD)
MLP in **deep_learning_nids.py**; SHAP + LIME. See **results/** for confusion matrix, ROC, training loss, shap_dl_summary, lime_dl_explanation.

## XAI Methods
- **SHAP** (TreeExplainer for tree models)
- **LIME**

## Stability metrics (notebook)
Cosine similarity, Jaccard@5, Spearman, sign agreement; noise σ ∈ {0.01, 0.02, 0.05, 0.10}.

## Main figures (from Full_Analysis)
- compare_model_performance.png, compare_f1_heatmap.png, compare_training_time.png  
- compare_xai_agreement.png, compare_xai_runtime.png  
- compare_stability_all.png, compare_stability_heatmap.png, compare_stability_metrics.png  

## Key findings (example wording for report)
1. SHAP tends to be more stable than LIME under perturbation (see plots + optional *p*-values in stability_statistical_tests.csv).  
2. Stability drops as noise increases.  
3. LIME can show weak agreement on NSL-KDD — discuss as reliability limitation.  
4. Use **stability_multiseed** results to report uncertainty across runs.

---

## Project structure
```
├── datasets/
├── models/
├── results/                    # All plots + CSVs (including stability_multiseed, statistical_tests, extended)
├── ishita_xai/                 # Extended XAI (SVM, KNN, MLP)
├── preprocessing.py            # + load_nsl_kdd_multiclass, load_unsw_nb15_multiclass
├── download_datasets.py
├── deep_learning_nids.py
├── multiclass_attack_analysis.py
├── autoencoder_nids.py
├── ensemble_stacking.py
├── stability_multiseed.py      # Multi-seed CI + paired t-tests
├── plot_extended_results.py    # extended_components_summary.png
├── Full_Analysis.ipynb
├── requirements.txt
└── README.md
```

## Python version and setup

**Recommended: Python 3.11** (3.10–3.12 also supported). Python 3.14 works but some optional packages may differ.

Create a virtual environment and install dependencies:

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the entire project

Run these steps in order so the full pipeline (data → training → analysis scripts) completes:

1. **Install dependencies** (use the same Python/venv for all steps)
   ```bash
   pip install -r requirements.txt
   ```

2. **Download datasets**
   ```bash
   python download_datasets.py
   ```

3. **Train models** — open **Full_Analysis.ipynb** and **Run All**. This trains and saves the models used by the scripts below.

4. **Run the analysis scripts** (multiclass, autoencoder, stacking, stability, summary plot)
   ```bash
   python multiclass_attack_analysis.py
   python autoencoder_nids.py
   python ensemble_stacking.py
   python stability_multiseed.py
   python plot_extended_results.py
   ```

Results go to `results/`; models to `models/`.

---

## GPU and speed

| Component | GPU? | Why |
|-----------|------|-----|
| **Autoencoder** (`autoencoder_nids.py`) | **Yes** (optional) | If TensorFlow is installed and a GPU is available, the script uses a Keras autoencoder on GPU. Otherwise it falls back to PCA (CPU). Install `tensorflow` for GPU support (Python 3.10–3.12). |
| **XGBoost** (Full_Analysis.ipynb) | **Yes** (optional) | The notebook tries `device='cuda'` and `tree_method='hist'` for XGBoost; if CUDA is available, training uses the GPU. Otherwise CPU is used. |
| **SHAP** (TreeExplainer) | No | Tree SHAP is already fast on CPU; GPU explainers are for deep models, not tree models. |
| **LIME** (stability_multiseed.py) | **No** | LIME runs many local model evaluations per explanation; the library is CPU-only. The script uses a small subset (e.g. 10 samples) and fewer LIME samples per explanation (500) so the SHAP-vs-LIME paired test finishes in a few minutes. |
| **Random Forest / GradientBoosting** | No | sklearn implementations are CPU-only. |

To speed things up: install `tensorflow` (and CUDA/cuDNN if you have an NVIDIA GPU) for the autoencoder; use a machine with a GPU and CUDA-enabled XGBoost for faster XGBoost training in the notebook.
