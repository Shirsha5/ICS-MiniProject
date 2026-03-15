# Stability and Reliability of Explainable AI (XAI) in ML-Based Network Intrusion Detection Systems

## Overview
This project evaluates the **stability and reliability** of Explainable AI methods (SHAP, LIME) applied to machine-learning-based Network Intrusion Detection Systems (NIDS). It compares explanation stability across multiple ML models, datasets, noise levels, and stability metrics.

## Datasets
| Dataset | Records | Features | Source |
|---------|---------|----------|--------|
| UNSW-NB15 | 175,341 | 42 | UNSW Sydney |
| NSL-KDD | 125,973 + 22,544 | 41 | UNB Canada |

## ML Models
- Random Forest
- XGBoost
- Decision Tree
- Gradient Boosting

## XAI Methods
- **SHAP** (SHapley Additive exPlanations) — TreeExplainer
- **LIME** (Local Interpretable Model-Agnostic Explanations)

## Stability Metrics
- Cosine Similarity
- Jaccard Similarity (Top-K feature overlap)
- Spearman Rank Correlation
- Feature Sign Agreement

## Results

### Model Performance Comparison
![Model Performance](results/compare_model_performance.png)

### F1-Score Heatmap
![F1 Heatmap](results/compare_f1_heatmap.png)

### Training Time Comparison
![Training Time](results/compare_training_time.png)

### SHAP vs LIME Feature Agreement
![XAI Agreement](results/compare_xai_agreement.png)

### XAI Runtime Comparison
![XAI Runtime](results/compare_xai_runtime.png)

### Explanation Stability vs Perturbation Noise
![Stability All](results/compare_stability_all.png)

### Stability Heatmap (SHAP vs LIME at noise=0.02)
![Stability Heatmap](results/compare_stability_heatmap.png)

### Multi-Metric Stability Comparison
![Stability Metrics](results/compare_stability_metrics.png)

## Key Findings
1. **SHAP is significantly more stable than LIME** — cosine similarity ~0.7 vs ~0.3 under perturbation
2. **Stability degrades with noise** for both methods, as expected
3. **GradientBoosting** has the most stable SHAP explanations on UNSW-NB15 (cos=0.83)
4. **LIME shows near-zero or negative correlations** on NSL-KDD — a critical reliability concern
5. **Correct predictions** tend to have more stable explanations than incorrect ones

## Project Structure
```
├── datasets/               # UNSW-NB15 and NSL-KDD CSVs (gitignored)
├── models/                 # Trained model .pkl files (gitignored)
├── results/                # All output tables, plots, figures
├── preprocessing.py        # Data loading, encoding, scaling
├── download_datasets.py    # Download datasets
├── Full_Analysis.ipynb     # Complete analysis notebook
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python download_datasets.py
```

Then open `Full_Analysis.ipynb` in Jupyter or VS Code and **Run All**.
