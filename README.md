# Stability and Reliability of Explainable AI (XAI) in ML-Based Network Intrusion Detection Systems

## Overview
This project evaluates the **stability and reliability** of Explainable AI methods (SHAP, LIME) applied to machine-learning-based Network Intrusion Detection Systems (NIDS). It compares explanation stability across multiple ML models, datasets, noise levels, and stability metrics.

Additionally, a deep learning based intrusion detection model is implemented to explore the performance of neural networks on the NSL-KDD dataset.

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

## Deep Learning Model (MLP + Explainable AI)

In addition to traditional machine learning models, a deep learning-based Network Intrusion Detection System (NIDS) was implemented using a Multi-Layer Perceptron (MLP) on the NSL-KDD dataset.

### Architecture
Input Layer → 128 Dense (ReLU) → Dropout (0.3) →  
64 Dense (ReLU) → Dropout (0.3) →  
Output Layer (Sigmoid)

### Dataset Used
- NSL-KDD

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Deep Learning Results & Interpretability

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix_dl.png)

**Insight:**  
The model achieves strong classification performance with relatively low false positives and false negatives, indicating effective intrusion detection capability.

---

### Training Loss Curve
![Training Loss](results/training_loss_dl.png)

**Insight:**  
Training and validation loss decrease smoothly, suggesting stable learning with minimal overfitting.

---

### ROC-AUC Curve
![ROC Curve](results/dl_roc_curve.png)

**Insight:**  
The ROC curve shows strong separability between attack and normal traffic, with a high AUC score indicating robust classification performance.

---

### SHAP Feature Importance (Global Explainability)
![SHAP](results/shap_dl_summary.png)

**Insight:**  
SHAP provides consistent global feature importance. It highlights the most influential features driving predictions and demonstrates stable interpretability.

---

### LIME Explanation (Local Explainability)
![LIME](results/lime_dl_explanation.png)

**Insight:**  
LIME explains individual predictions effectively but is sensitive to small input perturbations. Compared to SHAP, it is less stable but useful for local interpretability.

---

### Key Observations (Deep Learning)
- MLP achieves competitive performance on the NSL-KDD dataset  
- ROC-AUC confirms strong classification capability  
- SHAP explanations are more stable and reliable  
- LIME provides local insights but lacks consistency  
- Deep learning complements traditional ML models in intrusion detection

## XAI Methods
- **SHAP** (SHapley Additive exPlanations) — TreeExplainer
- **LIME** (Local Interpretable Model-Agnostic Explanations)

## Stability Metrics
- Cosine Similarity
- Jaccard Similarity (Top-K feature overlap)
- Spearman Rank Correlation
- Feature Sign Agreement

## Machine Learning Results & Interpretability 

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

## Key Findings & Observations
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
├── deep_learning_nids.py   # Deep learning intrusion detection model
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
