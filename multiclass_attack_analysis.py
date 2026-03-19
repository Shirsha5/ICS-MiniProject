"""Attack-type analysis: multi-class RF, per-attack SHAP, two-stage (binary then multi-class). Uses NSL-KDD."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

# Use preprocessing for binary; load raw for multiclass
from preprocessing import NSL_KDD_COLUMNS, NSL_KDD_CAT_COLS, load_nsl_kdd_multiclass

np.random.seed(42)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Loading NSL-KDD (multiclass)...")
X_train, X_test, y_bin_tr, y_bin_te, y_att_tr, y_att_te, le_attack, scaler = load_nsl_kdd_multiclass()
feature_names = list(X_train.columns)
X_train_np = X_train.values
X_test_np = X_test.values
n_classes = len(le_attack.classes_)
print(f"Attack types: {le_attack.classes_}")
print(f"Train: {X_train_np.shape}, Test: {X_test_np.shape}, Classes: {n_classes}")

print("\n--- Multi-class RF with class_weight ---")
clf_multi = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf_multi.fit(X_train_np, y_att_tr)
y_pred_multi = clf_multi.predict(X_test_np)
print(classification_report(y_att_te, y_pred_multi, labels=np.arange(len(le_attack.classes_)), target_names=le_attack.classes_, zero_division=0))

per_class = classification_report(y_att_te, y_pred_multi, labels=np.arange(len(le_attack.classes_)), target_names=le_attack.classes_, output_dict=True, zero_division=0)
rows = []
for name in le_attack.classes_:
    if name in per_class and isinstance(per_class[name], dict):
        rows.append({
            "Dataset": "NSL-KDD",
            "AttackType": name,
            "Precision": per_class[name].get("precision", 0),
            "Recall": per_class[name].get("recall", 0),
            "F1": per_class[name].get("f1-score", 0),
        })
pd.DataFrame(rows).to_csv("results/per_attack_metrics_multiclass.csv", index=False)
print("Saved results/per_attack_metrics_multiclass.csv")

# Confusion matrix (subset of classes if too many)
cm = confusion_matrix(y_att_te, y_pred_multi)
if cm.shape[0] <= 15:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=le_attack.classes_, yticklabels=le_attack.classes_, annot=True, fmt="d", cmap="Blues")
    plt.title("Multi-class Confusion Matrix (NSL-KDD)")
    plt.tight_layout()
    plt.savefig("results/multiclass_confusion_matrix.png")
    plt.close()
    print("Saved results/multiclass_confusion_matrix.png")

print("\n--- Per-attack SHAP summary ---")
explainer_shap = shap.TreeExplainer(clf_multi)
sample_per_class = 50
X_explain = []
attack_ids_explain = []
for c in range(min(n_classes, 10)):  # limit to 10 classes
    idx = np.where(y_att_te == c)[0]
    if len(idx) >= 5:
        np.random.seed(42)
        pick = np.random.choice(idx, size=min(sample_per_class, len(idx)), replace=False)
        X_explain.append(X_test_np[pick])
        attack_ids_explain.extend([c] * len(pick))
if X_explain:
    X_explain = np.vstack(X_explain)
    attack_ids_explain = np.array(attack_ids_explain)
    shap_vals = explainer_shap.shap_values(X_explain)
    if isinstance(shap_vals, list):
        shap_vals = np.stack(shap_vals, axis=0)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim == 3:
        mean_abs = mean_abs.mean(axis=0)
    top_per_feature = np.array(mean_abs).mean(axis=0)
    top_idx = np.argsort(top_per_feature)[-10:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    pd.DataFrame({"Feature": top_features, "MeanAbsSHAP": top_per_feature[top_idx]}).to_csv(
        "results/per_attack_top_shap_features.csv", index=False
    )
    print("Saved results/per_attack_top_shap_features.csv")

print("\n--- Two-stage (binary -> multi-class) ---")
stage1 = RandomForestClassifier(n_estimators=100, random_state=42)
stage1.fit(X_train_np, y_bin_tr)
pred_bin = stage1.predict(X_test_np)

# Stage 2: only on samples predicted as attack
attack_mask = pred_bin == 1
if attack_mask.sum() > 0:
    X_attack = X_test_np[attack_mask]
    y_attack_true = y_att_te[attack_mask]
    stage2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    stage2.fit(X_train_np[y_bin_tr == 1], y_att_tr[y_bin_tr == 1])
    y_attack_pred = stage2.predict(X_attack)
    print(classification_report(y_attack_true, y_attack_pred, labels=np.arange(len(le_attack.classes_)), target_names=le_attack.classes_, zero_division=0))
    f1_stage2 = f1_score(y_attack_true, y_attack_pred, average="macro", zero_division=0)
else:
    f1_stage2 = 0.0

# Full pipeline prediction for metric: normal stays normal, attack gets class from stage2
normal_idx = int(list(le_attack.classes_).index("normal")) if "normal" in le_attack.classes_ else 0
y_pred_two_stage = np.full(len(y_att_te), normal_idx, dtype=int)
if attack_mask.sum() > 0:
    for i, idx in enumerate(np.where(attack_mask)[0]):
        y_pred_two_stage[idx] = y_attack_pred[i]
f1_two_stage = f1_score(y_att_te, y_pred_two_stage, average="macro", zero_division=0)
print(f"Two-stage macro F1 (full pipeline): {f1_two_stage:.4f}")

joblib.dump(stage1, "models/NSL-KDD_two_stage_binary.pkl")
if attack_mask.sum() > 0:
    joblib.dump(stage2, "models/NSL-KDD_two_stage_multiclass.pkl")
pd.DataFrame([
    {"Model": "Multi-class RF (weighted)", "Dataset": "NSL-KDD", "MacroF1": f1_score(y_att_te, y_pred_multi, average="macro", zero_division=0)},
    {"Model": "Two-stage (binary + multi-class)", "Dataset": "NSL-KDD", "MacroF1": f1_two_stage},
]).to_csv("results/multiclass_two_stage_comparison.csv", index=False)
print("Saved results/multiclass_two_stage_comparison.csv")
print("Done: multiclass_attack_analysis.py")
