"""Stacking ensemble: LogisticRegression meta-learner on RF, XGBoost, GradientBoosting. Loads models from models/ if present."""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocessing import load_dataset

np.random.seed(42)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

def get_base_models():
    base = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    if HAS_XGB:
        base.append(("xgb", xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")))
    return base

def main(dataset_name="NSL-KDD"):
    print(f"Loading {dataset_name}...")
    X_train, X_test, y_train, y_test, scaler = load_dataset(dataset_name)
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    model_dir = "models"
    prefix = dataset_name.replace("-", "-")
    base_models = get_base_models()
    trained = []

    for name, model in base_models:
        path = os.path.join(model_dir, f"{prefix}_{name.replace('xgb', 'XGBoost').replace('rf', 'RandomForest').replace('gb', 'GradientBoosting')}.pkl")
        if os.path.isfile(path):
            m = joblib.load(path)
            print(f"  Loaded {name} from {path}")
        else:
            print(f"  Training {name}...")
            m = model
            m.fit(X_train_np, y_train_np)
            joblib.dump(m, path)
        trained.append((name, m))

    # Meta-features: base model predicted probabilities (or 0/1)
    def get_meta_features(X, base_list):
        out = []
        for name, m in base_list:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)[:, 1]
            else:
                p = m.predict(X).astype(float)
            out.append(p)
        return np.column_stack(out)

    X_meta_train = get_meta_features(X_train_np, trained)
    X_meta_test = get_meta_features(X_test_np, trained)

    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(X_meta_train, y_train_np)
    y_pred_stack = meta.predict(X_meta_test)
    y_pred_proba = meta.predict_proba(X_meta_test)[:, 1]

    acc = accuracy_score(y_test_np, y_pred_stack)
    f1 = f1_score(y_test_np, y_pred_stack, zero_division=0)
    prec = precision_score(y_test_np, y_pred_stack, zero_division=0)
    rec = recall_score(y_test_np, y_pred_stack, zero_division=0)
    print(f"\nStacking ensemble ({dataset_name}): Accuracy={acc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

    joblib.dump(meta, os.path.join(model_dir, f"{prefix}_stacking_meta.pkl"))
    row = {
        "Model": "Stacking (RF+GB+XGB)" if HAS_XGB else "Stacking (RF+GB)",
        "Dataset": dataset_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }
    out_path = "results/ensemble_stacking_results.csv"
    if os.path.isfile(out_path):
        pd.concat([pd.read_csv(out_path), pd.DataFrame([row])], ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(out_path, index=False)
    print("Saved results/ensemble_stacking_results.csv, models/*_stacking_meta.pkl")
    return acc, f1

if __name__ == "__main__":
    for ds in ["NSL-KDD", "UNSW-NB15"]:
        try:
            main(ds)
        except Exception as e:
            print(f"Skip {ds}: {e}")
    print("Done: ensemble_stacking.py")
