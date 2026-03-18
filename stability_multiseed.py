"""Stability across runs and statistical tests: SHAP cosine under perturbation (multi-seed, 95% CI) and paired t-tests (noise 0.01 vs 0.02, SHAP vs LIME). Run after Full_Analysis.ipynb."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import joblib

from preprocessing import load_dataset

os.makedirs("results", exist_ok=True)
SEEDS = [42, 43, 44]
NOISE_LEVELS = [0.01, 0.02]
N_SAMPLES = 40
DATASET = "UNSW-NB15"
MODEL_PATH = "models/UNSW-NB15_RandomForest.pkl"
N_LIME_SAMPLES = 10
LIME_NUM_SAMPLES = 500


def lime_stability_one_sample(explainer, model, x_orig, x_pert, fnames, num_features=10, num_samples=LIME_NUM_SAMPLES):
    def pred_fn(X):
        return model.predict_proba(X)

    def vec(x):
        e = explainer.explain_instance(x, pred_fn, num_features=num_features, num_samples=num_samples)
        d = dict(e.as_list())
        return np.array([d.get(f, 0.0) for f in fnames])

    v1, v2 = vec(x_orig), vec(x_pert)
    if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
        return 1.0
    return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])


def main():
    if not os.path.isfile(MODEL_PATH):
        print("Train models first (Full_Analysis.ipynb). Missing:", MODEL_PATH)
        return

    X_tr, X_te, y_tr, y_te, _ = load_dataset(DATASET)
    fnames = list(X_te.columns)
    model = joblib.load(MODEL_PATH)
    import shap
    explainer_shap = shap.TreeExplainer(model, X_tr)

    rows = []
    for seed in SEEDS:
        np.random.seed(seed)
        idx = np.random.choice(len(X_te), min(N_SAMPLES, len(X_te)), replace=False)
        X_s = X_te.iloc[idx]
        X_s_val = X_s.values
        sv_o = explainer_shap.shap_values(X_s)
        if isinstance(sv_o, list):
            sv_o = sv_o[1]
        sv_o = np.asarray(sv_o)
        if sv_o.ndim == 3:
            sv_o = sv_o[:, :, -1] if sv_o.shape[2] >= 2 else sv_o.reshape(sv_o.shape[0], -1)
        for noise in NOISE_LEVELS:
            rng = np.random.RandomState(seed + int(noise * 1000))
            X_p = X_s_val + rng.normal(0, noise, X_s_val.shape)
            X_p_df = pd.DataFrame(X_p, columns=fnames)
            sv_p = explainer_shap.shap_values(X_p_df)
            if isinstance(sv_p, list):
                sv_p = sv_p[1]
            sv_p = np.asarray(sv_p)
            if sv_p.ndim == 3:
                sv_p = sv_p[:, :, -1] if sv_p.shape[2] >= 2 else sv_p.reshape(sv_p.shape[0], -1)
            cos_list = []
            for i in range(len(X_s)):
                c = cosine_similarity(sv_o[i : i + 1], sv_p[i : i + 1])[0, 0]
                if np.isfinite(c):
                    cos_list.append(c)
            rows.append({"Seed": seed, "Noise": noise, "Cosine_mean": np.mean(cos_list)})

    df_agg = pd.DataFrame(rows)
    agg = df_agg.groupby("Noise")["Cosine_mean"].agg(["mean", "std", "count"])
    agg["CI95_low"] = agg["mean"] - 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg["CI95_high"] = agg["mean"] + 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg.to_csv("results/stability_multiseed_results.csv")
    print("=== Multi-seed SHAP stability (mean ± CI across seeds) ===")
    print(agg)

    np.random.seed(42)
    idx = np.random.choice(len(X_te), min(N_SAMPLES, len(X_te)), replace=False)
    X_s = X_te.iloc[idx]
    X_s_val = X_s.values
    sv_o = explainer_shap.shap_values(X_s)
    if isinstance(sv_o, list):
        sv_o = sv_o[1]
    sv_o = np.asarray(sv_o)
    if sv_o.ndim == 3:
        sv_o = sv_o[:, :, -1] if sv_o.shape[2] >= 2 else sv_o.reshape(sv_o.shape[0], -1)
    rng1 = np.random.RandomState(10001)
    X_p01 = X_s_val + rng1.normal(0, 0.01, X_s_val.shape)
    sv_p01 = explainer_shap.shap_values(pd.DataFrame(X_p01, columns=fnames))
    if isinstance(sv_p01, list):
        sv_p01 = sv_p01[1]
    sv_p01 = np.asarray(sv_p01)
    if sv_p01.ndim == 3:
        sv_p01 = sv_p01[:, :, -1] if sv_p01.shape[2] >= 2 else sv_p01.reshape(sv_p01.shape[0], -1)
    cos_01 = [
        float(cosine_similarity(sv_o[i : i + 1], sv_p01[i : i + 1])[0, 0])
        for i in range(len(X_s))
    ]
    rng2 = np.random.RandomState(10002)
    X_p02 = X_s_val + rng2.normal(0, 0.02, X_s_val.shape)
    sv_p02 = explainer_shap.shap_values(pd.DataFrame(X_p02, columns=fnames))
    if isinstance(sv_p02, list):
        sv_p02 = sv_p02[1]
    sv_p02 = np.asarray(sv_p02)
    if sv_p02.ndim == 3:
        sv_p02 = sv_p02[:, :, -1] if sv_p02.shape[2] >= 2 else sv_p02.reshape(sv_p02.shape[0], -1)
    cos_02 = [
        float(cosine_similarity(sv_o[i : i + 1], sv_p02[i : i + 1])[0, 0])
        for i in range(len(X_s))
    ]
    t_stat, p_val = stats.ttest_rel(cos_01, cos_02)
    paired_noise = {
        "test": "paired_ttest_SHAP_cosine_noise001_vs_002",
        "n_samples": len(cos_01),
        "mean_cos_001": np.mean(cos_01),
        "mean_cos_002": np.mean(cos_02),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "interpretation": "p<0.05 => stability significantly lower at higher noise",
    }

    n_lime = min(N_LIME_SAMPLES, len(X_s))
    X_sub = X_s.iloc[:n_lime]
    X_sub_val = X_sub.values
    sv_o_sub = explainer_shap.shap_values(X_sub)
    if isinstance(sv_o_sub, list):
        sv_o_sub = sv_o_sub[1]
    sv_o_sub = np.asarray(sv_o_sub)
    if sv_o_sub.ndim == 3:
        sv_o_sub = sv_o_sub[:, :, -1] if sv_o_sub.shape[2] >= 2 else sv_o_sub.reshape(sv_o_sub.shape[0], -1)
    rng = np.random.RandomState(1001)
    X_p_sub = X_sub_val + rng.normal(0, 0.02, X_sub_val.shape)
    sv_p_sub = explainer_shap.shap_values(pd.DataFrame(X_p_sub, columns=fnames))
    if isinstance(sv_p_sub, list):
        sv_p_sub = sv_p_sub[1]
    sv_p_sub = np.asarray(sv_p_sub)
    if sv_p_sub.ndim == 3:
        sv_p_sub = sv_p_sub[:, :, -1] if sv_p_sub.shape[2] >= 2 else sv_p_sub.reshape(sv_p_sub.shape[0], -1)
    shap_cos = []
    for i in range(n_lime):
        shap_cos.append(
            float(cosine_similarity(sv_o_sub[i : i + 1], sv_p_sub[i : i + 1])[0, 0])
        )
    lime_cos = []
    bg = X_tr.sample(min(500, len(X_tr)), random_state=42)
    from lime.lime_tabular import LimeTabularExplainer
    lime_explainer = LimeTabularExplainer(
        bg.values, mode="classification", feature_names=fnames,
        discretize_continuous=True, random_state=42,
    )
    print("Running SHAP vs LIME paired test...")
    for i in range(n_lime):
        try:
            lime_cos.append(
                lime_stability_one_sample(lime_explainer, model, X_sub_val[i], X_p_sub[i], fnames)
            )
        except Exception as ex:
            lime_cos.append(np.nan)
            print("  LIME sample {} failed: {}".format(i, ex))
    lime_cos = np.array(lime_cos)
    shap_cos = np.array(shap_cos)
    mask = np.isfinite(lime_cos) & np.isfinite(shap_cos)
    if mask.sum() >= 3:
        t2, p2 = stats.ttest_rel(shap_cos[mask], lime_cos[mask])
        paired_xai = {
            "test": "paired_ttest_SHAP_vs_LIME_cosine_noise002",
            "n_samples": int(mask.sum()),
            "mean_SHAP_cos": float(np.mean(shap_cos[mask])),
            "mean_LIME_cos": float(np.mean(lime_cos[mask])),
            "t_statistic": float(t2),
            "p_value": float(p2),
            "interpretation": "p<0.05 => SHAP and LIME stability differ significantly",
        }
    else:
        paired_xai = {"test": "LIME_skipped", "reason": "insufficient valid LIME runs (got {})".format(int(mask.sum()))}

    stats_out = pd.DataFrame([paired_noise, paired_xai])
    stats_out.to_csv("results/stability_statistical_tests.csv", index=False)
    print("\n=== Statistical tests ===")
    print(stats_out.to_string())
    print("\nSaved: results/stability_multiseed_results.csv, results/stability_statistical_tests.csv")


if __name__ == "__main__":
    main()
