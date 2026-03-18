"""Build one summary figure from multiclass, autoencoder, and stacking result CSVs."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Multi-class vs two-stage F1
p1 = "results/multiclass_two_stage_comparison.csv"
if os.path.isfile(p1):
    d = pd.read_csv(p1)
    axes[0].bar(d["Model"], d["MacroF1"], color=["steelblue", "coral"])
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Multi-class vs two-stage (NSL-KDD)")
    axes[0].tick_params(axis="x", rotation=15)
else:
    axes[0].text(0.5, 0.5, "Run multiclass_attack_analysis.py", ha="center")

# Autoencoder
p2 = "results/autoencoder_results.csv"
if os.path.isfile(p2):
    d = pd.read_csv(p2)
    axes[1].bar(["ROC-AUC", "F1"], [d["ROC-AUC"].iloc[0], d["F1"].iloc[0]], color="teal")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Autoencoder NIDS")
else:
    axes[1].text(0.5, 0.5, "Run autoencoder_nids.py", ha="center")

# Stacking
p3 = "results/ensemble_stacking_results.csv"
if os.path.isfile(p3):
    d = pd.read_csv(p3)
    axes[2].bar(d["Dataset"], d["F1"], color=["navy", "orange"])
    axes[2].set_ylabel("F1")
    axes[2].set_title("Stacking ensemble")
else:
    axes[2].text(0.5, 0.5, "Run ensemble_stacking.py", ha="center")

plt.tight_layout()
plt.savefig("results/extended_components_summary.png", dpi=150)
plt.close()
print("Saved results/extended_components_summary.png")
