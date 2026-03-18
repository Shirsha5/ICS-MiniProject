"""Unsupervised anomaly NIDS: train on normal traffic, reconstruction error as anomaly score. Uses TensorFlow GPU when available, else PCA."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from preprocessing import load_nsl_kdd

np.random.seed(42)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

USE_GPU_AUTOENCODER = True
model_ae = None
model_pca = None

if USE_GPU_AUTOENCODER:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print("TensorFlow GPU autoencoder: {} GPU(s) available.".format(len(gpus)))
        else:
            print("TensorFlow: no GPU found, using CPU for autoencoder.")
    except ImportError:
        pass

def _build_tf_autoencoder(n_features, latent_dim=32, seed=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.layers.Input(shape=(n_features,))
    enc = tf.keras.layers.Dense(64, activation="relu")(inp)
    enc = tf.keras.layers.Dense(latent_dim, activation="relu")(enc)
    dec = tf.keras.layers.Dense(64, activation="relu")(enc)
    out = tf.keras.layers.Dense(n_features, activation="linear")(dec)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

print("Loading NSL-KDD (binary)...")
X_train, X_test, y_train, y_test, scaler = load_nsl_kdd()
X_train_np = X_train.values.astype(np.float64)
X_test_np = X_test.values.astype(np.float64)
n_features = X_train_np.shape[1]

# Train on normal traffic only
normal_idx = (y_train == 0).values
X_normal = X_train_np[normal_idx]
if X_normal.shape[0] > 50000:
    np.random.seed(42)
    idx = np.random.choice(X_normal.shape[0], 50000, replace=False)
    X_normal = X_normal[idx]

if USE_GPU_AUTOENCODER:
    try:
        import tensorflow as tf
        latent = min(32, n_features - 1)
        model_ae = _build_tf_autoencoder(n_features, latent_dim=latent)
        print("Training neural autoencoder (GPU if available) on {} normal samples...".format(X_normal.shape[0]))
        model_ae.fit(X_normal, X_normal, epochs=20, batch_size=256, validation_split=0.1, verbose=0)
        # Reconstruction error = MSE per sample
        def reconstruction_error_tf(m, X):
            pred = m.predict(X, verbose=0)
            return np.mean((X - pred) ** 2, axis=1)
        train_errors = reconstruction_error_tf(model_ae, X_train_np)
        test_errors = reconstruction_error_tf(model_ae, X_test_np)
        backend_used = "TensorFlow (GPU)" if tf.config.list_physical_devices("GPU") else "TensorFlow (CPU)"
        print("Using {} autoencoder.".format(backend_used))
    except Exception as e:
        print("TensorFlow autoencoder failed ({}), using PCA.".format(e))
        model_ae = None

if model_ae is None:
    from sklearn.decomposition import PCA
    n_components = min(32, X_normal.shape[1] - 1)
    model_pca = PCA(n_components=n_components, random_state=42)
    model_pca.fit(X_normal)
    print("Training reconstruction model (PCA, CPU) on {} normal samples".format(X_normal.shape[0]))

    def reconstruction_error_pca(m, X):
        X_proj = m.transform(X)
        X_recon = m.inverse_transform(X_proj)
        return np.mean((X - X_recon) ** 2, axis=1)
    train_errors = reconstruction_error_pca(model_pca, X_train_np)
    test_errors = reconstruction_error_pca(model_pca, X_test_np)

# Labels: 1 = attack (anomaly), 0 = normal
y_score = test_errors
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_score)
ap = average_precision_score(y_test, y_score)
print("ROC-AUC: {:.4f}, Average Precision: {:.4f}".format(roc_auc, ap))

plt.figure()
plt.plot(fpr, tpr, label="Reconstruction (AUC={:.3f})".format(roc_auc))
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Autoencoder-style NIDS - ROC Curve")
plt.legend()
plt.savefig("results/autoencoder_roc.png")
plt.close()

threshold = np.percentile(train_errors[normal_idx], 99)
y_pred_ae = (test_errors >= threshold).astype(int)
acc = accuracy_score(y_test, y_pred_ae)
f1 = f1_score(y_test, y_pred_ae, zero_division=0)
print("At 99th pct threshold: Accuracy={:.4f}, F1={:.4f}".format(acc, f1))
print(classification_report(y_test, y_pred_ae, target_names=["Normal", "Attack"]))

model_name = "Autoencoder (TF GPU/CPU, unsupervised)" if model_ae is not None else "Autoencoder (PCA, unsupervised)"
pd.DataFrame([{
    "Model": model_name,
    "Dataset": "NSL-KDD",
    "ROC-AUC": roc_auc,
    "AveragePrecision": ap,
    "Accuracy": acc,
    "F1": f1,
}]).to_csv("results/autoencoder_results.csv", index=False)
# Save the model we actually used (TF model saved as Keras; PCA as joblib)
if model_ae is not None:
    model_ae.save("models/NSL-KDD_autoencoder.keras")
    joblib.dump({"backend": "keras", "path": "NSL-KDD_autoencoder.keras"}, "models/NSL-KDD_autoencoder.pkl")
else:
    joblib.dump(model_pca, "models/NSL-KDD_autoencoder.pkl")
print("Saved results/autoencoder_results.csv, models/NSL-KDD_autoencoder.*")
print("Done: autoencoder_nids.py")
