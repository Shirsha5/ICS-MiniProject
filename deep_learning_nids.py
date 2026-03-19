import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
import shap
from lime.lime_tabular import LimeTabularExplainer

print("Loading NSL-KDD dataset...")

train_path = "datasets/KDDTrain+.txt"
test_path = "datasets/KDDTest+.txt"

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

train = pd.read_csv(train_path, names=columns)
test = pd.read_csv(test_path, names=columns)

train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

X_train = train.drop(['label', 'difficulty'], axis=1)
y_train = train['label']

X_test = test.drop(['label', 'difficulty'], axis=1)
y_test = test['label']

cat_cols = ['protocol_type', 'service', 'flag']

for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#MLP IMPLEMENTATION
print("Training Deep Learning Model (MLP)...")

start = time.time()

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.1)

train_time = time.time() - start

print("Evaluating MLP...")

plt.figure()
plt.plot(model.history.history['loss'], label='Train Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss (MLP)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/training_loss_dl.png")
plt.close()

probs = model.predict(X_test)
preds = (probs > 0.5).astype(int)

mlp_acc = accuracy_score(y_test, preds)
mlp_prec = precision_score(y_test, preds)
mlp_rec = recall_score(y_test, preds)
mlp_f1 = f1_score(y_test, preds)

# ROC Curve for MLP
fpr, tpr, _ = roc_curve(y_test, probs)  
mlp_roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {mlp_roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Deep Learning NIDS (MLP)")
plt.legend()
plt.savefig("results/dl_roc_curve.png")
plt.close()

print("MLP ROC-AUC:", mlp_roc_auc)
print("\nMLP Results")
print("Accuracy:", mlp_acc)
print("Precision:", mlp_prec)
print("Recall:", mlp_rec)
print("F1 Score:", mlp_f1)
print("Train Time (s):", train_time)

# SHAP for MLP
print("Generating SHAP explanations for MLP...")
sample_X = X_test[:100]
explainer = shap.Explainer(model, X_train)
shap_values = explainer(sample_X)
shap.summary_plot(shap_values, sample_X, show=False)
plt.savefig("results/shap_dl_summary.png")
plt.close()
print("SHAP summary for MLP saved.")

# LIME for MLP
print("Generating LIME explanation for MLP...")
def predict_proba_wrapper(X):
    probs = model.predict(X)
    return np.hstack([1 - probs, probs])

explainer = LimeTabularExplainer(X_train, mode="classification", feature_names=[f"f{i}" for i in range(X_train.shape[1])])
exp = explainer.explain_instance(X_test[0], predict_proba_wrapper, num_features=10)
fig = exp.as_pyplot_figure()
fig.savefig("results/lime_dl_explanation.png")
plt.close()
print("LIME explanation for MLP saved.")

# Save MLP results
mlp_results = pd.DataFrame({
    "Model": ["Deep Learning MLP"],
    "Dataset": ["NSL-KDD"],
    "Accuracy": [mlp_acc],
    "Precision": [mlp_prec],
    "Recall": [mlp_rec],
    "F1 Score": [mlp_f1],
    "ROC-AUC": [mlp_roc_auc],
    "Train Time": [train_time]
})
mlp_results.to_csv("results/deep_learning_results.csv", index=False)
print("\nMLP results saved to results/deep_learning_results.csv")

# Confusion Matrix for MLP
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Deep Learning NIDS (MLP)")
plt.savefig("results/confusion_matrix_dl.png")
plt.close()

# CNN IMPLEMENTATION
print("\nTraining CNN Model...")
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

start_cnn = time.time()
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2)

cnn_train_time = time.time() - start_cnn

y_pred_prob = cnn_model.predict(X_test_cnn)
y_pred = (y_pred_prob > 0.5).astype(int)

cnn_acc = accuracy_score(y_test, y_pred)
cnn_prec = precision_score(y_test, y_pred)
cnn_rec = recall_score(y_test, y_pred)
cnn_f1 = f1_score(y_test, y_pred)

# ROC Curve for CNN
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
cnn_roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {cnn_roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CNN NIDS")
plt.legend()
plt.savefig("results/roc_curve_cnn.png")
plt.close()

print("CNN Metrics - Accuracy:", cnn_acc, "Precision:", cnn_prec, "Recall:", cnn_rec, "F1:", cnn_f1, "ROC-AUC:", cnn_roc_auc)

# Confusion Matrix for CNN
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN NIDS")
plt.savefig("results/confusion_matrix_cnn.png")
plt.close()

# Training Loss for CNN
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss (CNN)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/training_loss_cnn.png")
plt.close()

# SHAP for CNN
print("Generating SHAP explanations for CNN...")
explainer = shap.KernelExplainer(lambda x: cnn_model.predict(x.reshape(x.shape[0], x.shape[1], 1)), X_train[:100])
shap_values = explainer.shap_values(X_test[:50])
shap.summary_plot(shap_values, X_test[:50], show=False)
plt.savefig("results/shap_cnn_summary.png")
plt.close()
print("SHAP summary for CNN saved.")

# LIME for CNN
print("Generating LIME explanation for CNN...")
def predict_proba_cnn(X):
    probs = cnn_model.predict(X.reshape(X.shape[0], X.shape[1], 1))
    return np.hstack([1 - probs, probs])

explainer = LimeTabularExplainer(X_train, mode="classification", feature_names=[f"f{i}" for i in range(X_train.shape[1])])
exp = explainer.explain_instance(X_test[0], predict_proba_cnn, num_features=10)
fig = exp.as_pyplot_figure()
fig.savefig("results/lime_cnn_explanation.png")
plt.close()
print("LIME explanation for CNN saved.")


cnn_results = pd.DataFrame({
    "Model": ["CNN"],
    "Dataset": ["NSL-KDD"],
    "Accuracy": [cnn_acc],
    "Precision": [cnn_prec],
    "Recall": [cnn_rec],
    "F1 Score": [cnn_f1],
    "ROC-AUC": [cnn_roc_auc],
    "Train Time": [cnn_train_time]
})
cnn_results.to_csv("results/cnn_results.csv", index=False)
print("CNN results saved to results/cnn_results.csv")

# saving the comparison of mlp and cnn results
print("\nModel Comparison (MLP vs CNN)...")
df = pd.DataFrame([
    ["MLP", mlp_acc, mlp_f1, mlp_roc_auc],
    ["CNN", cnn_acc, cnn_f1, cnn_roc_auc]
], columns=["Model", "Accuracy", "F1", "ROC-AUC"])

df.to_csv("results/dl_model_comparison.csv", index=False)
df.set_index("Model").plot(kind="bar")
plt.title("Model Comparison: MLP vs CNN")
plt.ylabel("Score")
plt.savefig("results/dl_comparison_plot.png")
plt.close()
print("Comparison saved.")