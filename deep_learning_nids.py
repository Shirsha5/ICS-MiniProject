import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc 
import shap


print("Loading NSL-KDD dataset...")

train_path = "datasets/KDDTrain+.txt"
test_path = "datasets/KDDTest+.txt"


columns = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
"urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
"num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
"dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label","difficulty"
]

train = pd.read_csv(train_path, names=columns)
test = pd.read_csv(test_path, names=columns)


train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

X_train = train.drop(['label','difficulty'], axis=1)
y_train = train['label']

X_test = test.drop(['label','difficulty'], axis=1)
y_test = test['label']


cat_cols = ['protocol_type','service','flag']

for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Deep Learning Model...")

start = time.time()

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1
)

train_time = time.time() - start

print("Evaluating...")

plt.figure()

plt.plot(model.history.history['loss'], label='Train Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("results/training_loss_dl.png")
plt.close()

probs = model.predict(X_test)
preds = (probs > 0.5).astype(int)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Deep Learning NIDS")
plt.legend()

plt.savefig("results/dl_roc_curve.png")
plt.close()

print("ROC-AUC:", roc_auc)

print("\nResults")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

#SHAP explanation for MLP
print("Generating SHAP explanations...")

sample_X = X_test[:100]

explainer = shap.Explainer(model, X_train)
shap_values = explainer(sample_X)

# Summary plot
shap.summary_plot(shap_values, sample_X, show=False)

plt.savefig("results/shap_dl_summary.png")
plt.close()

print("SHAP summary saved.")

results = pd.DataFrame({
    "Model": ["Deep Learning MLP"],
    "Dataset": ["NSL-KDD"],
    "Accuracy": [acc],
    "Precision": [prec],
    "Recall": [rec],
    "F1 Score": [f1],
    "Train Time": [train_time]
})
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Attack"],
            yticklabels=["Normal","Attack"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Deep Learning NIDS")

plt.savefig("results/confusion_matrix_dl.png")
plt.close()

results.to_csv("results/deep_learning_results.csv", index=False)

print("\nResults saved to results/deep_learning_results.csv")