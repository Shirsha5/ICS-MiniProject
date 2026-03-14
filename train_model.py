from preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# DATASET PATHS
train_path = "datasets/UNSW_NB15_training-set.csv"
test_path = "datasets/UNSW_NB15_testing-set.csv"

# LOAD PREPROCESSED DATA
X_train, X_test, y_train, y_test = load_and_preprocess_data(train_path, test_path)

print("\nTraining Random Forest model...")

# TRAIN MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# PREDICTIONS
predictions = model.predict(X_test)

# EVALUATION
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

import joblib

joblib.dump(model, "ids_random_forest.pkl")

print("\nModel saved successfully.")