import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
import pandas as pd

from preprocessing import load_and_preprocess_data

# LOAD DATA

train_path = "datasets/UNSW_NB15_training-set.csv"
test_path = "datasets/UNSW_NB15_testing-set.csv"

X_train, X_test, y_train, y_test = load_and_preprocess_data(train_path, test_path)

# LOAD TRAINED MODEL

model = joblib.load("ids_random_forest.pkl")

print("Model loaded successfully.")

# SAMPLE DATA FOR SHAP

# Using only a subset for speed
sample_size = 1000
X_sample = X_test.sample(sample_size, random_state=42)

print("Sample size for SHAP:", X_sample.shape)

# SHAP EXPLAINER

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

print("SHAP values generated.")

# For binary classification, shap returns a list
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # use attack class explanations

# GLOBAL FEATURE IMPORTANCE

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="dot",
    max_display=20
)

# LIME EXPLAINER SETUP

feature_names = X_train.columns.tolist()

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["Normal", "Attack"],
    mode="classification"
)

print("LIME explainer initialized.")

# GENERATE LIME EXPLANATION

sample_index = 0
sample = X_sample.iloc[sample_index].values

lime_exp = lime_explainer.explain_instance(
    sample,
    model.predict_proba,
    num_features=10
)

print("\nLIME Explanation:")
print(lime_exp.as_list())


# SHOW LIME PLOT

import matplotlib.pyplot as plt

fig = lime_exp.as_pyplot_figure()
plt.show()