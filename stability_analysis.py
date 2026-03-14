import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from lime.lime_tabular import LimeTabularExplainer

from preprocessing import load_and_preprocess_data

# LOAD DATA

train_path = "datasets/UNSW_NB15_training-set.csv"
test_path = "datasets/UNSW_NB15_testing-set.csv"

X_train, X_test, y_train, y_test = load_and_preprocess_data(train_path, test_path)

# LOAD MODEL

model = joblib.load("ids_random_forest.pkl")
print("Model loaded.")

# SHAP EXPLAINER

explainer = shap.TreeExplainer(model)

# LIME EXPLAINER

feature_names = X_train.columns.tolist()

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["Normal", "Attack"],
    mode="classification"
)

print("LIME explainer ready.")

# SAMPLE DATA

X_sample = X_test.sample(200, random_state=42)
print("Samples selected:", X_sample.shape)

# PERTURBATION FUNCTION

def perturb_sample(sample, noise_level=0.02):
    noise = np.random.normal(0, noise_level, size=sample.shape)
    return sample + noise

# LIME VECTOR CONVERSION

def lime_to_vector(lime_exp, feature_names):

    vec = np.zeros(len(feature_names))

    for feature, weight in lime_exp.as_list():

        for i, f in enumerate(feature_names):
            if f in feature:
                vec[i] = weight

    return vec

# SHAP STABILITY EXPERIMENT

shap_similarities = []

for i in range(len(X_sample)):

    sample = X_sample.iloc[i]
    perturbed = perturb_sample(sample.values)

    sample_df = pd.DataFrame([sample.values], columns=X_train.columns)
    perturbed_df = pd.DataFrame([perturbed], columns=X_train.columns)

    shap_original = explainer.shap_values(sample_df)
    shap_perturbed = explainer.shap_values(perturbed_df)

    if isinstance(shap_original, list):
        shap_original = shap_original[1]
        shap_perturbed = shap_perturbed[1]

    sim = cosine_similarity(
    shap_original.reshape(1, -1),
    shap_perturbed.reshape(1, -1)
    )[0][0]

    shap_similarities.append(sim)


print("\nAverage SHAP Stability:", np.mean(shap_similarities))

# SHAP STABILITY DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(shap_similarities, bins=20, kde=True)

plt.title("Distribution of SHAP Explanation Stability")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Samples")

plt.show()

# LIME STABILITY EXPERIMENT

lime_similarities = []

for i in range(len(X_sample)):

    sample = X_sample.iloc[i]
    perturbed = perturb_sample(sample.values)

    exp_original = lime_explainer.explain_instance(
        sample.values,
        lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns)),
        num_features=10
    )

    exp_perturbed = lime_explainer.explain_instance(
        perturbed,
        lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns)),
        num_features=10
    )

    vec1 = lime_to_vector(exp_original, feature_names)
    vec2 = lime_to_vector(exp_perturbed, feature_names)

    sim = cosine_similarity([vec1], [vec2])[0][0]

    lime_similarities.append(sim)


print("\nAverage LIME Stability:", np.mean(lime_similarities))

# LIME STABILITY DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(lime_similarities, bins=20, kde=True)

plt.title("Distribution of LIME Explanation Stability")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Samples")

plt.show()

# MODEL PREDICTIONS

predictions = model.predict(X_test)

correct_indices = []
incorrect_indices = []

for i in range(len(predictions)):

    if predictions[i] == y_test.iloc[i]:
        correct_indices.append(i)
    else:
        incorrect_indices.append(i)

print("\nCorrect predictions:", len(correct_indices))
print("Incorrect predictions:", len(incorrect_indices))

# SAMPLE FROM EACH GROUP

correct_sample = X_test.iloc[correct_indices].sample(100, random_state=42)
incorrect_sample = X_test.iloc[incorrect_indices].sample(100, random_state=42)

# FUNCTION TO COMPUTE SHAP STABILITY

def compute_shap_stability(samples):

    sims = []

    for i in range(len(samples)):

        sample = samples.iloc[i]
        perturbed = perturb_sample(sample.values)

        sample_df = pd.DataFrame([sample.values], columns=X_train.columns)
        perturbed_df = pd.DataFrame([perturbed], columns=X_train.columns)

        shap_original = explainer.shap_values(sample_df)
        shap_perturbed = explainer.shap_values(perturbed_df)

        if isinstance(shap_original, list):
            shap_original = shap_original[1]
            shap_perturbed = shap_perturbed[1]

        sim = cosine_similarity(
            shap_original.reshape(1, -1),
            shap_perturbed.reshape(1, -1)
        )[0][0]

        sims.append(sim)

    return np.mean(sims)

# CORRECT vs INCORRECT STABILITY

correct_stability = compute_shap_stability(correct_sample)
incorrect_stability = compute_shap_stability(incorrect_sample)

print("\nSHAP Stability (Correct Predictions):", correct_stability)
print("SHAP Stability (Incorrect Predictions):", incorrect_stability)
