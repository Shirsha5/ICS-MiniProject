import pandas as pd

train_path = "datasets/UNSW_NB15_training-set.csv"
test_path = "datasets/UNSW_NB15_testing-set.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("\nTrain Shape:", train.shape)
print("Test Shape:", test.shape)

print("\nColumns:")
print(train.columns)

print("\nFirst 5 rows:")
print(train.head())

print("\nData Types:")
print(train.dtypes)

print("\nMissing Values:")
print(train.isnull().sum())

print("\nLabel Distribution:")
print(train['label'].value_counts())

print("\nAttack Categories:")
print(train['attack_cat'].value_counts())

print("\nStatistical Summary:")
print(train.describe())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,10))

corr = train.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()

sns.histplot(train['dur'], bins=50)
plt.title("Flow Duration Distribution")
plt.show()