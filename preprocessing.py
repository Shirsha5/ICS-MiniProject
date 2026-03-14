import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(train_path, test_path):

    # LOAD DATA
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # REMOVE ID COLUMN
    train = train.drop(columns=["id"])
    test = test.drop(columns=["id"])

    # SEPARATE FEATURES AND LABEL
    y_train = train["label"]
    y_test = test["label"]

    X_train = train.drop(columns=["label", "attack_cat"])
    X_test = test.drop(columns=["label", "attack_cat"])

    # ENCODE CATEGORICAL FEATURES
    categorical_cols = ["proto", "service", "state"]

    for col in categorical_cols:

        encoder = LabelEncoder()

        combined = pd.concat([X_train[col], X_test[col]])

        encoder.fit(combined)

        X_train[col] = encoder.transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    print("\nCategorical encoding completed.")

    return X_train, X_test, y_train, y_test