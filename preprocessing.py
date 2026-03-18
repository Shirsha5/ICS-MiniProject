"""
Preprocessing module for UNSW-NB15 and NSL-KDD datasets.
Handles loading, encoding, scaling, and train/test splitting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

NSL_KDD_CAT_COLS = ['protocol_type', 'service', 'flag']


def load_unsw_nb15(data_dir="datasets"):
    """Load UNSW-NB15 dataset and return preprocessed X_train, X_test, y_train, y_test."""
    train_path = f"{data_dir}/UNSW_NB15_training-set.csv"
    df = pd.read_csv(train_path)
    print(f"[UNSW-NB15] Loaded {df.shape[0]} rows, {df.shape[1]} cols")

    df = df.drop(columns=["id"], errors="ignore")
    y = df["label"]
    X = df.drop(columns=["label", "attack_cat"], errors="ignore")

    categorical_cols = ["proto", "service", "state"]
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    print(f"[UNSW-NB15] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[UNSW-NB15] Label dist (train): {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler


def load_nsl_kdd(data_dir="datasets"):
    """Load NSL-KDD dataset and return preprocessed X_train, X_test, y_train, y_test."""
    train_path = f"{data_dir}/KDDTrain+.txt"
    test_path = f"{data_dir}/KDDTest+.txt"

    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    print(f"[NSL-KDD] Train: {df_train.shape}, Test: {df_test.shape}")

    df_train = df_train.drop(columns=["difficulty"])
    df_test = df_test.drop(columns=["difficulty"])

    df_train["label"] = (df_train["label"] != "normal").astype(int)
    df_test["label"] = (df_test["label"] != "normal").astype(int)

    y_train = df_train["label"]
    y_test = df_test["label"]
    X_train = df_train.drop(columns=["label"])
    X_test = df_test.drop(columns=["label"])

    for col in NSL_KDD_CAT_COLS:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]])
        le.fit(combined.astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)

    print(f"[NSL-KDD] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[NSL-KDD] Label dist (train): {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler


def load_dataset(name, data_dir="datasets"):
    """Load a dataset by name. Returns X_train, X_test, y_train, y_test, scaler."""
    loaders = {
        "UNSW-NB15": load_unsw_nb15,
        "NSL-KDD": load_nsl_kdd,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    return loaders[name](data_dir)


if __name__ == "__main__":
    for ds in ["UNSW-NB15", "NSL-KDD"]:
        print(f"\n{'='*50}")
        X_tr, X_te, y_tr, y_te, sc = load_dataset(ds)
        print(f"Features: {X_tr.shape[1]}, Train samples: {len(y_tr)}, Test samples: {len(y_te)}")
