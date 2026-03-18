"""
Download datasets for XAI-NIDS research:
  1. UNSW-NB15 (from Kaggle-style public URLs)
  2. NSL-KDD (from UNB)
"""
import os
import urllib.request
import zipfile

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

URLS = {
    "UNSW_NB15_training-set.csv": "https://raw.githubusercontent.com/oshoyemi/project/main/UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv": "https://raw.githubusercontent.com/oshoyemi/project/main/UNSW_NB15_testing-set.csv",
    "KDDTrain+.txt": "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTest%2B.txt",
}

def download_file(name, url):
    dest = os.path.join(DATASET_DIR, name)
    if os.path.exists(dest):
        print(f"  [skip] {name} already exists")
        return
    print(f"  [download] {name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [done] {name} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

if __name__ == "__main__":
    print("Downloading datasets...\n")
    for name, url in URLS.items():
        download_file(name, url)
    print("\nAll downloads complete.")
    print("Files in datasets/:")
    for f in os.listdir(DATASET_DIR):
        sz = os.path.getsize(os.path.join(DATASET_DIR, f)) / 1e6
        print(f"  {f} ({sz:.1f} MB)")
