import kagglehub
import shutil
from pathlib import Path

def fetch_data():
    local_data_dir=Path("data/raw")
    local_data_dir.mkdir(parents=True,exist_ok=True)

    print("Fetching stellar dataset from Kaggle...")
    cache_path = kagglehub.dataset_download("fedesoriano/stellar-classification-dataset-sdss17")
    
    source_file = Path(cache_path) / "star_classification.csv"
    dest_file = local_data_dir / "star_classification.csv"
    
    shutil.copy(source_file, dest_file)
    print(f"[+] Data successfully copied to: {dest_file}")

if __name__ == "__main__":
    fetch_data()