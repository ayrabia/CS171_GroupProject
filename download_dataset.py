import os
import subprocess
import sys


def install_kaggle():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])


def download_dataset():
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle...")
        install_kaggle()

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("Kaggle API credentials not found.")
        print("1. Go to https://www.kaggle.com/settings > API > 'Create New Token'")
        print("2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("3. Run this script again.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    print("Downloading Brain Tumor MRI Dataset...")
    os.system(
        "kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/ --unzip"
    )
    print("Download complete. Dataset saved to data/")


if __name__ == "__main__":
    download_dataset()
