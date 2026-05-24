#!/usr/bin/env python3
"""
Download the Chest X-Ray Pneumonia dataset from Kaggle.

Prerequisites:
  1. Create a Kaggle account: https://www.kaggle.com
  2. Accept the dataset rules on the dataset page
  3. Install kaggle CLI: pip install kaggle
  4. Place kaggle.json in ~/.kaggle/kaggle.json (API token from Account settings)

Usage:
  python scripts/download_dataset.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET = "paultimothymooney/chest-xray-pneumonia"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {DATA_DIR}...")
    print(f"Dataset: https://www.kaggle.com/datasets/{DATASET}")

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                DATASET,
                "-p",
                str(DATA_DIR),
                "--unzip",
            ],
            check=True,
        )
    except FileNotFoundError:
        print(
            "ERROR: 'kaggle' CLI not found.\n"
            "Install with: pip install kaggle\n"
            "Then configure ~/.kaggle/kaggle.json with your API credentials."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print(
            "\nManual download:\n"
            "  1. Visit https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
            "  2. Download and extract to data/raw/chest_xray/\n"
            "  3. Expected structure: data/raw/chest_xray/train|val|test/NORMAL|PNEUMONIA/"
        )
        sys.exit(1)

    chest_xray = DATA_DIR / "chest_xray"
    if chest_xray.exists():
        print(f"\nSuccess! Dataset ready at: {chest_xray}")
    else:
        print("\nDownload complete. Verify chest_xray folder exists under data/raw/")


if __name__ == "__main__":
    main()
