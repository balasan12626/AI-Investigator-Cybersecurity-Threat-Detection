"""Download the public benchmark datasets used by the ML training scripts.

UNSW-NB15  -> data/external/unsw_nb15/   (~46 MB)  used by src/train_unsw_nb15.py
CIC-IDS2017 -> data/external/cicids2017/ (~260 MB) used by src/train_cicids2017.py

Usage:
    python src/download_datasets.py            # both datasets
    python src/download_datasets.py unsw       # only UNSW-NB15
    python src/download_datasets.py cicids     # only CIC-IDS2017
"""

import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HF = "https://huggingface.co/datasets"

UNSW = {
    # NOTE: in this mirror test.csv is the official TRAINING set (175,341 rows)
    # and train.csv is the official TESTING set (82,332 rows).
    "train.csv": f"{HF}/Mireu-Lab/UNSW-NB15/resolve/main/train.csv",
    "test.csv": f"{HF}/Mireu-Lab/UNSW-NB15/resolve/main/test.csv",
}
CIC_FILES = [
    "Friday-WorkingHours-Afternoon-DDos", "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Morning", "Monday-WorkingHours",
    "Thursday-WorkingHours-Afternoon-Infilteration", "Thursday-WorkingHours-Morning-WebAttacks",
    "Tuesday-WorkingHours", "Wednesday-workingHours",
]
CIC = {f"{n}.parquet": f"{HF}/bvsam/cic-ids-2017/resolve/main/machine_learning/{n}.pcap_ISCX.csv.parquet"
       for n in CIC_FILES}


def fetch(files: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in files.items():
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  exists  {dest.relative_to(BASE_DIR)}")
            continue
        print(f"  download {dest.relative_to(BASE_DIR)}", flush=True)
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    which = sys.argv[1:] or ["unsw", "cicids"]
    if "unsw" in which:
        print("UNSW-NB15:")
        fetch(UNSW, BASE_DIR / "data" / "external" / "unsw_nb15")
    if "cicids" in which:
        print("CIC-IDS2017:")
        fetch(CIC, BASE_DIR / "data" / "external" / "cicids2017")
