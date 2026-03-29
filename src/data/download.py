import os
from pathlib import Path

BOX_URL = "https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs"

EXPECTED_FILES = [
    "QED-2-to-2-diag-TreeLevel-0.txt",
    "QED-2-to-2-diag-TreeLevel-1.txt",
    "QED-2-to-2-diag-TreeLevel-2.txt",
    "QED-2-to-2-diag-TreeLevel-3.txt",
    "QED-2-to-2-diag-TreeLevel-4.txt",
    "QED-2-to-2-diag-TreeLevel-5.txt",
    "QED-2-to-2-diag-TreeLevel-6.txt",
    "QED-2-to-2-diag-TreeLevel-7.txt",
    "QED-2-to-2-diag-TreeLevel-8.txt",
    "QED-2-to-2-diag-TreeLevel-9.txt",
    "QCD-2-to-2-diag-TreeLevel-0.txt",
    "QCD-2-to-2-diag-TreeLevel-1.txt",
    "QCD-2-to-2-diag-TreeLevel-2.txt",
    "QCD-2-to-2-diag-TreeLevel-3.txt",
    "QCD-2-to-2-diag-TreeLevel-4.txt",
    "QCD-2-to-2-diag-TreeLevel-5.txt",
    "QCD-2-to-2-diag-TreeLevel-6.txt",
]


def validate_dataset(data_dir):
    data_dir = Path(data_dir)
    found = sorted(f.name for f in data_dir.glob("*.txt"))
    missing = [f for f in EXPECTED_FILES if f not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} files in {data_dir}. "
            f"Download from: {BOX_URL}\n"
            f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return [data_dir / f for f in sorted(found)]


def get_data_files(data_dir="data/raw"):
    return validate_dataset(data_dir)
