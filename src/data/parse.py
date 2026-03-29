import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def _detect_model(filename):
    name = Path(filename).stem.upper()
    if "QED" in name:
        return "QED"
    elif "QCD" in name:
        return "QCD"
    raise ValueError(f"Cannot determine physics model from filename: {filename}")


def parse_file(path):
    path = Path(path)
    model = _detect_model(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split on " : " to avoid splitting inside expressions with colons
            parts = re.split(r'\s*:\s*', line, maxsplit=3)
            if len(parts) != 4:
                continue
            rows.append({
                "interaction": parts[0].strip(),
                "feynman_diagram": parts[1].strip(),
                "amplitude": parts[2].strip(),
                "squared_amplitude": parts[3].strip(),
                "physics_model": model,
                "source_file": path.name,
            })
    return rows


def parse_all(data_dir, verbose=True):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt"))
    all_rows = []
    iterator = tqdm(files, desc="Parsing") if verbose else files
    for f in iterator:
        all_rows.extend(parse_file(f))
    return pd.DataFrame(all_rows)
