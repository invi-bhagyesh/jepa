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
            # The raw data format uses " : " as field separator, but the
            # Feynman diagram field itself contains colons (e.g. "Vertex V_0:u(X_1)").
            # The actual amplitude and squared amplitude are the LAST two
            # colon-separated fields. Split from the right to get them correctly.
            parts = line.split(" : ")
            if len(parts) < 4:
                continue
            # Last field = squared amplitude, second-to-last = amplitude
            # Everything before that = interaction + Feynman diagram
            squared_amplitude = parts[-1].strip()
            amplitude = parts[-2].strip()
            # First part is always the interaction type
            interaction = parts[0].strip()
            # Everything in between is the Feynman diagram
            feynman_diagram = " : ".join(parts[1:-2]).strip()

            rows.append({
                "interaction": interaction,
                "feynman_diagram": feynman_diagram,
                "amplitude": amplitude,
                "squared_amplitude": squared_amplitude,
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
