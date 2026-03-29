import re


def normalize_indices(amplitude, squared_amplitude):
    """Remap global indices (_N) to local sequential indices, jointly across
    both amplitude and squared_amplitude to preserve index correspondence."""
    index_map = {}
    counter = [1]

    def _replace(match):
        original = match.group(0)
        if original not in index_map:
            index_map[original] = f"_{counter[0]}"
            counter[0] += 1
        return index_map[original]

    pattern = re.compile(r'_\d+')
    norm_amp = pattern.sub(_replace, amplitude)
    norm_sq = pattern.sub(_replace, squared_amplitude)
    return norm_amp, norm_sq


def normalize_dataframe(df):
    """Apply joint index normalization to all rows in a DataFrame."""
    norm_amps, norm_sqs = [], []
    for _, row in df.iterrows():
        na, ns = normalize_indices(row["amplitude"], row["squared_amplitude"])
        norm_amps.append(na)
        norm_sqs.append(ns)
    df = df.copy()
    df["amplitude"] = norm_amps
    df["squared_amplitude"] = norm_sqs
    return df
