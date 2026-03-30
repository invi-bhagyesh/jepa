import re
from pathlib import Path
import pandas as pd
import numpy as np

FEYNMAN_DIR_DEFAULT = "data/feynman"


def load_equations(csv_path):
    df = pd.read_csv(csv_path)
    # standardize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def load_features(data_dir, equation_name):
    """Load feature data for a specific equation from the extracted tarball."""
    data_dir = Path(data_dir)
    # files are named like "I.6.20a" matching the equation filename
    candidates = list(data_dir.glob(f"**/{equation_name}"))
    if not candidates:
        # try without dots
        clean = equation_name.replace(".", "_")
        candidates = list(data_dir.glob(f"**/{clean}"))
    if not candidates:
        return None
    return np.loadtxt(candidates[0])


# regex tokenizer for mathematical equations
# order: multi-char functions first, then operators, then variables/numbers
EQUATION_TOKEN_PATTERN = re.compile(
    r'sqrt|exp|log|ln|'                            # functions (short)
    r'arcsin|arccos|arctan|'                       # inverse trig (before sin/cos/tan)
    r'tanh|cosh|sinh|'                             # hyperbolic (before sin/cos/tan)
    r'asin|acos|atan|sin|cos|tan|'                 # trig
    r'\*\*|'                                       # exponentiation
    r'[+\-*/()^,]|'                               # operators
    r'\d+\.\d+|'                                   # decimal numbers
    r'[a-zA-Z_]\w*|'                               # variables (catches theta1, m_0, etc.)
    r'\d+'                                         # integers (last, so theta1 isn't split)
)

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class EquationTokenizer:
    def __init__(self):
        self.token2id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id2token = dict(enumerate(SPECIAL_TOKENS))

    @property
    def vocab_size(self):
        return len(self.token2id)

    @property
    def pad_id(self):
        return self.token2id["<PAD>"]

    def tokenize(self, equation):
        return EQUATION_TOKEN_PATTERN.findall(equation)

    def build_vocab(self, equations):
        from collections import Counter
        counts = Counter()
        for eq in equations:
            counts.update(self.tokenize(eq))
        for token, _ in counts.most_common():
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

    def encode(self, equation, add_special=True):
        tokens = self.tokenize(equation)
        ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]
        if add_special:
            ids = [self.token2id["<SOS>"]] + ids + [self.token2id["<EOS>"]]
        return ids

    def decode(self, ids, strip_special=True):
        tokens = [self.id2token.get(i, "<UNK>") for i in ids]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return " ".join(tokens)

    def save(self, path):
        with open(path, "w") as f:
            for token, idx in sorted(self.token2id.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")

    @classmethod
    def load(cls, path):
        tok = cls()
        tok.token2id = {}
        tok.id2token = {}
        with open(path) as f:
            for line in f:
                token, idx = line.strip().split("\t")
                idx = int(idx)
                tok.token2id[token] = idx
                tok.id2token[idx] = token
        return tok


def prepare_feynman_dataset(csv_path, data_dir=None):
    """Load equations, build tokenizer, return processed DataFrame."""
    df = load_equations(csv_path)

    # the 'Formula' column contains the target equations
    formula_col = None
    for col in df.columns:
        if 'formula' in col.lower() or 'equation' in col.lower():
            formula_col = col
            break
    if formula_col is None:
        # fallback: last column or try common names
        formula_col = df.columns[-1]

    equations = df[formula_col].dropna().astype(str).tolist()

    tokenizer = EquationTokenizer()
    tokenizer.build_vocab(equations)

    # add tokenized info to dataframe
    df["tokens"] = df[formula_col].apply(
        lambda x: tokenizer.tokenize(str(x)) if pd.notna(x) else []
    )
    df["token_count"] = df["tokens"].apply(len)

    return df, tokenizer, formula_col
