import re
from collections import Counter

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

# order matters: longer patterns first to avoid partial matches
TOKEN_PATTERN = re.compile(
    r'gamma_minus|gamma_plus|gamma_mu|'  # multi-char physics symbols
    r'g_s|'                               # strong coupling
    r'\*\*|'                              # exponentiation
    r'_\d+|'                              # normalized indices
    r'p\d|'                               # external momenta
    r'-?\d+/\d+|'                         # fractions
    r'-?\d+\.?\d*|'                       # integers / decimals
    r'[+\-*/(),{}]|'                      # single-char operators/delimiters
    r'[a-zA-Z_]\w*'                       # remaining identifiers
)


class PhysicsTokenizer:
    def __init__(self):
        self.token2id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id2token = dict(enumerate(SPECIAL_TOKENS))

    @property
    def vocab_size(self):
        return len(self.token2id)

    @property
    def pad_id(self):
        return self.token2id[PAD]

    @property
    def sos_id(self):
        return self.token2id[SOS]

    @property
    def eos_id(self):
        return self.token2id[EOS]

    def tokenize(self, expr):
        return TOKEN_PATTERN.findall(expr)

    def build_vocab(self, expressions):
        counts = Counter()
        for expr in expressions:
            counts.update(self.tokenize(expr))
        for token, _ in counts.most_common():
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

    def encode(self, expr, add_special=True):
        tokens = self.tokenize(expr)
        ids = [self.token2id.get(t, self.token2id[UNK]) for t in tokens]
        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, strip_special=True):
        tokens = [self.id2token.get(i, UNK) for i in ids]
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
