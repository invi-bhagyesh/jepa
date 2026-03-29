import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def build_splits(df, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train = df.iloc[indices[:train_end]].reset_index(drop=True)
    val = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
    test = df.iloc[indices[val_end:]].reset_index(drop=True)
    return train, val, test


class AmplitudeDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.src = [tokenizer.encode(row["amplitude"]) for _, row in df.iterrows()]
        self.tgt = [tokenizer.encode(row["squared_amplitude"]) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx], dtype=torch.long), \
               torch.tensor(self.tgt[idx], dtype=torch.long)


def collate_fn(batch, pad_id=0):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(s)] = s
        tgt_padded[i, :len(t)] = t

    return src_padded, tgt_padded


def get_loader(df, tokenizer, batch_size=64, shuffle=True):
    ds = AmplitudeDataset(df, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, drop_last=False)
