# SYMBA -- LM-JEPA for Squared Amplitude Calculation

Applying LM-JEPA self-supervised pretraining to the SYMBA squared amplitude prediction pipeline. First application of Joint Embedding Predictive Architectures to symbolic physics expressions.

## Tasks Completed

| Task              | Notebook                         | Description                                                                                                     |
| ----------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Common Task 1.1   | `00_feynman_preprocessing.ipynb` | AI Feynman dataset preprocessing -- equation tokenization with documented rationale                             |
| Common Task 1.2   | `01_data_preprocessing.ipynb`    | SYMBA amplitude dataset -- parsing, joint index normalization, physics-aware tokenization, 80-10-10 split       |
| Specific Test 2.5 | `02_jepa_training.ipynb`         | LM-JEPA pretraining + seq2seq finetuning on QED and QCD separately, baseline comparison, masking ratio ablation |

## Repository Structure

```
src/
  data/
    feynman.py        # AI Feynman equation loader and tokenizer (Task 1.1)
    download.py       # SYMBA dataset validation
    parse.py          # raw data parsing (interaction : diagram : amp : sq_amp)
    normalize.py      # joint index normalization across amp/sq_amp pairs
    tokenizer.py      # physics-aware word-level tokenizer (Task 1.2)
    dataset.py        # PyTorch dataset, collation, data loaders
  models/
    encoder.py        # transformer encoder with positional encoding (max_len=4096)
    decoder.py        # transformer decoder with causal masking
    jepa.py           # context encoder + EMA target encoder + predictor MLP
    seq2seq.py        # encoder-decoder with beam search decoding
  training/
    pretrain.py       # JEPA pretraining loop (smooth L1, cosine EMA schedule)
    finetune.py       # seq2seq supervised training (cross-entropy, early stopping)
    evaluate.py       # token accuracy, sequence exact match, top-5 accuracy
  utils/
    masking.py        # contiguous span mask generation
    scheduling.py     # EMA momentum and learning rate schedules

notebooks/
  00_feynman_preprocessing.ipynb    # Common Task 1.1
  01_data_preprocessing.ipynb       # Common Task 1.2
  02_jepa_training.ipynb            # Specific Test 2.5

proposal/
  proposal.tex                      # GSoC proposal (LaTeX)

data/                               # not committed
  feynman/            # AI Feynman dataset (downloaded by notebook 00)
  raw/                # SYMBA 17 .txt files (manual download)
  processed/          # generated splits (CSV) and vocab
```

## Setup

```bash
pip install -r requirements.txt
```

### Task 1.1 data

Notebook `00_feynman_preprocessing.ipynb` downloads both files automatically:

- `Feynman_with_units.tar.gz` (~4 GB, 100 feature files with 1M rows each)
- `FeynmanEquations.csv` (130 target equations)

### Task 1.2 data

Download the 17 data files from the [Box link](https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs) and place in `data/raw/`:

```
data/raw/
  QED-2-to-2-diag-TreeLevel-{0..9}.txt   (10 files)
  QCD-2-to-2-diag-TreeLevel-{0..6}.txt   (7 files)
```

## Running

Run notebooks in order on Colab or any GPU environment:

1. `00_feynman_preprocessing.ipynb` -- downloads AI Feynman data, tokenizes 130 equations (vocab: 117 tokens)
2. `01_data_preprocessing.ipynb` -- parses SYMBA data, normalizes indices jointly per sample, builds physics-aware tokenizer, 80-10-10 split, saves as CSV
3. `02_jepa_training.ipynb` -- JEPA pretrain (50 epochs) on all amplitudes, finetune QED and QCD separately (100 epochs, early stop), baseline comparison, masking ratio ablation (20%, 35%, 50%)

## Model Architecture

| Component           | Specification                                               |
| ------------------- | ----------------------------------------------------------- |
| Encoder             | 4-layer transformer, d=256, 4 heads, FFN=1024, max_len=4096 |
| Decoder             | 4-layer transformer, same dimensions, causal masking        |
| JEPA target encoder | EMA copy, momentum 0.996 -> 1.0 (cosine)                    |
| JEPA predictor      | 2-layer MLP: 256 -> 1024 -> 256                             |
| JEPA loss           | Smooth L1 on predicted vs target embeddings                 |
| Seq2seq loss        | Cross-entropy with teacher forcing                          |
| Inference           | Beam search, width 5                                        |

## Evaluation

| Metric               | Description                                     |
| -------------------- | ----------------------------------------------- |
| Token accuracy       | % correctly predicted tokens (ignoring padding) |
| Sequence exact match | % of full sequences predicted exactly           |
| Top-5 accuracy       | Correct sequence in top-5 beam outputs          |

QED and QCD reported separately. JEPA-pretrained vs baseline (no pretraining) comparison. Ablation over masking ratios (20%, 35%, 50%).

## References

- Alnuqaydan et al. (2023) -- [SYMBA: Symbolic Computation of Squared Amplitudes in HEP with ML](https://arxiv.org/abs/2206.08901)
- Huang, LeCun & Balestriero (2025) -- [LLM-JEPA: Large Language Models Meet JEPA](https://arxiv.org/abs/2509.14252)
- Bardhan et al. (2025) -- [HEP-JEPA: A Foundation Model for Collider Physics Using JEPA](https://arxiv.org/abs/2502.03933)
- Assran et al. (2023) -- [I-JEPA: Self-Supervised Learning from Images with JEPA](https://arxiv.org/abs/2301.08243)
- Lample & Charton (2019) -- [Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412)
