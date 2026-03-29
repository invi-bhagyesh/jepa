# SYMBA -- LM-JEPA for Squared Amplitude Calculation

GSoC 2026 evaluation tasks for [ML4SCI / SYMBA](https://ml4sci.org/gsoc/projects/2026/project_SYMBA.html).

## Tasks Completed

- **Common Task 1.1**: AI Feynman dataset preprocessing -- equation tokenization with documented rationale
- **Common Task 1.2**: SYMBA amplitude dataset -- parsing, index normalization, physics-aware tokenization, 80-10-10 split
- **Specific Test 2.5**: LM-JEPA pretraining + seq2seq finetuning, trained separately on QED and QCD, baseline comparison, masking ratio ablation

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
    encoder.py        # transformer encoder with positional encoding
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

data/
  feynman/            # AI Feynman dataset (downloaded by notebook 00)
  raw/                # SYMBA 17 .txt files (manual download)
  processed/          # generated splits and vocab
```

## Setup

```bash
pip install -r requirements.txt
```

### Task 1.1 data

Notebook `00_feynman_preprocessing.ipynb` downloads both files automatically:
- `Feynman_with_units.tar.gz` (features)
- `FeynmanEquations.csv` (target equations)

### Task 1.2 data

Download the 17 data files from the [Box link](https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs) and place in `data/raw/`:

```
data/raw/
  QED-2-to-2-diag-TreeLevel-{0..9}.txt
  QCD-2-to-2-diag-TreeLevel-{0..6}.txt
```

## Running

Run notebooks in order:

1. `00_feynman_preprocessing.ipynb` -- downloads AI Feynman data, tokenizes equations, reports statistics
2. `01_data_preprocessing.ipynb` -- parses SYMBA data, normalizes indices, builds tokenizer, 80-10-10 split
3. `02_jepa_training.ipynb` -- JEPA pretrain on all amplitudes, finetune QED and QCD separately, baseline comparison, masking ratio ablation

## Model

- Encoder/Decoder: 4 layers, 256 hidden dim, 4 heads, 1024 FFN (~6-8M params)
- JEPA pretraining: span masking at 35%, smooth L1 loss, EMA target encoder (momentum 0.996 to 1.0)
- Seq2seq finetuning: teacher-forced cross-entropy, beam search (width 5) at inference
- Baseline: same architecture, no pretraining

## Evaluation

| Metric | Description |
|---|---|
| Token accuracy | % correctly predicted tokens (ignoring padding) |
| Sequence exact match | % of full sequences predicted exactly |
| Top-5 accuracy | correct sequence in top-5 beam outputs |

QED and QCD reported separately. JEPA-pretrained vs baseline (no pretraining) comparison. Ablation over masking ratios (20%, 35%, 50%).

## References

- Alnuqaydan et al. -- SYMBA: Symbolic Machine Learning for High Energy Physics Calculations
- Huang, LeCun & Balestriero (2025) -- LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures
- Bardhan et al. (2025) -- HEP-JEPA: A Foundation Model for Collider Physics Using JEPA
- Lample & Charton (2019) -- Deep Learning for Symbolic Mathematics
- Udrescu & Tegmark (2019) -- AI Feynman: A Physics-Inspired Method for Symbolic Regression
