# Educational LLaDA-style Diffusion Language Model

A compact TensorFlow/Keras implementation of a **masked discrete-diffusion language model inspired by LLaDA**.

This repository is intended for learning and experimentation. It is deliberately described as **LLaDA-style**, not as a drop-in reproduction of the official implementation.

## Why this repo?

The official LLaDA repository provides pretrained models, inference, evaluation, and training guidelines, but not a full training framework. This repository complements it with a small TensorFlow/Keras implementation that can be **read, modified, trained, and inspected end-to-end**, making it especially useful for learning and experimentation.

## What it implements

The notebook includes:

- a fully bidirectional Transformer (`use_causal_mask=False`);
- a dedicated absorbing `MASK` token, separate from EOS;
- per-sequence masking levels and masked-token-only cross-entropy;
- random and low-confidence reverse-diffusion remasking;
- optional semi-autoregressive block variants;
- optional unsupervised classifier-free guidance (CFG) using a prompt-masked unconditional branch;
- exponential moving average (EMA) weights for evaluation and generation;
- memory-saving sparse vocabulary projection at masked positions;
- fixed-corruption diagnostics and pseudo-perplexity (PPP);
- explicit pipeline stages with timing and progress output so slow steps are easy to identify.

## Educational / experimental choices

The notebook intentionally exposes several choices that are **not required parts of the basic LLaDA objective**:

- an easy-to-hard corruption curriculum that eventually reaches `t_max = 1.0`;
- an optional early-training repetition regularizer (`rep_schedule_peak`; set to `0.0` to disable);
- optional Gumbel-max temperature annealing during generation (`tau_start` / `tau_end`; set both to `0.0` for greedy sampling);
- an optional left-token sampling repetition penalty (`repeat_penalty`; set to `0.0` to disable);
- a short guidance-off bootstrap at the beginning of CFG generation.

The prompt-masked CFG formula itself follows the unsupervised CFG form used by LLaDA.

## Repository contents

```text
.
├── LLaDA_style_diffusion_language_model.ipynb
├── README.md
├── requirements.txt
├── LICENSE
├── NOTICE.md
├── CITATION.cff
├── CHANGELOG.md
└── .gitignore
```

The repository intentionally stays notebook-first. Model checkpoints, downloaded WikiText data, and generated outputs are local runtime artifacts and are excluded from Git by default.

## Quick start

### 1. Create an environment

Python **3.10 or newer** is required because the notebook uses modern type-union syntax such as `float | None`.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open:

```text
LLaDA_style_diffusion_language_model.ipynb
```

### 2. Run the `quick` preset first

The notebook defaults to:

```python
RUN_PRESET = "quick"
```

The quick preset uses:

- embedding dimension: 384
- Transformer layers: 6
- attention heads: 6
- context length: 128
- batch size: 16
- training rounds: 8
- maximum training steps per round: 250
- training text limit: 5,000,000 characters
- validation windows: 128
- PPP positions per sequence: 32

A GPU is strongly recommended. The notebook prints the detected TensorFlow version, GPU devices, and precision policy before expensive work begins. If no GPU is detected, even the quick preset can be slow.

### 3. Use `full` only intentionally

The `full` preset is substantially larger:

- embedding dimension: 1024
- Transformer layers: 12
- attention heads: 16
- batch size: 16
- training rounds: 180
- maximum training steps per round: 4000
- full training corpus
- full validation split

Change the preset only when you intend to launch a large training run:

```python
RUN_PRESET = "full"
```

## Notebook layout

The notebook is split into two halves.

### Part I — Building blocks

Sections 1–8 define the model, data utilities, schedules, masking utilities, losses, sampler, EMA utilities, evaluation metrics, and training loop. No dataset is downloaded and no model is trained while merely defining these building blocks.

### Part II — Run the pipeline

The remaining cells execute each expensive stage separately:

1. inspect runtime and select a preset;
2. load/cache WikiText-103;
3. load the GPT-2 tokenizer and tokenize;
4. build training/validation windows;
5. prepare a held-out prompt;
6. visualize the forward masking process;
7. train;
8. inspect training history;
9. generate;
10. evaluate masked bits per token (BPT);
11. optionally evaluate pseudo-perplexity (PPP).

Each stage prints timing information. This makes it much easier to distinguish data loading, tokenization, TensorFlow tracing, training, validation, generation, and PPP evaluation when diagnosing a slow run.

## Data

The notebook uses the Hugging Face `Salesforce/wikitext` dataset, specifically `wikitext-103-raw-v1`.

On a first run, the source dataset may need to be downloaded and cached before the quick preset applies its runtime text limit. The local corpus cache is stored below `data/`, which is ignored by Git.

Dataset card:

- https://huggingface.co/datasets/Salesforce/wikitext

Original paper:

- Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. *Pointer Sentinel Mixture Models*. 2016. arXiv:1609.07843.

This repository does **not** redistribute WikiText.

## Metrics

The notebook reports several diagnostics:

- **masked CE / masked perplexity at fixed `t=0.15`**;
- **masked BPT** at selected mask rates;
- **pseudo-perplexity (PPP)** via leave-one-out masking.

These metrics should **not be compared directly with standard autoregressive perplexity**. A bidirectional masked model can condition on tokens on both sides of the prediction target.

PPP is intentionally placed in its own optional cell because it is one of the most computationally expensive evaluations in the notebook.

## Reproducibility

The notebook seeds NumPy and TensorFlow and uses deterministic validation subsampling for the quick preset.

For a publication-quality release:

1. restart the kernel;
2. run all cells from top to bottom using `quick`;
3. confirm that the notebook completes without manual intervention;
4. save the notebook with the quick-run outputs visible;
5. record the exact package versions from the successful environment if strict archival reproducibility is required.

`requirements.txt` uses conservative version ranges rather than claiming an exact environment that has not been independently reproduced on every platform.

## Relationship to LLaDA

This project is an independent educational TensorFlow/Keras implementation inspired by:

> Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. **Large Language Diffusion Models.** arXiv:2502.09992, 2025.

Paper:

- https://arxiv.org/abs/2502.09992

Official repository:

- https://github.com/ML-GSAI/LLaDA

Please cite the original LLaDA work when using this repository in research or teaching materials that build on its ideas.

## License

The source code and notebook in this repository are released under the MIT License. See `LICENSE`.

The MIT License applies to this repository's code only. Third-party datasets, papers, software packages, model weights, and other resources remain subject to their own licenses and terms. See `NOTICE.md`.