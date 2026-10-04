# Changelog

All notable changes to the public educational notebook will be documented here.

## [1.0.0] - 2026-10-03

### Added

- Teaching-oriented notebook structure with explicit **Part I — Building blocks** and **Part II — Run the pipeline**.
- Separate pipeline cells for runtime inspection, WikiText loading, tokenizer loading/tokenization, window construction, prompt preparation, forward-process visualization, training, generation, masked-BPT evaluation, and optional PPP.
- `quick` and `full` presets.
- Deterministic validation subsampling for the quick preset.
- Per-stage timing, progress messages, and GPU / precision diagnostics.
- EMA evaluation/checkpointing and sparse masked-position vocabulary projection.

### Clarified

- The project is **LLaDA-style**, not a drop-in reproduction.
- Gumbel-max noise is temperature sampling, not merely a tie-break.
- Prompt-masked unsupervised CFG follows the LLaDA form; the early guidance-off bootstrap is notebook-specific.
- Fixed masked perplexity and PPP are not directly comparable with conventional autoregressive perplexity.

### Changed from the development notebook

- Renamed misleading GPT-specific / metric names.
- Changed the corruption curriculum to reach `t_max = 1.0`.
- Restricted low-confidence remasking candidates to currently masked positions.
- Made sampling temperature and repetition penalties explicit.
- Separated expensive final generation/BPT/PPP from the training function.
- Cleaned WikiText escaped punctuation before tokenization.
- Reorganized the former monolithic workflow for GitHub/teaching use.