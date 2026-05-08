# Text Summarization Tool

This repo contains an end-to-end abstractive summarization project built around Hugging Face Transformers, the XSum dataset, and a Gradio demo app.

## Project Layout

```text
requirements.txt
mlplo/
  app.py            # Gradio UI for inference (single + batch mode)
  common.py         # Shared utilities
  compare.py        # Compare two models side-by-side
  data_cleaning.py  # Dataset preparation
  eval.py           # Standalone evaluation (ROUGE + BERTScore)
  report.py         # HTML Evaluation Report generator
  train.py          # Fine-tuning loop
tests/              # Pytest suite
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare a small debug dataset first:

```bash
python -m mlplo.data_cleaning --debug --output-dir mlplo/data/processed/xsum_debug
```

4. Run a smoke-test training job:

```bash
python -m mlplo.train --dataset-dir mlplo/data/processed/xsum_debug --output-dir mlplo/checkpoints/bart-base-xsum-debug --num-train-epochs 1 --per-device-train-batch-size 2 --per-device-eval-batch-size 2 --gradient-accumulation-steps 2 --run-test-eval
```

5. Evaluate the trained checkpoint:

```bash
python -m mlplo.eval --dataset-dir mlplo/data/processed/xsum_debug --model-path mlplo/checkpoints/bart-base-xsum-debug --include-bertscore
```

6. Generate an Evaluation Report:

```bash
python -m mlplo.report --checkpoint-dir mlplo/checkpoints/bart-base-xsum-debug
```

7. Launch the Gradio app:

```bash
python -m mlplo.app --model-path mlplo/checkpoints/bart-base-xsum-debug
```

## Running Tests

To run the full test suite for edge cases:
```bash
python -m pytest tests/ -v
```

## Colab Portability

The scripts are path-based and CLI-driven, so the same commands work in Google Colab after cloning the repo and installing `requirements.txt`. If you want a faster first pass, keep using `--debug` or provide `--train-samples`, `--validation-samples`, and `--test-samples`.

## Notes

- Training defaults to `facebook/bart-base` for fine-tuning.
- The Gradio app falls back to `facebook/bart-large-xsum` if no local checkpoint is supplied, which makes the UI useful before fine-tuning finishes.
- Mixed precision is enabled automatically when CUDA is available.
- BERTScore is excluded from the training loop (to keep it fast) and is opt-in for evaluation using the `--include-bertscore` flag.
