from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from datasets import Dataset
import torch
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "mlplo"
DATA_DIR = PACKAGE_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints"
ARTIFACT_DIR = PACKAGE_ROOT / "artifacts"

DEFAULT_MODEL_NAME = "facebook/bart-large-xsum"
DEFAULT_DATASET_NAME = "xsum"
DEFAULT_TEXT_COLUMN = "document"
DEFAULT_SUMMARY_COLUMN = "summary"
DEFAULT_APP_FALLBACK_MODEL = "Adive01/bart-large-xsum-finetuned"
DEFAULT_INPUT_MAX_LENGTH = 1024
DEFAULT_TARGET_MAX_LENGTH = 96

# datasets uses fork-based multiprocessing which is unreliable on Windows
IS_WINDOWS = sys.platform == "win32"


# ── Directory helpers ──────────────────────────────────────────────────────────

def ensure_project_dirs() -> None:
    for directory in (DATA_DIR, PROCESSED_DIR, CACHE_DIR, CHECKPOINT_DIR, ARTIFACT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalize_text(text: object) -> str:
    """Coerce *any* value to a clean, readable string stripped of web artifacts.

    Removes noise that degrades BART's summaries when text is pasted from websites:
    cookie banners, share buttons, ad labels, bylines, etc.
    """
    if text is None:
        return ""
    try:
        raw = str(text)
    except Exception:
        return ""

    # Normalise whitespace first
    cleaned = raw.replace("\u00a0", " ")
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

    # Strip common web-page junk patterns
    WEB_JUNK = [
        r"scroll down for video\.?",
        r"advertisement\.?",
        r"share this article\.?",
        r"click here to\s+\w+[^.]*\.",
        r"cookie(s)? (policy|notice|settings)[^.]*\.",
        r"by [A-Z][a-z]+ [A-Z][a-z]+\s*\|",   # bylines "By John Smith |"
        r"\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
        r"published:?\s*\d{1,2}[:/]\d{1,2}",
        r"updated:?\s*\d{1,2}[:/]\d{1,2}",
        r"follow us on (twitter|facebook|instagram|linkedin)[^.]*\.",
        r"subscribe (to|for)[^.]*\.",
        r"sign up[^.]*newsletter[^.]*\.",
        r"\[.*?\]",       # [image caption], [video], etc.
        r"read more:?[^.]*\.",
        r"related:?[^.]*\.",
    ]
    for pattern in WEB_JUNK:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()



def count_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def shorten_model_name(path_or_name: str) -> str:
    if not path_or_name:
        return "Unknown"
    path = Path(path_or_name)
    if path.exists() or path.is_absolute():
        return path.name
    return path_or_name


# ── I/O helpers ────────────────────────────────────────────────────────────────

def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True) for row in rows]
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Model / tokenizer helpers ──────────────────────────────────────────────────

def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_preprocess_function(
    tokenizer,
    text_column: str,
    summary_column: str,
    max_input_length: int,
    max_target_length: int,
) -> Callable[[dict[str, list[str]]], dict[str, list[list[int]]]]:
    """Return a batched map function that tokenizes source + target texts."""

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        if text_column not in batch:
            raise KeyError(
                f"Text column '{text_column}' not found in batch. "
                f"Available columns: {list(batch.keys())}"
            )
        if summary_column not in batch:
            raise KeyError(
                f"Summary column '{summary_column}' not found in batch. "
                f"Available columns: {list(batch.keys())}"
            )
        model_inputs = tokenizer(
            batch[text_column],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[summary_column],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def resolve_mixed_precision() -> dict[str, bool]:
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    try:
        bf16_available = torch.cuda.is_bf16_supported()
    except (AttributeError, RuntimeError, AssertionError):
        bf16_available = False
    return {"fp16": not bf16_available, "bf16": bf16_available}


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def existing_default_checkpoint() -> str | None:
    """Return the most recently modified valid checkpoint directory, or None.

    A directory is considered a valid checkpoint if it contains either
    ``model.safetensors`` or ``pytorch_model.bin``.
    """
    if not CHECKPOINT_DIR.exists():
        return None
    candidates: list[Path] = []
    for entry in CHECKPOINT_DIR.rglob("*"):
        if entry.is_dir():
            has_model = (
                (entry / "model.safetensors").exists()
                or (entry / "pytorch_model.bin").exists()
            )
            if has_model:
                candidates.append(entry)
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def resolve_model_reference(path_or_name: str | None, fallback: str | None = None) -> str:
    if path_or_name:
        candidate = Path(path_or_name)
        return str(candidate.resolve()) if candidate.exists() else path_or_name
    if fallback:
        return fallback
    raise ValueError("A model path or model name is required.")


def validate_model_dir(path: str | Path) -> None:
    """Raise FileNotFoundError with a clear message if a checkpoint dir looks incomplete."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {p}")
    has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    if not has_weights:
        raise FileNotFoundError(
            f"No model weights found in '{p}'. "
            "Expected 'model.safetensors' or 'pytorch_model.bin'."
        )


# ── Dataset helpers (single source of truth) ──────────────────────────────────

def maybe_limit_split(split: Dataset, limit: int | None) -> Dataset:
    """Select the first *limit* rows from a Dataset split, or return it unchanged."""
    if limit is None or limit >= len(split):
        return split
    return split.select(range(limit))


# ── Metrics (single source of truth) ──────────────────────────────────────────

def build_compute_metrics(tokenizer, *, include_bertscore: bool = False):
    """Return a ``compute_metrics`` callable suitable for ``Seq2SeqTrainer``.

    Parameters
    ----------
    tokenizer:
        Used to decode predicted and label token IDs.
    include_bertscore:
        When ``True``, also compute BERTScore F1 (requires ``bert-score``).
        Keep ``False`` during training — BERTScore downloads a ~400 MB model
        on first use and is 10-20× slower than ROUGE.  Set ``True`` only for
        standalone evaluation passes (``mlplo.eval``).
    """
    import evaluate  # deferred: keeps module importable without evaluate installed

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.asarray(predictions)
        predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions = [p.strip() for p in decoded_predictions]
        decoded_labels = [lb.strip() for lb in decoded_labels]

        rouge_result = rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )

        prediction_lengths = [
            int(np.count_nonzero(pred != tokenizer.pad_token_id))
            for pred in predictions
        ]

        metrics: dict[str, float] = {
            "rouge1": round(rouge_result["rouge1"], 4),
            "rouge2": round(rouge_result["rouge2"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
            "gen_len": round(float(np.mean(prediction_lengths)), 2),
        }

        if include_bertscore:
            from bert_score import score as bert_score_fn

            LOGGER.info("Computing BERTScore (downloads model on first use)…")
            safe_preds = [p if p.strip() else "..." for p in decoded_predictions]
            safe_labels = [lb if lb.strip() else "..." for lb in decoded_labels]
            _, _, F1 = bert_score_fn(safe_preds, safe_labels, lang="en", verbose=False)
            metrics["bertscore_f1"] = round(float(F1.mean().item()), 4)

        return metrics

    return compute_metrics
