from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from .common import (
    CACHE_DIR,
    DEFAULT_DATASET_NAME,
    DEFAULT_INPUT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_SUMMARY_COLUMN,
    DEFAULT_TARGET_MAX_LENGTH,
    DEFAULT_TEXT_COLUMN,
    IS_WINDOWS,
    PROCESSED_DIR,
    build_preprocess_function,
    count_words,
    ensure_project_dirs,
    load_tokenizer,
    maybe_limit_split,
    normalize_text,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, filter, deduplicate, and tokenize XSum for BART."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--summary-column", default=DEFAULT_SUMMARY_COLUMN)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR / "xsum_bart_base"))
    parser.add_argument("--max-input-length", type=int, default=DEFAULT_INPUT_MAX_LENGTH)
    parser.add_argument("--max-target-length", type=int, default=DEFAULT_TARGET_MAX_LENGTH)
    parser.add_argument("--min-document-words", type=int, default=50)
    parser.add_argument("--max-document-words", type=int, default=1024)
    parser.add_argument("--min-summary-words", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Worker processes for dataset.map(). Forced to 1 on Windows.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use tiny split sizes (256/64/64) for a fast smoke-test.",
    )
    return parser.parse_args()


def clean_batch(
    batch: dict[str, list[str]], text_column: str, summary_column: str
) -> dict[str, list[str]]:
    return {
        text_column: [normalize_text(text) for text in batch[text_column]],
        summary_column: [normalize_text(text) for text in batch[summary_column]],
    }


def is_valid_example(
    example: dict[str, str],
    text_column: str,
    summary_column: str,
    min_document_words: int,
    max_document_words: int,
    min_summary_words: int,
) -> bool:
    document_length = count_words(example.get(text_column, ""))
    summary_length = count_words(example.get(summary_column, ""))
    return (
        min_document_words <= document_length <= max_document_words
        and summary_length >= min_summary_words
        and bool(example.get(text_column, "").strip())
        and bool(example.get(summary_column, "").strip())
    )


def deduplicate_split(split: Dataset, text_column: str) -> tuple[Dataset, int]:
    """Remove exact-duplicate documents using a hash set (O(n) time)."""
    seen: set[str] = set()
    keep: list[int] = []
    for index, example in enumerate(split):
        doc = example[text_column]
        if doc in seen:
            continue
        seen.add(doc)
        keep.append(index)
    removed = len(split) - len(keep)
    return split.select(keep), removed


def _safe_output_dir(output_dir: Path) -> None:
    """Raise FileExistsError if the directory is non-empty, with PermissionError guard."""
    if not output_dir.exists():
        return
    try:
        non_empty = any(output_dir.iterdir())
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot read output directory '{output_dir}'. "
            "It may be locked by another process (e.g. OneDrive sync)."
        ) from exc
    if non_empty:
        raise FileExistsError(
            f"Output directory '{output_dir}' is not empty. "
            "Choose a new path or clear it first."
        )


def _resolve_num_proc(requested: int) -> int:
    """Force num_proc=1 on Windows; warn if the user asked for more."""
    if IS_WINDOWS and requested > 1:
        LOGGER.warning(
            "Multiprocessing with num_proc=%d is unreliable on Windows "
            "(datasets uses fork). Falling back to num_proc=1.",
            requested,
        )
        return 1
    return requested


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    ensure_project_dirs()

    # ── Validate length arguments ──────────────────────────────────────────────
    if args.max_input_length <= args.max_target_length:
        raise ValueError(
            f"--max-input-length ({args.max_input_length}) must be greater than "
            f"--max-target-length ({args.max_target_length})."
        )

    # ── Debug mode: use None-safe check so --train-samples 0 is respected ─────
    if args.debug:
        if args.train_samples is None:
            args.train_samples = 256
        if args.validation_samples is None:
            args.validation_samples = 64
        if args.test_samples is None:
            args.test_samples = 64

    output_dir = Path(args.output_dir)
    _safe_output_dir(output_dir)

    num_proc = _resolve_num_proc(args.num_proc)

    # ── Load dataset ───────────────────────────────────────────────────────────
    LOGGER.info("Loading dataset '%s'…", args.dataset_name)
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{args.dataset_name}'. "
            "Check your internet connection and dataset name."
        ) from exc

    # ── Validate expected splits exist ────────────────────────────────────────
    required_splits = {"train", "validation", "test"}
    missing = required_splits - set(dataset.keys())
    if missing:
        LOGGER.warning(
            "Dataset '%s' is missing splits: %s. Skipping those splits.",
            args.dataset_name,
            missing,
        )

    subset_limits = {
        "train": args.train_samples,
        "validation": args.validation_samples,
        "test": args.test_samples,
    }
    dataset = DatasetDict(
        {
            split_name: maybe_limit_split(split, subset_limits.get(split_name))
            for split_name, split in dataset.items()
        }
    )

    # ── Normalize ──────────────────────────────────────────────────────────────
    LOGGER.info("Normalizing text…")
    dataset = dataset.map(
        clean_batch,
        batched=True,
        fn_kwargs={
            "text_column": args.text_column,
            "summary_column": args.summary_column,
        },
        num_proc=num_proc,
        desc="Whitespace cleanup",
    )

    # ── Filter ────────────────────────────────────────────────────────────────
    LOGGER.info("Filtering unusable rows…")
    dataset = dataset.filter(
        is_valid_example,
        fn_kwargs={
            "text_column": args.text_column,
            "summary_column": args.summary_column,
            "min_document_words": args.min_document_words,
            "max_document_words": args.max_document_words,
            "min_summary_words": args.min_summary_words,
        },
        num_proc=num_proc,
        desc="Length filtering",
    )

    # ── Deduplicate ───────────────────────────────────────────────────────────
    dedupe_report: dict[str, int] = {}
    deduped_splits: dict[str, Dataset] = {}
    LOGGER.info("Deduplicating rows…")
    for split_name, split in dataset.items():
        deduped_split, removed = deduplicate_split(split, args.text_column)
        deduped_splits[split_name] = deduped_split
        dedupe_report[split_name] = removed
    dataset = DatasetDict(deduped_splits)

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.model_name)
    preprocess_fn = build_preprocess_function(
        tokenizer=tokenizer,
        text_column=args.text_column,
        summary_column=args.summary_column,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    LOGGER.info("Tokenizing rows…")
    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=num_proc,
        desc="Tokenization",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    LOGGER.info("Saving tokenized dataset to %s", output_dir)
    tokenized_dataset.save_to_disk(str(output_dir))

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "model_name": args.model_name,
        "text_column": args.text_column,
        "summary_column": args.summary_column,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "subset_limits": subset_limits,
        "splits": {name: len(split) for name, split in tokenized_dataset.items()},
        "duplicates_removed": dedupe_report,
    }
    write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Finished preprocessing. Split sizes: %s", manifest["splits"])


if __name__ == "__main__":
    main()
