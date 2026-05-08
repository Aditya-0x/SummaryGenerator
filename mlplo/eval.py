from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .common import (
    ARTIFACT_DIR,
    DEFAULT_SUMMARY_COLUMN,
    DEFAULT_TARGET_MAX_LENGTH,
    DEFAULT_TEXT_COLUMN,
    build_compute_metrics,
    ensure_project_dirs,
    existing_default_checkpoint,
    load_tokenizer,
    maybe_limit_split,
    resolve_mixed_precision,
    resolve_model_reference,
    validate_model_dir,
    write_json,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned summarization checkpoint."
    )
    parser.add_argument(
        "--dataset-dir", required=True, help="Path produced by mlplo.data_cleaning."
    )
    parser.add_argument("--model-path", default=existing_default_checkpoint())
    parser.add_argument(
        "--split", default="test", choices=["train", "validation", "test"]
    )
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--summary-column", default=DEFAULT_SUMMARY_COLUMN)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument(
        "--generation-max-length", type=int, default=DEFAULT_TARGET_MAX_LENGTH
    )
    parser.add_argument("--generation-num-beams", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--preview-rows", type=int, default=5)
    parser.add_argument(
        "--include-bertscore",
        action="store_true",
        help=(
            "Compute BERTScore F1 in addition to ROUGE. "
            "Downloads a ~400 MB model on first use."
        ),
    )
    parser.add_argument(
        "--output-file", default=str(ARTIFACT_DIR / "eval_metrics.json")
    )
    parser.add_argument(
        "--predictions-file", default=str(ARTIFACT_DIR / "sample_predictions.jsonl")
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    ensure_project_dirs()

    if not args.model_path:
        raise ValueError(
            "No model path provided and no default checkpoint exists yet. "
            "Train a model first with mlplo.train."
        )

    # ── Validate dataset path ─────────────────────────────────────────────────
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")

    # ── Validate model directory ──────────────────────────────────────────────
    model_reference = resolve_model_reference(args.model_path)
    validate_model_dir(model_reference)

    LOGGER.info("Loading dataset from %s", dataset_path)
    tokenized_dataset = load_from_disk(str(dataset_path))

    if args.split not in tokenized_dataset:
        available = list(tokenized_dataset.keys())
        raise KeyError(
            f"Split '{args.split}' not found in dataset. Available: {available}"
        )

    evaluation_split = maybe_limit_split(
        tokenized_dataset[args.split], args.max_samples
    )

    # ── Load model ────────────────────────────────────────────────────────────
    LOGGER.info("Loading model from %s", model_reference)
    tokenizer = load_tokenizer(model_reference)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_reference)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    precision = resolve_mixed_precision()

    if args.include_bertscore:
        LOGGER.info(
            "BERTScore enabled. A ~400 MB model will be downloaded on first use."
        )

    compute_metrics = build_compute_metrics(
        tokenizer, include_bertscore=args.include_bertscore
    )

    temp_output_dir = ARTIFACT_DIR / "eval_tmp"
    evaluation_args = Seq2SeqTrainingArguments(
        output_dir=str(temp_output_dir),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=evaluation_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    LOGGER.info("Running evaluation on split '%s'…", args.split)
    prediction_output = trainer.predict(evaluation_split, metric_key_prefix=args.split)
    metrics = prediction_output.metrics
    write_json(args.output_file, metrics)
    LOGGER.info("Metrics: %s", metrics)

    # ── Decode predictions and write sample file ──────────────────────────────
    generated_token_ids = prediction_output.predictions
    if isinstance(generated_token_ids, tuple):
        generated_token_ids = generated_token_ids[0]

    generated_token_ids = np.asarray(generated_token_ids)
    generated_token_ids = np.where(
        generated_token_ids < 0, tokenizer.pad_token_id, generated_token_ids
    )
    decoded_predictions = tokenizer.batch_decode(
        generated_token_ids, skip_special_tokens=True
    )

    # Guard against preview_rows exceeding available samples
    n_preview = min(args.preview_rows, len(decoded_predictions), len(evaluation_split))
    preview_rows = []
    for index in range(n_preview):
        row = evaluation_split[index]
        prediction = decoded_predictions[index].strip()
        record: dict = {
            "source": row.get(args.text_column, ""),
            "reference": row.get(args.summary_column, ""),
            "prediction": prediction,
        }
        if not prediction:
            record["empty_prediction"] = True
            LOGGER.warning("Empty prediction at index %d.", index)
        preview_rows.append(record)

    write_jsonl(args.predictions_file, preview_rows)
    LOGGER.info(
        "Evaluation complete. Metrics → %s | Predictions → %s",
        args.output_file,
        args.predictions_file,
    )


if __name__ == "__main__":
    main()
