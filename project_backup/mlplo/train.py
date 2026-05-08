from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from datasets import load_from_disk
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from .common import (
    CHECKPOINT_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_TARGET_MAX_LENGTH,
    build_compute_metrics,
    ensure_project_dirs,
    load_tokenizer,
    maybe_limit_split,
    resolve_mixed_precision,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BART on a prepared summarization dataset."
    )
    parser.add_argument(
        "--dataset-dir", required=True, help="Path produced by mlplo.data_cleaning."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=str(CHECKPOINT_DIR / "bart-large-xsum"))
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)   # lower LR for large model
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)  # more epochs + early stopping
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--label-smoothing", type=float, default=0.1)   # regularisation
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--generation-max-length", type=int, default=DEFAULT_TARGET_MAX_LENGTH
    )
    parser.add_argument("--generation-num-beams", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--run-test-eval",
        action="store_true",
        help="Run an additional evaluation pass on the held-out test split.",
    )
    return parser.parse_args()


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Handle output directory creation / overwriting safely."""
    if not output_dir.exists() or not any(output_dir.iterdir()):
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if not overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' is not empty. "
            "Pass --overwrite-output-dir to replace it."
        )

    # Atomic-ish overwrite: move to a temp name, then delete
    tmp = output_dir.parent / (output_dir.name + ".__tmp_delete")
    try:
        output_dir.rename(tmp)
        shutil.rmtree(tmp)
    except Exception:
        # If rename failed, try in-place rmtree as fallback
        if tmp.exists():
            shutil.rmtree(tmp)
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)

    # ── Validate dataset path ─────────────────────────────────────────────────
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {dataset_path}\n"
            "Run mlplo.data_cleaning first."
        )

    # ── Load dataset splits ───────────────────────────────────────────────────
    LOGGER.info("Loading prepared dataset from %s", dataset_path)
    tokenized_dataset = load_from_disk(str(dataset_path))

    required = {"train", "validation"}
    missing = required - set(tokenized_dataset.keys())
    if missing:
        raise KeyError(
            f"Dataset at '{dataset_path}' is missing required splits: {missing}. "
            "Re-run mlplo.data_cleaning to regenerate the dataset."
        )

    train_dataset = maybe_limit_split(tokenized_dataset["train"], args.max_train_samples)
    eval_dataset = maybe_limit_split(tokenized_dataset["validation"], args.max_eval_samples)
    has_test = "test" in tokenized_dataset
    test_dataset = (
        maybe_limit_split(tokenized_dataset["test"], args.max_test_samples)
        if has_test
        else None
    )

    # ── Validate resume-from-checkpoint ──────────────────────────────────────
    resume_path = args.resume_from_checkpoint
    if resume_path is not None and not Path(resume_path).exists():
        raise FileNotFoundError(
            f"--resume-from-checkpoint path does not exist: {resume_path}"
        )

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir, overwrite=args.overwrite_output_dir)
    metrics_dir = output_dir / "metrics"

    # ── Model + tokenizer ─────────────────────────────────────────────────────
    LOGGER.info("Loading tokenizer and model '%s'…", args.model_name)
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            LOGGER.warning(
                "Model '%s' does not support gradient_checkpointing_enable(); skipping.",
                args.model_name,
            )

    precision = resolve_mixed_precision()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # BERTScore is intentionally excluded from training-time compute_metrics.
    # It downloads a ~400 MB model and is 10-20× slower than ROUGE.
    # Use mlplo.eval with --include-bertscore for BERTScore evaluation.
    compute_metrics = build_compute_metrics(tokenizer, include_bertscore=False)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    LOGGER.info("Starting training…")
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    write_json(metrics_dir / "train_metrics.json", train_result.metrics)

    LOGGER.info("Running final validation…")
    validation_metrics = trainer.evaluate(
        eval_dataset=eval_dataset, metric_key_prefix="validation"
    )
    write_json(metrics_dir / "validation_metrics.json", validation_metrics)

    if args.run_test_eval:
        if test_dataset is None:
            LOGGER.warning(
                "--run-test-eval requested but dataset has no 'test' split; skipping."
            )
        else:
            LOGGER.info("Running held-out test evaluation…")
            test_metrics = trainer.evaluate(
                eval_dataset=test_dataset, metric_key_prefix="test"
            )
            write_json(metrics_dir / "test_metrics.json", test_metrics)

    # Free GPU memory before any downstream process reuses the device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info("Training complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
