from __future__ import annotations

import argparse
import logging
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM

from .common import (
    ARTIFACT_DIR,
    DEFAULT_SUMMARY_COLUMN,
    DEFAULT_TEXT_COLUMN,
    ensure_project_dirs,
    load_tokenizer,
    maybe_limit_split,
    resolve_model_reference,
    validate_model_dir,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two models side-by-side on a test set."
    )
    parser.add_argument("--model-a", required=True, help="Path to Model A checkpoint.")
    parser.add_argument("--model-b", required=True, help="Path to Model B checkpoint.")
    parser.add_argument(
        "--dataset-dir", required=True, help="Prepared dataset directory."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--summary-column", default=DEFAULT_SUMMARY_COLUMN)
    parser.add_argument(
        "--output-file", default=str(ARTIFACT_DIR / "comparison.html")
    )
    return parser.parse_args()


@torch.inference_mode()
def generate_summaries(
    model_path: str, dataset, text_col: str, device: torch.device
) -> list[str]:
    ref = resolve_model_reference(model_path)
    validate_model_dir(ref)

    LOGGER.info(f"Loading {ref}...")
    tokenizer = load_tokenizer(ref)
    model = AutoModelForSeq2SeqLM.from_pretrained(ref).to(device)
    model.eval()

    predictions = []
    for item in dataset:
        text = item[text_col]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        out = model.generate(**inputs, max_length=128, num_beams=4)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        predictions.append(pred)

    del model
    torch.cuda.empty_cache()
    return predictions


def score_predictions(predictions: list[str], references: list[str]) -> dict:
    rouge = evaluate.load("rouge")
    r_res = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    
    from bert_score import score as bert_score_fn
    safe_preds = [p if p.strip() else "..." for p in predictions]
    safe_refs = [r if r.strip() else "..." for r in references]
    
    LOGGER.info("Computing BERTScore...")
    _, _, f1 = bert_score_fn(safe_preds, safe_refs, lang="en", verbose=False)
    
    return {
        "rouge1": r_res["rouge1"],
        "rouge2": r_res["rouge2"],
        "rougeL": r_res["rougeL"],
        "bertscore": float(f1.mean().item()),
    }


def generate_html(
    model_a_name: str,
    model_b_name: str,
    scores_a: dict,
    scores_b: dict,
    dataset,
    preds_a: list[str],
    preds_b: list[str],
    text_col: str,
    sum_col: str,
) -> str:
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            .better {{ background-color: #e8f5e9; font-weight: bold; color: #2e7d32; }}
            .source-col {{ width: 30%; font-size: 0.9em; color: #555; }}
            .ref-col {{ width: 20%; font-size: 0.9em; background: #fafafa; }}
            .pred-col {{ width: 25%; }}
        </style>
    </head>
    <body>
        <h1>Model Comparison</h1>
        
        <h2>Aggregate Scores</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Model A: {model_a_name}</th>
                <th>Model B: {model_b_name}</th>
            </tr>
    """

    for k in ["rouge1", "rouge2", "rougeL", "bertscore"]:
        va = scores_a[k]
        vb = scores_b[k]
        ca = "better" if va >= vb else ""
        cb = "better" if vb > va else ""
        html += f"""
            <tr>
                <td><strong>{k.upper()}</strong></td>
                <td class="{ca}">{va:.4f}</td>
                <td class="{cb}">{vb:.4f}</td>
            </tr>
        """

    html += """
        </table>
        
        <h2>Side-by-Side Predictions</h2>
        <table>
            <tr>
                <th>Source</th>
                <th>Reference</th>
                <th>Model A</th>
                <th>Model B</th>
            </tr>
    """

    for i, item in enumerate(dataset):
        html += f"""
            <tr>
                <td class="source-col">{item[text_col]}</td>
                <td class="ref-col">{item[sum_col]}</td>
                <td class="pred-col">{preds_a[i]}</td>
                <td class="pred-col">{preds_b[i]}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """
    return html


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    ensure_project_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info(f"Loading dataset {args.dataset_dir} (split: {args.split})...")
    dataset = load_from_disk(args.dataset_dir)[args.split]
    dataset = maybe_limit_split(dataset, args.max_samples)

    refs = [item[args.summary_column] for item in dataset]

    LOGGER.info("--- Processing Model A ---")
    preds_a = generate_summaries(args.model_a, dataset, args.text_column, device)
    scores_a = score_predictions(preds_a, refs)

    LOGGER.info("--- Processing Model B ---")
    preds_b = generate_summaries(args.model_b, dataset, args.text_column, device)
    scores_b = score_predictions(preds_b, refs)

    name_a = Path(args.model_a).name
    name_b = Path(args.model_b).name

    LOGGER.info("Generating HTML report...")
    html = generate_html(
        name_a,
        name_b,
        scores_a,
        scores_b,
        dataset,
        preds_a,
        preds_b,
        args.text_column,
        args.summary_column,
    )

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html, encoding="utf-8")
    
    LOGGER.info(f"Comparison report written to {out_file.absolute()}")


if __name__ == "__main__":
    main()
