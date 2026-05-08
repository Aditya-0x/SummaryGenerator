from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .common import ARTIFACT_DIR, existing_default_checkpoint

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML evaluation report.")
    parser.add_argument(
        "--checkpoint-dir",
        default=existing_default_checkpoint(),
        help="Path to the trained model checkpoint directory containing metrics.",
    )
    parser.add_argument(
        "--output-file",
        default=str(ARTIFACT_DIR / "eval_report.html"),
        help="Output HTML file path.",
    )
    return parser.parse_args()


def load_metrics(checkpoint_dir: Path) -> dict[str, dict[str, float]]:
    metrics = {}
    metrics_dir = checkpoint_dir / "metrics"
    if not metrics_dir.exists():
        return metrics

    for split in ["train", "validation", "test"]:
        file_path = metrics_dir / f"{split}_metrics.json"
        if file_path.exists():
            try:
                metrics[split] = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as e:
                LOGGER.warning(f"Failed to load {file_path}: {e}")
    return metrics


def load_predictions(checkpoint_dir: Path) -> list[dict]:
    # We look for the predictions file in the artifact directory,
    # since eval.py writes it there by default.
    pred_file = ARTIFACT_DIR / "sample_predictions.jsonl"
    preds = []
    if pred_file.exists():
        try:
            for line in pred_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    preds.append(json.loads(line))
        except Exception as e:
            LOGGER.warning(f"Failed to load predictions from {pred_file}: {e}")
    return preds


def generate_html(checkpoint_name: str, metrics: dict, predictions: list) -> str:
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report - {checkpoint_name}</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #fcfcfc; }}
            .metric-val {{ font-family: monospace; font-size: 1.1em; }}
            .pred-box {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #3498db; }}
            .pred-source {{ font-size: 0.9em; color: #666; margin-bottom: 10px; }}
            .pred-ref {{ font-weight: bold; color: #27ae60; margin-bottom: 5px; }}
            .pred-out {{ font-weight: bold; color: #8e44ad; }}
            .empty-warn {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <p><strong>Checkpoint:</strong> <code>{checkpoint_name}</code></p>

        <h2>Overall Metrics</h2>
        <table>
            <tr>
                <th>Split</th>
                <th>Loss</th>
                <th>ROUGE-1</th>
                <th>ROUGE-2</th>
                <th>ROUGE-L</th>
                <th>BERTScore F1</th>
                <th>Avg Gen Length</th>
            </tr>
    """

    for split in ["train", "validation", "test"]:
        m = metrics.get(split, {})
        if not m:
            continue
        
        prefix = split + "_" if split != "train" else ""
        
        loss = m.get(f"{prefix}loss", m.get("train_loss", "-"))
        r1 = m.get(f"{prefix}rouge1", "-")
        r2 = m.get(f"{prefix}rouge2", "-")
        rl = m.get(f"{prefix}rougeL", "-")
        bf1 = m.get(f"{prefix}bertscore_f1", "-")
        glen = m.get(f"{prefix}gen_len", "-")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        html += f"""
            <tr>
                <td><strong>{split.title()}</strong></td>
                <td class="metric-val">{fmt(loss)}</td>
                <td class="metric-val">{fmt(r1)}</td>
                <td class="metric-val">{fmt(r2)}</td>
                <td class="metric-val">{fmt(rl)}</td>
                <td class="metric-val">{fmt(bf1)}</td>
                <td class="metric-val">{fmt(glen)}</td>
            </tr>
        """

    html += """
        </table>
        
        <h2>Sample Predictions</h2>
    """

    if not predictions:
        html += "<p>No predictions found.</p>"
    else:
        for i, p in enumerate(predictions):
            empty_tag = " <span class='empty-warn'>(EMPTY PREDICTION)</span>" if p.get("empty_prediction") else ""
            html += f"""
            <div class="pred-box">
                <div class="pred-source"><strong>Source:</strong> {p.get("source", "")}</div>
                <div class="pred-ref">Target: {p.get("reference", "")}</div>
                <div class="pred-out">Model:{empty_tag} {p.get("prediction", "")}</div>
            </div>
            """

    html += """
    </body>
    </html>
    """
    return html


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.checkpoint_dir:
        LOGGER.error("No checkpoint directory provided or found.")
        return

    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists():
        LOGGER.error(f"Checkpoint directory not found: {checkpoint_path}")
        return

    metrics = load_metrics(checkpoint_path)
    predictions = load_predictions(checkpoint_path)

    html_content = generate_html(checkpoint_path.name, metrics, predictions)

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")
    
    LOGGER.info(f"Evaluation report generated at: {out_file.absolute()}")


if __name__ == "__main__":
    main()
