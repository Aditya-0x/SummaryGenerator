from __future__ import annotations

import argparse
import csv
import logging
import tempfile
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM

from .common import (
    DEFAULT_APP_FALLBACK_MODEL,
    DEFAULT_INPUT_MAX_LENGTH,
    default_device,
    ensure_project_dirs,
    existing_default_checkpoint,
    load_json,
    load_tokenizer,
    normalize_text,
    resolve_model_reference,
)

LOGGER = logging.getLogger(__name__)

try:
    import PyPDF2

    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# ── Generation Presets ────────────────────────────────────────────────────────
MODE_PRESETS = {
    "QUICK PULSE": {
        "max_new_tokens": 72,
        "min_new_tokens": 18,
        "num_beams": 4,
        "length_penalty": 1.25,
    },
    "KEY NOTES": {
        "max_new_tokens": 104,
        "min_new_tokens": 24,
        "num_beams": 5,
        "length_penalty": 1.05,
    },
    "DEEP CONTEXT": {
        "max_new_tokens": 152,
        "min_new_tokens": 34,
        "num_beams": 6,
        "length_penalty": 0.92,
    },
}

DEFAULT_MODE = "QUICK PULSE"

# ── Wonder Makers-inspired CSS ────────────────────────────────────────────────
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --black: #000000;
  --white: #FFFFFF;
  --lime: #D4FF00;
  --lime-dim: rgba(212, 255, 0, 0.15);
  --lime-glow: rgba(212, 255, 0, 0.08);
  --grey-100: #F5F5F5;
  --grey-400: #9CA3AF;
  --grey-600: #52525B;
  --grey-800: #27272A;
  --grey-900: #18181B;
  --border: rgba(255, 255, 255, 0.06);
  --border-hover: rgba(255, 255, 255, 0.12);
  --fn: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --mono: 'JetBrains Mono', monospace;
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

/* ─── Global Reset ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--black) !important;
  color: var(--white) !important;
  font-family: var(--fn) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  overflow-x: hidden;
}

/* Ambient glow — subtle purple/blue vignette like Wonder Makers */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 50% 50% at 0% 0%, rgba(120, 80, 255, 0.06), transparent 70%),
    radial-gradient(ellipse 40% 40% at 100% 100%, rgba(212, 255, 0, 0.03), transparent 60%);
  pointer-events: none;
  z-index: -1;
}

/* ─── Gradio Container Overrides ─── */
.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
  padding: 0 !important;
  background: transparent !important;
}

footer { display: none !important; }

/* Kill ALL default Gradio backgrounds */
.gradio-container, .gradio-container *,
.gr-box, .gr-panel, .gr-form, .gr-block,
[class*="block"], [class*="form"], [class*="panel"],
[class*="accordion"], [class*="markdown"] {
  background: transparent !important;
  color: var(--white) !important;
}

/* ─── HERO HEADER ─── */
.wm-hero {
  text-align: center;
  padding: 64px 24px 48px;
  position: relative;
}
.wm-hero h1 {
  font-family: var(--fn) !important;
  font-size: 3.2rem !important;
  font-weight: 900 !important;
  letter-spacing: -0.04em !important;
  text-transform: uppercase !important;
  line-height: 1.05 !important;
  margin: 0 0 16px 0 !important;
  background: linear-gradient(135deg, var(--white) 60%, var(--grey-400));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.wm-hero .wm-sub {
  font-size: 0.95rem;
  color: var(--grey-400);
  font-weight: 400;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0;
}
.wm-hero .wm-accent {
  display: inline-block;
  background: var(--lime);
  color: var(--black);
  font-weight: 700;
  font-size: 0.7rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  padding: 6px 18px;
  border-radius: 100px;
  margin-top: 20px;
}

/* ─── DIVIDER LINE ─── */
.wm-divider {
  height: 1px;
  background: var(--border);
  margin: 0 32px;
}

/* ─── WORKSPACE ─── */
.wm-workspace {
  display: grid !important;
  grid-template-columns: 1fr 1fr;
  gap: 2px;
  padding: 0 !important;
  margin: 0 !important;
}

.wm-pane {
  padding: 40px 36px !important;
  min-height: 480px;
  display: flex;
  flex-direction: column;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  position: relative;
}

/* Vertical separator between panes */
.wm-pane:first-child {
  border-right: 1px solid var(--border) !important;
}

.wm-pane-label {
  font-size: 0.65rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
  color: var(--grey-600) !important;
  margin-bottom: 24px !important;
  display: flex;
  align-items: center;
  gap: 10px;
}
.wm-pane-label .wm-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--lime);
  box-shadow: 0 0 8px var(--lime);
}
.wm-pane-label .wm-dot-cyan {
  background: #06b6d4;
  box-shadow: 0 0 8px rgba(6, 182, 212, 0.6);
}

/* ─── TEXT AREAS ─── */
.wm-input textarea, .wm-output textarea {
  background: rgba(255, 255, 255, 0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--white) !important;
  font-family: var(--fn) !important;
  font-size: 0.95rem !important;
  line-height: 1.8 !important;
  padding: 20px 24px !important;
  resize: none !important;
  transition: border-color 0.4s var(--ease), box-shadow 0.4s var(--ease) !important;
}
.wm-input textarea:focus {
  border-color: rgba(212, 255, 0, 0.3) !important;
  box-shadow: 0 0 0 4px var(--lime-glow), inset 0 1px 4px rgba(0,0,0,0.3) !important;
  outline: none !important;
}
.wm-input textarea::placeholder {
  color: var(--grey-600) !important;
  font-style: italic;
}

/* ─── BUTTONS ─── */
.wm-btn-primary {
  background: var(--lime) !important;
  color: var(--black) !important;
  font-family: var(--fn) !important;
  font-weight: 700 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 100px !important;
  padding: 16px 40px !important;
  cursor: pointer !important;
  transition: transform 0.3s var(--ease), box-shadow 0.3s var(--ease), background 0.3s !important;
}
.wm-btn-primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(212, 255, 0, 0.25) !important;
  background: #e0ff33 !important;
}
.wm-btn-primary:active {
  transform: translateY(0) !important;
}

.wm-btn-ghost {
  background: transparent !important;
  color: var(--grey-400) !important;
  font-family: var(--fn) !important;
  font-weight: 500 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border: 1px solid var(--border) !important;
  border-radius: 100px !important;
  padding: 14px 28px !important;
  cursor: pointer !important;
  transition: all 0.3s var(--ease) !important;
}
.wm-btn-ghost:hover {
  border-color: var(--grey-400) !important;
  color: var(--white) !important;
}

/* ─── ACTION ROW ─── */
.wm-actions {
  display: flex;
  gap: 12px;
  margin-top: 20px;
  align-items: center;
}

/* ─── TOKEN COUNTER ─── */
.wm-tokens {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.05em;
  margin-top: 12px;
}
.wm-tokens-normal { color: var(--grey-600) !important; }
.wm-tokens-warning {
  color: #FF6B6B !important;
  text-shadow: 0 0 12px rgba(255, 107, 107, 0.3);
}

/* ─── SIDEBAR ─── */
.wm-sidebar {
  background: rgba(0, 0, 0, 0.95) !important;
  border-right: 1px solid var(--border) !important;
  padding: 32px 24px !important;
}
.wm-sidebar h3, .wm-sidebar h4 {
  font-size: 0.6rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
  color: var(--grey-600) !important;
  margin-bottom: 16px !important;
}

/* ─── FILE UPLOAD ─── */
.wm-upload [data-testid="dropzone"] {
  border: 1px dashed var(--border) !important;
  border-radius: 12px !important;
  background: transparent !important;
  padding: 24px !important;
  transition: border-color 0.3s var(--ease) !important;
}
.wm-upload [data-testid="dropzone"]:hover {
  border-color: rgba(212, 255, 0, 0.3) !important;
}

/* ─── TABS ─── */
.tabs { border: none !important; }
button.tab-nav {
  font-family: var(--fn) !important;
  font-size: 0.65rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  color: var(--grey-600) !important;
  border: none !important;
  background: transparent !important;
  padding: 12px 24px !important;
  transition: color 0.3s !important;
}
button.tab-nav.selected {
  color: var(--white) !important;
  border-bottom: 2px solid var(--lime) !important;
}
button.tab-nav:hover { color: var(--white) !important; }

/* ─── ACCORDION ─── */
.wm-accordion button {
  font-family: var(--fn) !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--grey-400) !important;
  background: transparent !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* ─── MODEL INFO ─── */
.wm-model-info {
  padding: 20px 0;
  border-top: 1px solid var(--border);
  margin-top: 24px;
}
.wm-model-info p, .wm-model-info li {
  font-size: 0.8rem !important;
  color: var(--grey-400) !important;
  line-height: 1.7 !important;
}
.wm-model-info strong {
  color: var(--white) !important;
}

/* ─── BATCH TAB ─── */
.wm-batch-info {
  background: rgba(212, 255, 0, 0.04);
  border: 1px solid rgba(212, 255, 0, 0.1);
  border-radius: 12px;
  padding: 20px 24px;
  font-family: var(--mono);
  font-size: 0.8rem;
  line-height: 1.8;
  color: var(--grey-400);
  margin: 16px 0 24px;
}
.wm-batch-info strong {
  color: var(--lime);
  font-weight: 600;
}

/* ─── SLIDERS ─── */
input[type="range"] {
  accent-color: var(--lime) !important;
}

/* ─── RESPONSIVE ─── */
@media (max-width: 768px) {
  .wm-workspace { grid-template-columns: 1fr !important; }
  .wm-pane:first-child {
    border-right: none !important;
    border-bottom: 1px solid var(--border) !important;
  }
  .wm-hero h1 { font-size: 2rem !important; }
}
"""


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the ML summarization UI.")
    parser.add_argument("--model-path", default=existing_default_checkpoint())
    parser.add_argument("--fallback-model", default=DEFAULT_APP_FALLBACK_MODEL)
    parser.add_argument("--max-input-length", type=int, default=DEFAULT_INPUT_MAX_LENGTH)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_model_info(model_path: str) -> str:
    path = Path(model_path)
    if not path.exists():
        return f"**Hub Model** — `{model_path}`"
    info = f"**Checkpoint** — `{path.name}`\n"
    metrics_path = path / "metrics" / "test_metrics.json"
    if metrics_path.exists():
        try:
            m = load_json(metrics_path)
            r1 = m.get("test_rouge1", 0)
            rl = m.get("test_rougeL", 0)
            info += f"- ROUGE-1: **{r1:.4f}**\n- ROUGE-L: **{rl:.4f}**\n"
        except Exception:
            pass
    return info


def read_file_content(file_obj) -> str:
    if file_obj is None:
        return ""
    file_path = Path(file_obj.name)
    if file_path.suffix.lower() == ".pdf":
        if not HAS_PYPDF2:
            raise gr.Error("PyPDF2 is not installed. Run `pip install pypdf2` for PDF support.")
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            raise gr.Error(f"Failed to read PDF: {e}")
    else:
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise gr.Error(f"Failed to read file: {e}")


# ── Build the UI ──────────────────────────────────────────────────────────────
def build_demo(
    model, tokenizer, model_reference: str, max_input_length: int, device: torch.device
) -> gr.Blocks:
    default_preset = MODE_PRESETS[DEFAULT_MODE]

    def count_tokens(text: str) -> str:
        cleaned = normalize_text(text)
        if not cleaned:
            return f"<span class='wm-tokens-normal'>{0:03d} / {max_input_length} TOKENS</span>"
        tokens = tokenizer(cleaned, truncation=False)["input_ids"]
        count = len(tokens)
        if count > max_input_length:
            return (
                f"<span class='wm-tokens-warning'>⚠ {count:,} / {max_input_length} TOKENS "
                f"— INPUT WILL BE TRUNCATED</span>"
            )
        return f"<span class='wm-tokens-normal'>{count:,} / {max_input_length} TOKENS</span>"

    @torch.inference_mode()
    def summarize(text, max_new_tokens, min_new_tokens, num_beams, length_penalty):
        cleaned_text = normalize_text(text)
        if not cleaned_text:
            raise gr.Error("Please enter a document to summarize.")

        tokenized = tokenizer(
            cleaned_text, return_tensors="pt", truncation=True, max_length=max_input_length
        ).to(device)

        try:
            generated = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                min_length=min_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=3,
                early_stopping=True,
                max_time=45.0,
            )
        except torch.cuda.OutOfMemoryError:
            raise gr.Error(
                "CUDA Out of Memory. Reduce input length or beam count."
            )
        except Exception as e:
            raise gr.Error(f"Generation failed: {e}")

        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def batch_summarize(file_obj, max_new_tokens, min_new_tokens, num_beams, length_penalty):
        if file_obj is None:
            raise gr.Error("Upload a .txt file with one document per line.")
        try:
            lines = Path(file_obj.name).read_text(encoding="utf-8").splitlines()
        except Exception as e:
            raise gr.Error(f"Failed to read file: {e}")

        results = []
        for line in lines:
            if not line.strip():
                continue
            summary = summarize(line, max_new_tokens, min_new_tokens, num_beams, length_penalty)
            results.append({"source": line.strip(), "summary": summary})

        out_path = Path(tempfile.gettempdir()) / "batch_results.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "summary"])
            writer.writeheader()
            writer.writerows(results)
        return str(out_path)

    # ── Theme ─────────────────────────────────────────────────────────────────
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.lime,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.zinc,
    ).set(
        body_background_fill="#000000",
        block_background_fill="transparent",
        input_background_fill="rgba(255,255,255,0.02)",
        body_text_color="#FFFFFF",
        block_label_text_color="#52525B",
    )

    with gr.Blocks(title="Prism Studio", theme=theme) as demo:

        # Inject CSS via HTML since Gradio 6 moved css= to launch()
        gr.HTML(f"<style>{APP_CSS}</style>")

        # ── Hero Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="wm-hero">
            <h1>PRISM<br>STUDIO.</h1>
            <p class="wm-sub">Neural Text Summarization · Engineered</p>
            <span class="wm-accent">BART Fine-Tuned on XSum</span>
        </div>
        <div class="wm-divider"></div>
        """)

        # ── Sidebar ──────────────────────────────────────────────────────────
        with gr.Sidebar(elem_classes=["wm-sidebar"]):
            gr.HTML("<h3>Control Panel</h3>")
            mode_selector = gr.Dropdown(
                choices=list(MODE_PRESETS.keys()),
                value=DEFAULT_MODE,
                label="Generation Preset",
            )

            with gr.Accordion("Advanced Tuning", open=False, elem_classes=["wm-accordion"]):
                max_new_tokens = gr.Slider(
                    32, 256, value=default_preset["max_new_tokens"], step=8, label="Max tokens"
                )
                min_new_tokens = gr.Slider(
                    8, 96, value=default_preset["min_new_tokens"], step=4, label="Min tokens"
                )
                num_beams = gr.Slider(
                    1, 8, value=default_preset["num_beams"], step=1, label="Beams"
                )
                length_penalty = gr.Slider(
                    0.6, 2.0, value=default_preset["length_penalty"], step=0.05, label="Length penalty"
                )

            gr.HTML("<div class='wm-model-info'></div>")
            gr.HTML("<h4>Active Model</h4>")
            gr.Markdown(load_model_info(model_reference))

        # ── Tabs ─────────────────────────────────────────────────────────────
        with gr.Tabs():
            # ── STUDIO TAB ───────────────────────────────────────────────────
            with gr.Tab("STUDIO"):
                with gr.Row(elem_classes=["wm-workspace"]):
                    # Left — Source
                    with gr.Column(elem_classes=["wm-pane"]):
                        gr.HTML("""
                            <div class="wm-pane-label">
                                <span class="wm-dot"></span> SOURCE DOCUMENT
                            </div>
                        """)
                        file_upload = gr.File(
                            label="Upload .txt or .pdf",
                            file_types=[".txt", ".pdf"],
                            elem_classes=["wm-upload"],
                        )
                        input_text = gr.Textbox(
                            show_label=False,
                            placeholder="Paste your document here...",
                            lines=16,
                            elem_classes=["wm-input"],
                        )
                        token_display = gr.HTML(
                            f"<div class='wm-tokens'>"
                            f"<span class='wm-tokens-normal'>000 / {max_input_length} TOKENS</span>"
                            f"</div>"
                        )
                        with gr.Row(elem_classes=["wm-actions"]):
                            clear_btn = gr.Button("CLEAR", elem_classes=["wm-btn-ghost"])
                            summarize_btn = gr.Button("SUMMARIZE →", elem_classes=["wm-btn-primary"])

                    # Right — Output
                    with gr.Column(elem_classes=["wm-pane"]):
                        gr.HTML("""
                            <div class="wm-pane-label">
                                <span class="wm-dot wm-dot-cyan"></span> GENERATED OUTPUT
                            </div>
                        """)
                        output_text = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            lines=20,
                            elem_classes=["wm-output"],
                        )

            # ── BATCH TAB ────────────────────────────────────────────────────
            with gr.Tab("BATCH"):
                gr.HTML("""
                    <div class="wm-pane-label" style="padding: 32px 0 8px;">
                        <span class="wm-dot"></span> BULK INFERENCE
                    </div>
                """)
                gr.HTML("""
                    <div class="wm-batch-info">
                        <strong>TEMPLATE FORMAT</strong><br>
                        Line 1: First document to summarize.<br>
                        Line 2: Second document to summarize.<br>
                        Line 3: Third document to summarize.
                    </div>
                """)
                batch_upload = gr.File(
                    label="Upload batch .txt",
                    file_types=[".txt"],
                    elem_classes=["wm-upload"],
                )
                batch_btn = gr.Button("RUN BATCH →", elem_classes=["wm-btn-primary"])
                batch_download = gr.File(label="Download CSV Results", interactive=False)

        # ── Event Wiring ─────────────────────────────────────────────────────
        def update_params(mode):
            p = MODE_PRESETS[mode]
            return p["max_new_tokens"], p["min_new_tokens"], p["num_beams"], p["length_penalty"]

        mode_selector.change(
            update_params,
            inputs=[mode_selector],
            outputs=[max_new_tokens, min_new_tokens, num_beams, length_penalty],
        )
        file_upload.change(read_file_content, inputs=[file_upload], outputs=[input_text])
        input_text.change(count_tokens, inputs=[input_text], outputs=[token_display])
        summarize_btn.click(
            summarize,
            inputs=[input_text, max_new_tokens, min_new_tokens, num_beams, length_penalty],
            outputs=[output_text],
        )
        clear_btn.click(
            lambda: (
                None,
                "",
                f"<div class='wm-tokens'><span class='wm-tokens-normal'>000 / {max_input_length} TOKENS</span></div>",
                "",
            ),
            inputs=None,
            outputs=[file_upload, input_text, token_display, output_text],
        )
        batch_btn.click(
            batch_summarize,
            inputs=[batch_upload, max_new_tokens, min_new_tokens, num_beams, length_penalty],
            outputs=[batch_download],
        )

    return demo


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    ensure_project_dirs()

    model_reference = resolve_model_reference(args.model_path, fallback=args.fallback_model)
    device = default_device()

    LOGGER.info("Loading model from %s", model_reference)
    tokenizer = load_tokenizer(model_reference)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_reference)
    if getattr(model.generation_config, "max_length", None) == 20:
        model.generation_config.max_length = None
    model.to(device)
    model.eval()

    demo = build_demo(model, tokenizer, model_reference, args.max_input_length, device)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
