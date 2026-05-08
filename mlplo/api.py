import logging
import os
import re
from contextlib import asynccontextmanager
from typing import List, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM

from google import genai

from .common import (
    DEFAULT_APP_FALLBACK_MODEL,
    DEFAULT_INPUT_MAX_LENGTH,
    default_device,
    existing_default_checkpoint,
    load_tokenizer,
    normalize_text,
    resolve_model_reference,
)

LOGGER = logging.getLogger(__name__)

ml_context = {}

# ── Gemini setup ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    LOGGER.info("Gemini API key loaded — client ready.")
else:
    LOGGER.warning("GEMINI_API_KEY not set — Gemini features will be unavailable.")

# ── Chunking constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 850   # tokens per chunk (well within BART's 1024 limit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = existing_default_checkpoint()
    model_reference = resolve_model_reference(model_path, fallback=DEFAULT_APP_FALLBACK_MODEL)
    device = default_device()

    LOGGER.info(f"Loading BART model from {model_reference}")
    tokenizer = load_tokenizer(model_reference)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_reference)

    if getattr(model.generation_config, "max_length", None) == 20:
        model.generation_config.max_length = None

    model.to(device)
    model.eval()

    ml_context["model"] = model
    ml_context["tokenizer"] = tokenizer
    ml_context["device"] = device
    ml_context["max_input_length"] = DEFAULT_INPUT_MAX_LENGTH

    yield
    ml_context.clear()


app = FastAPI(title="Prism Studio API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    text: str
    engine: str = "bart"            # "bart" | "gemini"
    max_new_tokens: int = 128
    min_new_tokens: int = 30
    num_beams: int = 4
    length_penalty: float = 1.5     # >1 encourages longer, more complete summaries
    gemini_model: str = "gemini-3.0-flash"
    polish: bool = False            # if True, run Gemini to clean up BART's output


class SummarizeResponse(BaseModel):
    summary: str
    engine_used: str
    chunks_processed: int = 1


# ── Sentence-aware text splitter ──────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Split text into sentences respecting abbreviations."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _build_sentence_chunks(text: str, tokenizer, max_tokens: int) -> List[str]:
    """
    Split text into chunks that respect sentence boundaries.
    Each chunk is at most max_tokens tokens long.
    Returns a list of text strings (not token IDs) — one per chunk.
    """
    sentences = _split_sentences(text)
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))

        # If adding this sentence would exceed the limit, flush current chunk
        if current_len + sent_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Keep the last sentence for overlap context
            current_sentences = [current_sentences[-1]] if current_sentences else []
            current_len = len(tokenizer.encode(current_sentences[0], add_special_tokens=False)) if current_sentences else 0

        current_sentences.append(sent)
        current_len += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ── BART: single-chunk inference ──────────────────────────────────────────────
def _bart_generate_one(text_chunk: str, request: SummarizeRequest) -> str:
    """Summarise a single text chunk with BART."""
    tokenizer = ml_context["tokenizer"]
    model = ml_context["model"]
    device = ml_context["device"]

    tokenized = tokenizer(
        text_chunk,
        return_tensors="pt",
        truncation=True,
        max_length=DEFAULT_INPUT_MAX_LENGTH,
        padding=False,
    ).to(device)

    try:
        with torch.inference_mode():
            # BART generation parameters
            gen_kwargs = {
                "max_new_tokens": request.max_new_tokens,
                "min_length": request.min_new_tokens,
                "length_penalty": request.length_penalty,
                "num_beams": request.num_beams,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.5,   # Strongly discourage hallucination by phrase reuse
            }
            generated = model.generate(
                **tokenized,
                **gen_kwargs
            )
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="CUDA Out of Memory — try a shorter document or fewer beams.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BART generation failed: {e}")

    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


# ── BART: hierarchical Map-Reduce ─────────────────────────────────────────────
def _bart_summarize(text: str, request: SummarizeRequest) -> Tuple[str, int]:
    """
    Sentence-aware Map-Reduce summarisation:
      MAP:    Split into sentence-boundary chunks → summarise each
      REDUCE: Summarise the combined chunk summaries
    Returns (final_summary, num_chunks).
    """
    tokenizer = ml_context["tokenizer"]
    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    # ── Single pass — text fits in BART's window ──────────────────────────────
    if total_tokens <= CHUNK_SIZE:
        return _bart_generate_one(text, request), 1

    # ── MAP — sentence-aware chunking ─────────────────────────────────────────
    chunks = _build_sentence_chunks(text, tokenizer, CHUNK_SIZE)
    LOGGER.info(f"Chunked document into {len(chunks)} sentence-aware chunks")

    chunk_summaries: List[str] = []
    for i, chunk in enumerate(chunks):
        LOGGER.info(f"Summarising chunk {i+1}/{len(chunks)}")
        chunk_summaries.append(_bart_generate_one(chunk, request))

    num_chunks = len(chunk_summaries)
    combined = " ".join(chunk_summaries)

    # ── REDUCE — summarise the combined chunk summaries ───────────────────────
    combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
    if combined_tokens <= CHUNK_SIZE:
        # Combined summaries are short enough for one final pass
        final = _bart_generate_one(combined, request)
    else:
        # Recursively reduce (handles extremely long documents)
        final, _ = _bart_summarize(combined, request)

    return final, num_chunks


# ── Gemini polish (optional post-processing of BART output) ──────────────────
def _gemini_polish(original_text: str, rough_summary: str, gemini_model: str) -> str:
    """Use Gemini to fact-check and rewrite BART's output based on the original document."""
    if not gemini_client:
        return rough_summary
    prompt = (
        "You are an expert editor. I will provide you with a SOURCE DOCUMENT and a ROUGH SUMMARY generated by a smaller AI.\n\n"
        "Your task is to produce a highly polished, professional, and detailed summary of the SOURCE DOCUMENT.\n"
        "1. Use the ROUGH SUMMARY as a starting point or inspiration.\n"
        "2. If the ROUGH SUMMARY contains hallucinations or makes zero sense, IGNORE IT entirely and write a completely new, accurate summary based ONLY on the SOURCE DOCUMENT.\n"
        "3. Ensure the final output is fluent, detailed, and directly captures the core message of the SOURCE document.\n\n"
        f"SOURCE DOCUMENT:\n{original_text}\n\n"
        f"ROUGH SUMMARY:\n{rough_summary}\n\n"
        "POLISHED SUMMARY:"
    )
    response = gemini_client.models.generate_content(model=gemini_model, contents=prompt)
    return response.text.strip()


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    cleaned_text = normalize_text(request.text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Please enter a document to summarize.")

    # ── Pure Gemini path ──────────────────────────────────────────────────────
    if request.engine == "gemini":
        if not gemini_client:
            raise HTTPException(status_code=503, detail="Gemini API key not configured.")
        try:
            prompt = (
                "You are an expert summarizer. Produce a concise, accurate, and well-written "
                "summary of the following document. Preserve key facts and conclusions. "
                "Do not add information that is not in the document.\n\n"
                f"DOCUMENT:\n{cleaned_text}\n\nSUMMARY:"
            )
            response = gemini_client.models.generate_content(
                model=request.gemini_model, contents=prompt
            )
            return SummarizeResponse(
                summary=response.text.strip(),
                engine_used=request.gemini_model,
                chunks_processed=1,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

    # ── BART path (with optional Gemini polish) ───────────────────────────────
    final_summary, num_chunks = _bart_summarize(cleaned_text, request)

    # Optional: use Gemini to clean up BART's output
    if request.polish and gemini_client:
        try:
            LOGGER.info("Applying Grounded Gemini polish to BART output...")
            final_summary = _gemini_polish(cleaned_text, final_summary, request.gemini_model)
            engine_label = f"bart-large-xsum + {request.gemini_model} polish"
        except Exception as e:
            LOGGER.error(f"Gemini polish failed: {e}. Falling back to raw BART output.")
            engine_label = f"bart-large-xsum (polish failed: {request.gemini_model})"
    else:
        engine_label = "bart-large-xsum" if num_chunks == 1 else f"bart-large-xsum (×{num_chunks} chunks)"

    return SummarizeResponse(
        summary=final_summary,
        engine_used=engine_label,
        chunks_processed=num_chunks,
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {
        "bart": "loaded",
        "gemini": "ready" if gemini_client else "no_key",
        "bart_max_tokens": "unlimited (hierarchical chunking)",
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
