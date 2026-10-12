"""
FastAPI server for the Attention Interpretability & Positional Encoding Toolkit.

Endpoints:
  POST /analyse/attention      -- attention entropy, distance, head types for input text
  POST /probe/rope             -- verify RoPE relative-position invariance
  GET  /probe/alibi            -- verify ALiBi slope formula
  GET  /probe/sinusoidal       -- sinusoidal frequency analysis
  POST /probe/subject-verb     -- subject-verb agreement experiment
  GET  /health                 -- health check

The toolkit is designed to be used with a GPTLanguageModel from Project 1
(loaded from its checkpoint) OR with the toy model initialised at startup.
"""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.interpretability import extract_attention_patterns, analyse_all_layers
from app.probing import (
    probe_rope_relative_invariance,
    probe_alibi_slopes,
    probe_sinusoidal_frequencies,
    probe_subject_verb_agreement,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_state = {"model": None, "tokenizer": None, "source": None}

# Folder holding model.pt + tokenizer.json from the Transformer Lab (Project 1).
# The model code itself is included in this package (app/model.py etc.), so only
# the trained weights are needed. Without them, an untrained toy model is used.
M5P1_CHECKPOINT_DIR = os.getenv("M5P1_CHECKPOINT_DIR", "checkpoints")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Try to load Project 1's trained model; fall back to a toy model if not found."""
    ckpt_path = Path(M5P1_CHECKPOINT_DIR) / "model.pt"
    tok_path  = Path(M5P1_CHECKPOINT_DIR) / "tokenizer.json"

    try:
        if ckpt_path.exists() and tok_path.exists():
            from app.tokenizer import CharTokenizer
            from app.model import GPTLanguageModel
            from app.config import ModelConfig

            tokenizer = CharTokenizer.load(str(tok_path))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = ModelConfig(**ckpt["config"])
            model = GPTLanguageModel(config)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _state["model"] = model
            _state["tokenizer"] = tokenizer
            _state["source"] = "m5p1_trained_checkpoint"
            logger.info("Loaded M5P1 trained checkpoint")
        else:
            raise FileNotFoundError("M5P1 checkpoint not found")
    except Exception as e:
        logger.warning(f"Could not load M5P1 model: {e}. Using toy model.")
        _load_toy_model()

    yield
    _state["model"] = None
    _state["tokenizer"] = None


def _load_toy_model():
    """Load a tiny toy model for demonstration when no trained model is available."""
    from app.tokenizer import CharTokenizer
    from app.model import GPTLanguageModel
    from app.config import ModelConfig

    vocab = sorted("abcdefghijklmnopqrstuvwxyz .,")
    tokenizer = CharTokenizer(vocab)
    config = ModelConfig(
        vocab_size=len(vocab), d_model=32, n_heads=4, n_layers=3,
        max_seq_len=64, dropout=0.0, pos_encoding="rope", activation="gelu",
    )
    model = GPTLanguageModel(config)
    model.eval()
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["source"] = "toy_untrained_model"
    logger.info("Using toy untrained model (for architecture/probing demos)")


app = FastAPI(
    title="Attention Interpretability & Positional Encoding Toolkit",
    description=(
        "Analysis tools for transformer attention patterns: entropy, head types, "
        "distance analysis, and mathematical verification of RoPE/ALiBi/sinusoidal "
        "positional encoding properties."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────
class AttentionAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyse (min 4 chars from vocabulary)")
    layer: Optional[int] = Field(None, description="Specific layer to analyse (None = all layers)")


class RoPEProbeRequest(BaseModel):
    d_head: int = Field(32, ge=2)
    n_pairs: int = Field(10, ge=2, le=50)
    relative_distance: int = Field(5, ge=1)
    tolerance: float = Field(1e-4, gt=0)


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return f"""<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>🔍 Attention Interpretability Toolkit</h1>
    <p>Model source: <b>{_state.get('source', 'none')}</b></p>
    <p><a href="/docs">📖 API Docs</a></p>
    <h3>Available probes</h3>
    <ul>
      <li>POST /analyse/attention -- entropy, head types, distance for any text</li>
      <li>POST /probe/rope -- verify RoPE relative-position invariance</li>
      <li>GET  /probe/alibi -- verify ALiBi slope formula</li>
      <li>GET  /probe/sinusoidal -- sinusoidal frequency analysis</li>
      <li>POST /probe/subject-verb -- syntactic agreement experiment</li>
    </ul>
    </body></html>"""


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["model"] else "no_model",
        "model_source": _state.get("source"),
    }


@app.post("/analyse/attention")
async def analyse_attention(req: AttentionAnalysisRequest):
    """
    Analyse attention patterns for a given text.

    Returns per-head entropy, average attention distance, and head-type
    classification (local / diagonal / global / uniform) for each layer.
    """
    model = _state["model"]
    tokenizer = _state["tokenizer"]

    ids = tokenizer.encode(req.text)[:model.config.max_seq_len]
    if len(ids) < 2:
        raise HTTPException(400, "Text too short or contains no known characters. Need at least 2.")

    idx = torch.tensor([ids])
    with torch.no_grad():
        _, _, attentions = model(idx, return_attention=True)

    if req.layer is not None:
        if req.layer < 0 or req.layer >= len(attentions):
            raise HTTPException(400, f"Layer {req.layer} out of range (model has {len(attentions)} layers)")
        attentions = [attentions[req.layer]]
        start_layer = req.layer
    else:
        start_layer = 0

    results = []
    for i, attn in enumerate(attentions):
        layer_result = extract_attention_patterns(attn)
        layer_result["layer"] = start_layer + i
        results.append(layer_result)

    return {
        "text": req.text,
        "n_tokens": len(ids),
        "tokens": [tokenizer.itos[i] for i in ids],
        "n_layers_analysed": len(results),
        "layer_results": results,
    }


@app.post("/probe/rope")
async def probe_rope(req: RoPEProbeRequest):
    """
    Mathematically verify RoPE's relative-position invariance property.

    Tests that dot(rotate(q, m), rotate(k, n)) is identical for all (m, n)
    pairs with the same relative distance m - n.
    """
    result = probe_rope_relative_invariance(
        d_head=req.d_head,
        n_pairs=req.n_pairs,
        relative_distance=req.relative_distance,
        tolerance=req.tolerance,
    )
    return result


@app.get("/probe/alibi")
async def probe_alibi(n_heads: int = 8, seq_len: int = 20):
    """
    Verify that ALiBi bias slopes follow the geometric formula
    slope_h = 2^(-8h/n_heads) and that biases decay linearly with distance.
    """
    return probe_alibi_slopes(n_heads=n_heads, seq_len=seq_len)


@app.get("/probe/sinusoidal")
async def probe_sinusoidal(d_model: int = 32, n_positions: int = 100):
    """
    Analyse the frequency structure of sinusoidal positional encodings.
    Verifies that each embedding dimension oscillates at its expected frequency.
    """
    return probe_sinusoidal_frequencies(d_model=d_model, n_positions=n_positions)


@app.post("/probe/subject-verb")
async def probe_subject_verb(max_span_length: int = 6):
    """
    Synthetic subject-verb agreement experiment.

    Measures how attention from the verb position to the subject position
    changes as intervening material grows. Tests long-range dependency capacity.
    """
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    return probe_subject_verb_agreement(model, tokenizer, max_span_length)
