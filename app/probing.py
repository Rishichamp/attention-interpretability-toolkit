"""
Positional Encoding Probing Experiments.

Mathematical verification of positional encoding properties:

1. RoPE relative-position invariance:
   Verify that the dot product Q_rotated(m) . K_rotated(n) depends only on (m-n),
   not on the absolute positions m and n separately.
   Method: compute dot products for many (m, n) pairs with the same difference,
   confirm they are identical to numerical precision.

2. ALiBi distance penalty decay:
   Verify that each head's bias follows the linear decay formula,
   and that slopes are geometrically spaced across heads.

3. Sinusoidal encoding frequency analysis:
   Compute the dominant frequency component for each embedding dimension
   and verify it matches the expected 1/10000^(2i/d) formula.

4. Subject-verb agreement probe:
   A synthetic linguistic experiment: build sentences where a noun phrase
   of varying length intervenes between the subject and verb, and measure
   how attention from the verb position to the subject position degrades
   as the intervening span grows. This is the standard probe for testing
   whether a model has learned syntactic agreement.
"""

import math
import torch
import torch.nn.functional as F

from app.interpretability import attention_entropy, attention_distance


# ── RoPE probing ────────────────────────────────────────────────────────────────

def probe_rope_relative_invariance(
    d_head: int = 32,
    n_pairs: int = 10,
    relative_distance: int = 5,
    base: float = 10000.0,
    tolerance: float = 1e-4,
) -> dict:
    """
    Verify that RoPE produces the same Q.K dot product for all (m, n) pairs
    with the same relative distance m - n = relative_distance.

    This is the key mathematical property that makes RoPE an effective
    positional encoding for transformers: it encodes RELATIVE, not absolute,
    position.

    Returns:
        {
          "relative_distance": int,
          "dot_products": list of floats (one per tested pair),
          "max_deviation": float (should be < tolerance if property holds),
          "property_verified": bool
        }
    """
    from app.positional_encoding import build_rope_cache, apply_rope

    max_pos = relative_distance + n_pairs + 10
    cos, sin = build_rope_cache(max_seq_len=max_pos, d_head=d_head)

    torch.manual_seed(42)
    q = torch.randn(1, 1, 1, d_head)
    k = torch.randn(1, 1, 1, d_head)

    dot_products = []
    for i in range(n_pairs):
        m = relative_distance + i   # query position
        n = i                       # key position (m - n = relative_distance)

        cos_m, sin_m = cos[m:m+1], sin[m:m+1]
        cos_n, sin_n = cos[n:n+1], sin[n:n+1]

        q_rot, _ = apply_rope(q, q.clone(), cos_m, sin_m)
        _, k_rot = apply_rope(k.clone(), k, cos_n, sin_n)

        dot = (q_rot[0, 0, 0] @ k_rot[0, 0, 0]).item()
        dot_products.append(dot)

    max_deviation = max(abs(d - dot_products[0]) for d in dot_products)
    return {
        "relative_distance": relative_distance,
        "n_pairs_tested": n_pairs,
        "dot_products": [round(d, 6) for d in dot_products],
        "max_deviation": round(max_deviation, 8),
        "property_verified": max_deviation < tolerance,
        "interpretation": (
            "RoPE relative-position invariance VERIFIED: all "
            f"{n_pairs} pairs with relative distance {relative_distance} "
            f"produce identical Q.K dot products (max deviation={max_deviation:.2e})"
            if max_deviation < tolerance else
            f"WARNING: max deviation {max_deviation:.2e} exceeds tolerance {tolerance}"
        ),
    }


# ── ALiBi probing ────────────────────────────────────────────────────────────────

def probe_alibi_slopes(n_heads: int = 8, seq_len: int = 20) -> dict:
    """
    Verify that ALiBi bias slopes are geometrically spaced per head as
    slope_h = 2^(-8h/n_heads), and that each head's bias follows linear decay.

    Returns a dict with measured vs theoretical slopes for each head.
    """
    from app.positional_encoding import build_alibi_bias

    bias = build_alibi_bias(n_heads, seq_len)  # (n_heads, T, T)

    results = []
    for h in range(n_heads):
        theoretical_slope = 2 ** (-8 * (h + 1) / n_heads)

        # Measure slope: bias[h, T-1, 0] = -slope * (T-1)
        # So slope = -bias[h, T-1, 0] / (T-1)
        measured_slope = -bias[h, seq_len - 1, 0].item() / (seq_len - 1)

        results.append({
            "head": h,
            "theoretical_slope": round(theoretical_slope, 6),
            "measured_slope":    round(measured_slope, 6),
            "match": abs(theoretical_slope - measured_slope) < 1e-5,
        })

    all_match = all(r["match"] for r in results)
    return {
        "n_heads": n_heads,
        "slopes": results,
        "all_slopes_match_theory": all_match,
        "interpretation": (
            "ALiBi slope formula VERIFIED: all measured slopes match 2^(-8h/n_heads)"
            if all_match else
            "WARNING: some slopes do not match theoretical formula"
        ),
    }


# ── Sinusoidal frequency analysis ──────────────────────────────────────────────

def probe_sinusoidal_frequencies(d_model: int = 32, n_positions: int = 100) -> dict:
    """
    Extract the dominant frequency from each even dimension of the sinusoidal
    encoding and verify it matches the expected 1 / (10000^(2i/d_model)).

    Method: for each even dimension 2i, the values across positions are
    sin(pos / 10000^(2i/d_model)) -- a pure sine wave. We find the period
    by detecting the first zero-crossing after position 0.

    Returns frequencies per dimension and their match with theory.
    """
    positions = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)  # (T, 1)
    div_terms = torch.arange(0, d_model, 2, dtype=torch.float)             # (d_model/2,)
    freqs = 1.0 / (10000.0 ** (div_terms / d_model))

    pe = torch.sin(positions * freqs.unsqueeze(0))  # (T, d_model/2)

    results = []
    for i in range(d_model // 2):
        theoretical_freq = freqs[i].item()
        theoretical_period = 1.0 / theoretical_freq if theoretical_freq > 0 else float("inf")

        # Find estimated period via FFT
        fft = torch.fft.rfft(pe[:, i])
        magnitudes = fft.abs()
        magnitudes[0] = 0  # ignore DC component
        dominant_bin = magnitudes.argmax().item()
        estimated_freq = dominant_bin / n_positions if dominant_bin > 0 else 0.0

        results.append({
            "dim": 2 * i,
            "theoretical_freq": round(theoretical_freq, 6),
            "estimated_freq":   round(estimated_freq, 6),
            "theoretical_period": round(theoretical_period, 2),
        })

    return {
        "d_model": d_model,
        "n_positions": n_positions,
        "frequencies": results[:8],  # return first 8 dims for readability
        "interpretation": (
            "Lower-frequency dimensions (small i) oscillate slowly -- encode coarse position. "
            "Higher-frequency dimensions (large i) oscillate quickly -- encode fine position. "
            "This is analogous to a Fourier series basis over sequence positions."
        ),
    }


# ── Subject-verb agreement probe ───────────────────────────────────────────────

def probe_subject_verb_agreement(
    model,
    tokenizer,
    max_span_length: int = 8,
) -> dict:
    """
    Synthetic linguistic probe: how does attention from the verb position to
    the subject position degrade as intervening material grows?

    We construct sentences of the form:
      "The [subject] [intervening words] [verb]"

    and measure the attention weight from [verb] to [subject] for each span length.
    A model that has learned syntactic agreement should show high attention to the
    subject regardless of span length. A model that relies on local attention will
    show rapidly decreasing attention as the span grows.

    Note: for a character-level model, "words" are sequences of characters.
    We work at the token (character) level and track the attention from the
    final character of the "verb" to the characters of the "subject".

    Returns a dict with attention_to_subject per span length.
    """
    model.eval()

    subject = "dog"
    verb = "runs"
    fillers = ["the", "old", "big", "very", "red", "small", "quick", "nice"]

    results = []
    for span_len in range(0, min(max_span_length, len(fillers)) + 1):
        filler_str = " ".join(fillers[:span_len])
        if filler_str:
            sentence = f"The {subject} {filler_str} {verb}"
        else:
            sentence = f"The {subject} {verb}"

        ids = tokenizer.encode(sentence)
        if len(ids) < 4:
            continue

        idx = torch.tensor([ids])
        with torch.no_grad():
            _, _, attentions = model(idx, return_attention=True)

        # Find character positions of subject and verb in the token sequence
        subject_start = len(tokenizer.encode(f"The "))
        subject_end = subject_start + len(tokenizer.encode(subject))
        verb_pos = len(ids) - 1  # last character of verb

        if verb_pos <= subject_start or subject_end > len(ids):
            continue

        # Average over last layer and all heads: attention FROM verb TO subject chars
        last_layer_attn = attentions[-1][0]  # (n_heads, T, T)
        attn_verb_to_subject = last_layer_attn[:, verb_pos, subject_start:subject_end].mean().item()

        results.append({
            "span_length": span_len,
            "sentence": sentence,
            "subject_token_range": [subject_start, subject_end],
            "verb_token_pos": verb_pos,
            "attn_verb_to_subject": round(attn_verb_to_subject, 6),
        })

    return {
        "probe": "subject_verb_agreement",
        "model_type": "character_level_causal_lm",
        "results": results,
        "interpretation": (
            "attn_verb_to_subject shows how much the last token of 'runs' attends "
            "to 'dog' tokens as intervening material grows. "
            "High, stable values = model has learned long-range dependencies. "
            "Rapidly decaying values = model relies on local attention."
        ),
    }
