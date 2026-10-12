"""
Tests for the Attention Interpretability & Positional Encoding Toolkit.

Tests verify the mathematical properties of each analysis function,
not just that they run -- the key distinction in this project.
"""
import math
import torch
import pytest

from app.interpretability import (
    attention_entropy, mean_attention_entropy, attention_distance,
    head_similarity, classify_head, extract_attention_patterns,
)
from app.probing import (
    probe_rope_relative_invariance, probe_alibi_slopes,
    probe_sinusoidal_frequencies,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_uniform_attention(B=1, H=2, T=6) -> torch.Tensor:
    """Uniform attention: every token attends equally to all past tokens."""
    attn = torch.zeros(B, H, T, T)
    for i in range(T):
        attn[:, :, i, :i+1] = 1.0 / (i + 1)
    return attn


def make_local_attention(B=1, H=2, T=6) -> torch.Tensor:
    """Local attention: every token attends only to the immediately previous token."""
    attn = torch.zeros(B, H, T, T)
    attn[:, :, 0, 0] = 1.0   # first token attends to itself
    for i in range(1, T):
        attn[:, :, i, i-1] = 1.0
    return attn


def make_diagonal_attention(B=1, H=2, T=6) -> torch.Tensor:
    """Diagonal attention: every token attends only to itself."""
    attn = torch.zeros(B, H, T, T)
    for i in range(T):
        attn[:, :, i, i] = 1.0
    return attn


# ── Entropy tests ──────────────────────────────────────────────────────────────

def test_entropy_uniform_is_maximum():
    """Uniform attention should produce maximum entropy = log(T)."""
    T = 8
    attn = torch.zeros(1, 1, T, T)
    for i in range(T):
        attn[:, :, i, :i+1] = 1.0 / (i + 1)

    # Entropy for the last token (attends uniformly to all T positions)
    # H = log(T) for a uniform distribution over T outcomes
    ent = attention_entropy(attn)  # (B, H, T)
    last_row_entropy = ent[0, 0, T-1].item()
    expected_max = math.log(T)
    assert abs(last_row_entropy - expected_max) < 0.01


def test_entropy_deterministic_is_zero():
    """Deterministic attention (one position gets all weight) has entropy 0."""
    T = 6
    attn = torch.zeros(1, 1, T, T)
    for i in range(T):
        attn[:, :, i, i] = 1.0  # attend only to self

    ent = attention_entropy(attn)
    # Each row is a degenerate distribution -> entropy = 0
    assert (ent < 0.01).all()


def test_entropy_ordering():
    """Uniform attention entropy > local attention entropy."""
    uniform = make_uniform_attention()
    local   = make_local_attention()

    ent_uniform = mean_attention_entropy(uniform)
    ent_local   = mean_attention_entropy(local)

    # Uniform has higher entropy on average
    assert ent_uniform.mean() > ent_local.mean()


# ── Attention distance tests ───────────────────────────────────────────────────

def test_distance_diagonal_is_zero():
    """Self-attention (diagonal) should have average distance = 0."""
    attn = make_diagonal_attention()
    dist = attention_distance(attn)  # (B, H, T)
    assert (dist.abs() < 1e-6).all()


def test_distance_local_is_one():
    """Local (previous token) attention should have average distance ≈ 1 (for T>1 positions)."""
    attn = make_local_attention()
    dist = attention_distance(attn)  # (B, H, T)
    # Position 0 attends to itself (dist=0), positions 1+ attend to i-1 (dist=1)
    # Average over positions 1..T-1 should be 1.0
    T = attn.size(-1)
    avg_dist_non_zero = dist[:, :, 1:].mean().item()
    assert abs(avg_dist_non_zero - 1.0) < 1e-5


def test_distance_uniform_is_increasing():
    """For uniform attention, later query positions attend over larger spans -> higher distance."""
    B, H, T = 1, 1, 10
    attn = make_uniform_attention(B=B, H=H, T=T)
    dist = attention_distance(attn)  # (B, H, T)
    # Later positions have larger average span, so distance should increase
    for i in range(1, T):
        assert dist[0, 0, i] >= dist[0, 0, i-1] - 1e-6


# ── Head similarity tests ──────────────────────────────────────────────────────

def test_similarity_diagonal_is_one():
    """Each head's similarity with itself should be 1.0."""
    attn = make_uniform_attention(H=4)
    sim = head_similarity(attn)  # (H, H)
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_similarity_identical_heads_is_one():
    """Two identical attention heads should have similarity 1.0."""
    B, T = 1, 6
    H = 2
    pattern = torch.zeros(B, 1, T, T)
    for i in range(T):
        pattern[:, :, i, :i+1] = 1.0 / (i+1)
    attn = pattern.expand(-1, H, -1, -1).clone()  # both heads identical

    sim = head_similarity(attn)
    assert torch.allclose(sim, torch.ones(H, H), atol=1e-4)


def test_similarity_different_heads_less_than_one():
    """Different attention patterns should have similarity < 1.0."""
    B, T = 1, 8
    attn = torch.zeros(B, 2, T, T)
    # Head 0: diagonal (self-attention)
    for i in range(T):
        attn[:, 0, i, i] = 1.0
    # Head 1: local (previous token)
    attn[:, 1, 0, 0] = 1.0
    for i in range(1, T):
        attn[:, 1, i, i-1] = 1.0

    sim = head_similarity(attn)
    assert sim[0, 1].item() < 0.99


# ── Head classification tests ──────────────────────────────────────────────────

def test_classify_diagonal_head():
    T = 8
    attn = torch.zeros(T, T)
    for i in range(T):
        attn[i, i] = 1.0
    assert classify_head(attn) == "diagonal"


def test_classify_local_head():
    T = 8
    attn = torch.zeros(T, T)
    attn[0, 0] = 1.0
    for i in range(1, T):
        attn[i, i-1] = 1.0
    assert classify_head(attn) == "local"


def test_classify_uniform_head():
    T = 8
    attn = torch.zeros(T, T)
    for i in range(T):
        attn[i, :i+1] = 1.0 / (i+1)
    result = classify_head(attn)
    assert result == "uniform"


def test_extract_attention_patterns_structure():
    attn = make_uniform_attention(B=2, H=4, T=8)
    result = extract_attention_patterns(attn)
    assert "entropy" in result
    assert "avg_distance" in result
    assert "similarity" in result
    assert "head_types" in result
    assert "summary" in result
    assert len(result["entropy"]) == 4
    assert len(result["head_types"]) == 4
    assert len(result["similarity"]) == 4
    assert len(result["similarity"][0]) == 4


# ── RoPE probing tests ─────────────────────────────────────────────────────────

def test_rope_relative_invariance_holds():
    result = probe_rope_relative_invariance(
        d_head=8, n_pairs=5, relative_distance=3, tolerance=1e-3
    )
    assert result["property_verified"], (
        f"RoPE relative invariance failed: {result['interpretation']}"
    )


def test_rope_relative_invariance_holds_multiple_distances():
    for dist in [1, 3, 7, 15]:
        result = probe_rope_relative_invariance(
            d_head=16, n_pairs=5, relative_distance=dist, tolerance=1e-3
        )
        assert result["property_verified"], (
            f"RoPE failed for relative_distance={dist}"
        )


def test_rope_different_distances_give_different_dots():
    """Different relative distances should produce different dot products."""
    from app.positional_encoding import build_rope_cache, apply_rope

    d_head = 16
    cos, sin = build_rope_cache(max_seq_len=30, d_head=d_head)
    torch.manual_seed(7)
    q = torch.randn(1, 1, 1, d_head)
    k = torch.randn(1, 1, 1, d_head)

    def dot_at_dist(d: int) -> float:
        q_rot, _ = apply_rope(q, q.clone(), cos[d:d+1], sin[d:d+1])
        _, k_rot = apply_rope(k.clone(), k, cos[:1], sin[:1])
        return (q_rot[0,0,0] @ k_rot[0,0,0]).item()

    dot_d3 = dot_at_dist(3)
    dot_d7 = dot_at_dist(7)
    assert abs(dot_d3 - dot_d7) > 1e-4, "Different distances gave the same dot product"


# ── ALiBi probing tests ────────────────────────────────────────────────────────

def test_alibi_slopes_match_theory():
    result = probe_alibi_slopes(n_heads=4, seq_len=16)
    assert result["all_slopes_match_theory"], result["slopes"]


def test_alibi_slopes_match_theory_powers_of_two():
    """The ALiBi formula is designed for n_heads that are powers of 2."""
    for n in [2, 4, 8]:
        result = probe_alibi_slopes(n_heads=n, seq_len=20)
        assert result["all_slopes_match_theory"], f"Failed for n_heads={n}: {result['slopes']}"


def test_alibi_slopes_geometric_sequence():
    """Slopes should form a geometric sequence (ratio = 2^(-8/n_heads) between adjacent heads)."""
    from app.positional_encoding import build_alibi_bias
    n_heads, seq_len = 8, 20
    bias = build_alibi_bias(n_heads, seq_len)

    slopes = [-bias[h, seq_len-1, 0].item() / (seq_len-1) for h in range(n_heads)]
    for i in range(1, len(slopes)):
        ratio = slopes[i] / slopes[i-1]
        expected_ratio = 2 ** (-8 / n_heads)
        assert abs(ratio - expected_ratio) < 1e-4, (
            f"Non-geometric sequence at head {i}: ratio={ratio:.6f} expected={expected_ratio:.6f}"
        )


# ── Sinusoidal frequency tests ─────────────────────────────────────────────────

def test_sinusoidal_frequency_structure():
    result = probe_sinusoidal_frequencies(d_model=16, n_positions=200)
    freqs = result["frequencies"]
    assert len(freqs) > 0
    # First dimension (dim=0) should have the HIGHEST theoretical frequency
    # (smallest period), last should have the lowest
    # (the formula: freq = 1/10000^(2i/d_model) decreases as i increases)
    first_freq  = freqs[0]["theoretical_freq"]
    second_freq = freqs[1]["theoretical_freq"]
    assert first_freq >= second_freq, "Frequencies should decrease with dimension index"


def test_sinusoidal_max_freq_is_one():
    """The first sinusoidal dimension (i=0) has frequency 1 (period=1)."""
    result = probe_sinusoidal_frequencies(d_model=32, n_positions=100)
    first_dim = result["frequencies"][0]
    assert abs(first_dim["theoretical_freq"] - 1.0) < 1e-5


def test_sinusoidal_frequency_decreases_geometrically():
    """Consecutive frequencies should decrease by a constant factor."""
    result = probe_sinusoidal_frequencies(d_model=16, n_positions=200)
    freqs = [f["theoretical_freq"] for f in result["frequencies"][:6]]
    ratios = [freqs[i+1] / freqs[i] for i in range(len(freqs)-1)]
    # Each ratio should be the same (geometric decrease)
    for r in ratios[1:]:
        assert abs(r - ratios[0]) < 1e-4, "Frequencies do not decrease geometrically"
