"""
Attention Interpretability Toolkit.

Provides quantitative analysis of attention weight matrices extracted from
any model that returns (B, n_heads, T, T) attention tensors -- including
the GPTLanguageModel from Project 1's /attention endpoint.

Five analysis tools:

  1. attention_entropy()         -- how "focused" vs "diffuse" each head is
  2. head_specialisation()       -- identify local/diagonal/uniform head types
  3. attention_distance()        -- average distance between attending positions
  4. head_similarity()           -- cosine similarity between heads (detect redundancy)
  5. extract_attention_patterns() -- classify each head: local / diagonal / global

These are the same analysis techniques used in the BertViz section of the
Module 5 notes, but implemented from scratch so the maths is explicit.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of each attention distribution.

    H(head) = -sum_k(w_k * log(w_k + eps))

    High entropy = diffuse, attends broadly (low attention, global context).
    Low entropy  = focused, attends to few positions (high attention, local/specific).

    Args:
        attn: (B, n_heads, T, T) attention weights (post-softmax, rows sum to 1)

    Returns:
        entropy: (B, n_heads, T) entropy per query position per head
    """
    eps = 1e-9
    log_attn = torch.log(attn + eps)
    return -(attn * log_attn).sum(dim=-1)


def mean_attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Average entropy across query positions and batch -- one value per head.

    Returns: (n_heads,) mean entropy for each head
    """
    # attn: (B, n_heads, T, T) -> entropy: (B, n_heads, T) -> mean over B and T
    return attention_entropy(attn).mean(dim=(0, 2))


def attention_distance(attn: torch.Tensor) -> torch.Tensor:
    """
    Average absolute distance between query and key positions, weighted by attention.

    For each query position i, the expected distance is:
      E[|i - j|]_w = sum_j w_{i,j} * |i - j|

    Heads with LOW average distance are LOCAL (attend to nearby tokens).
    Heads with HIGH average distance are GLOBAL (attend across the sequence).

    Args:
        attn: (B, n_heads, T, T)

    Returns:
        avg_distance: (B, n_heads, T) -- expected distance per query position per head
    """
    T = attn.size(-1)
    positions = torch.arange(T, dtype=attn.dtype, device=attn.device)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (T, T)
    # Weighted average: sum_k(w_{i,k} * |i - k|) for each query i
    return (attn * distances.unsqueeze(0).unsqueeze(0)).sum(dim=-1)


def head_similarity(attn: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine similarity between head attention patterns.

    Used to detect REDUNDANT heads -- heads with very high mutual similarity
    are likely doing the same thing and could be pruned.

    Args:
        attn: (B, n_heads, T, T)

    Returns:
        sim_matrix: (n_heads, n_heads) symmetric similarity matrix,
                    averaged over batch and query positions.
    """
    B, H, T, _ = attn.shape
    # Flatten each head's full attention matrix: (B, H, T*T)
    attn_flat = attn.view(B, H, T * T)
    # Average over batch
    attn_mean = attn_flat.mean(dim=0)  # (H, T*T)
    # Normalise each head vector
    norms = attn_mean.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    attn_norm = attn_mean / norms
    return torch.matmul(attn_norm, attn_norm.T)  # (H, H)


def classify_head(attn_head: torch.Tensor, local_threshold: float = 0.5) -> str:
    """
    Classify a single head's attention pattern.

    Args:
        attn_head: (T, T) attention matrix for one head
        local_threshold: fraction of attention within distance 1 to call "local"

    Returns one of: "local" | "diagonal" | "uniform" | "global"

    - local:    predominantly attends to the immediately preceding token
    - diagonal: attends mostly to itself (self-attention)
    - uniform:  attention spread evenly (high entropy, near 1/T per position)
    - global:   complex non-local pattern (none of the above)
    """
    T = attn_head.size(0)
    if T < 3:
        return "global"

    # How much is on the diagonal (attending to self)?
    diag_mass = attn_head.diagonal().mean().item()

    # How much is on the super-diagonal (attending to immediately previous token)?
    prev_mass = 0.0
    if T > 1:
        for i in range(1, T):
            prev_mass += attn_head[i, i - 1].item()
        prev_mass /= (T - 1)

    # How uniform is the distribution? Compare to 1/T
    max_entropy = math.log(T) if T > 1 else 1.0
    mean_ent = -(attn_head * (attn_head + 1e-9).log()).sum(dim=-1).mean().item()
    relative_entropy = mean_ent / max_entropy

    if prev_mass > local_threshold:
        return "local"
    elif diag_mass > local_threshold:
        return "diagonal"
    elif relative_entropy > 0.60:  # near-maximum entropy = uniform (causal LM: earlier rows have fewer keys)
        return "uniform"
    else:
        return "global"


def extract_attention_patterns(attn: torch.Tensor) -> dict:
    """
    Full interpretability summary for a batch of attention tensors.

    Args:
        attn: (B, n_heads, T, T) from a single transformer layer

    Returns a dict containing:
      - entropy:        (n_heads,) mean entropy per head
      - avg_distance:   (n_heads,) mean attention distance per head
      - similarity:     (n_heads, n_heads) head similarity matrix
      - head_types:     list of str -- head classification per head
      - summary:        list of human-readable summary strings
    """
    B, H, T, _ = attn.shape

    entropies = mean_attention_entropy(attn)          # (H,)
    distances = attention_distance(attn).mean(dim=(0, 2))  # (H,) via mean over B and T
    similarity = head_similarity(attn)                # (H, H)

    # Classify each head using the first example's averaged attention
    attn_avg = attn.mean(dim=0)  # (H, T, T) averaged over batch
    head_types = [classify_head(attn_avg[h]) for h in range(H)]

    summary = []
    for h in range(H):
        summary.append(
            f"Head {h}: type={head_types[h]:10s} | "
            f"entropy={entropies[h].item():.3f} | "
            f"avg_distance={distances[h].item():.2f}"
        )

    return {
        "entropy":      entropies.tolist(),
        "avg_distance": distances.tolist(),
        "similarity":   similarity.tolist(),
        "head_types":   head_types,
        "summary":      summary,
    }


def analyse_all_layers(
    attentions: list[torch.Tensor],
    verbose: bool = True,
) -> list[dict]:
    """
    Run extract_attention_patterns on every layer's attention tensors.

    Args:
        attentions: list of (B, n_heads, T, T), one per layer

    Returns:
        layer_results: list of dicts, one per layer
    """
    results = []
    for layer_idx, attn in enumerate(attentions):
        result = extract_attention_patterns(attn)
        result["layer"] = layer_idx
        results.append(result)
        if verbose:
            print(f"\n[Layer {layer_idx}]")
            for line in result["summary"]:
                print(f"  {line}")
    return results
