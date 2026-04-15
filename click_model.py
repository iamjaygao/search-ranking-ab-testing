"""
Position-Based Click Model (PBM) for simulating user clicks on search results.

The PBM assumes click probability factorizes into:
    P(click | rank k) = P(examine | rank k) * P(relevant | document)

where examination probability decays with rank (position bias), and
relevance is a document-level property independent of position.
"""

import numpy as np
from typing import Tuple


def position_bias(k: int) -> float:
    """
    Compute examination probability for rank position k (1-indexed).

    Uses the standard logarithmic decay:
        v(k) = 1 / log2(k + 2)

    At rank 1: v(1) = 1/log2(3) ≈ 0.631
    At rank 10: v(10) = 1/log2(12) ≈ 0.278

    Args:
        k: Rank position (1-indexed).

    Returns:
        Examination probability in (0, 1].
    """
    if k < 1:
        raise ValueError(f"Rank position must be >= 1, got {k}")
    return 1.0 / np.log2(k + 2)


def click_probabilities(relevance_scores: np.ndarray) -> np.ndarray:
    """
    Compute per-position click probabilities for a single ranked list.

    P(click | rank k) = relevance_score[k] * position_bias(k)

    Args:
        relevance_scores: Array of shape (K,) with relevance scores in [0, 1],
                          ordered by rank position (index 0 = rank 1).

    Returns:
        Array of shape (K,) with click probabilities in [0, 1].
    """
    if relevance_scores.ndim != 1:
        raise ValueError("relevance_scores must be a 1-D array")
    if len(relevance_scores) == 0:
        raise ValueError("relevance_scores must not be empty")

    K = len(relevance_scores)
    biases = np.array([position_bias(k) for k in range(1, K + 1)])
    return relevance_scores * biases


def simulate_clicks(
    relevance_scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate binary click outcomes for a single ranked list (one impression).

    Each position is clicked independently with probability:
        P(click | rank k) = relevance_score[k] * position_bias(k)

    Args:
        relevance_scores: Array of shape (K,) with relevance scores in [0, 1].
        rng: NumPy random Generator for reproducibility.

    Returns:
        Array of shape (K,) with binary click indicators (0 or 1).
    """
    probs = click_probabilities(relevance_scores)
    return rng.binomial(n=1, p=probs).astype(np.int32)


def simulate_session(
    all_relevance_scores: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate clicks across N queries, each with K ranked results.

    Args:
        all_relevance_scores: Array of shape (N, K) with relevance scores in [0, 1].
                              Rows are queries, columns are rank positions.
        rng: NumPy random Generator for reproducibility.

    Returns:
        clicks: Array of shape (N, K) with binary click indicators.
        ctr_per_query: Array of shape (N,) with per-query CTR (clicks / K).
    """
    if all_relevance_scores.ndim != 2:
        raise ValueError("all_relevance_scores must be a 2-D array of shape (N, K)")

    N, K = all_relevance_scores.shape
    clicks = np.zeros((N, K), dtype=np.int32)

    for i in range(N):
        clicks[i] = simulate_clicks(all_relevance_scores[i], rng)

    ctr_per_query = clicks.mean(axis=1)  # fraction of positions clicked per query
    return clicks, ctr_per_query
