"""
Metric computation for A/B test evaluation.

Covers aggregate CTR, per-query CTR lift, Cohen's d effect size,
and a position-weighted proxy NDCG for paired testing.
"""

import numpy as np
from typing import Dict


def aggregate_ctr(clicks: np.ndarray) -> float:
    """
    Compute aggregate Click-Through Rate across all impressions.

    Defined as total clicks / total impressions (positions).

    Args:
        clicks: Array of shape (N, K) with binary click indicators.

    Returns:
        Aggregate CTR in [0, 1].

    Raises:
        ValueError: If clicks array is empty.
    """
    if clicks.size == 0:
        raise ValueError("clicks array must not be empty")
    return float(clicks.sum() / clicks.size)


def ctr_lift(ctr_control: float, ctr_treatment: float) -> Dict[str, float]:
    """
    Compute absolute and relative CTR lift from control to treatment.

    Args:
        ctr_control: Aggregate CTR for control group (A).
        ctr_treatment: Aggregate CTR for treatment group (B).

    Returns:
        Dictionary with:
            - "absolute_lift": ctr_B - ctr_A
            - "relative_lift_pct": (ctr_B - ctr_A) / ctr_A * 100

    Raises:
        ValueError: If ctr_control is zero (undefined relative lift).
    """
    if ctr_control <= 0:
        raise ValueError(
            f"Control CTR must be > 0 to compute relative lift; got {ctr_control}"
        )
    absolute = ctr_treatment - ctr_control
    relative_pct = (absolute / ctr_control) * 100.0
    return {"absolute_lift": absolute, "relative_lift_pct": relative_pct}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent samples.

    Cohen's d = (mean_B - mean_A) / pooled_std

    Uses pooled standard deviation (equal-variance assumption):
        s_pooled = sqrt(((n_A - 1)*s_A^2 + (n_B - 1)*s_B^2) / (n_A + n_B - 2))

    Interpretation: |d| < 0.2 small, 0.2–0.5 medium, > 0.8 large.

    Args:
        a: Per-query CTR array for control group.
        b: Per-query CTR array for treatment group.

    Returns:
        Cohen's d (signed, positive means treatment > control).

    Raises:
        ValueError: If either array is empty or pooled std is zero.
    """
    if a.size == 0 or b.size == 0:
        raise ValueError("Both arrays must be non-empty")

    n_a, n_b = len(a), len(b)
    mean_diff = b.mean() - a.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        raise ValueError("Pooled standard deviation is zero; cannot compute Cohen's d")

    return float(mean_diff / pooled_std)


def proxy_ndcg(clicks: np.ndarray) -> np.ndarray:
    """
    Compute a position-weighted proxy metric (proxy NDCG) per query.

    For each query, we compute the DCG-style score using actual clicks
    as binary gains and the ideal DCG assuming all top positions are clicked:

        proxy_NDCG_i = DCG_i / IDCG

    where:
        DCG_i  = sum_k [ click[i,k] / log2(k + 2) ]
        IDCG   = sum_k=1^K [ 1 / log2(k + 2) ]   (all positions clicked)

    This is a position-aware engagement signal suitable for paired t-test
    because query-level scores are naturally paired across A and B.

    Args:
        clicks: Array of shape (N, K) with binary click indicators.

    Returns:
        Array of shape (N,) with per-query proxy NDCG scores in [0, 1].

    Raises:
        ValueError: If clicks array is empty.
    """
    if clicks.size == 0:
        raise ValueError("clicks array must not be empty")

    N, K = clicks.shape
    # Position discounts: 1/log2(k+2) for k in [1..K]
    discounts = np.array([1.0 / np.log2(k + 2) for k in range(1, K + 1)])

    dcg = clicks @ discounts          # shape (N,)
    idcg = discounts.sum()            # scalar: best possible score

    return dcg / idcg


def summarize_metrics(
    clicks_a: np.ndarray,
    clicks_b: np.ndarray,
    ctr_per_query_a: np.ndarray,
    ctr_per_query_b: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the full set of descriptive metrics for an A/B comparison.

    Args:
        clicks_a: Shape (N, K) binary clicks for control.
        clicks_b: Shape (N, K) binary clicks for treatment.
        ctr_per_query_a: Shape (N,) per-query CTR for control.
        ctr_per_query_b: Shape (N,) per-query CTR for treatment.

    Returns:
        Dictionary of scalar metrics keyed by descriptive name.
    """
    ctr_a = aggregate_ctr(clicks_a)
    ctr_b = aggregate_ctr(clicks_b)
    lift = ctr_lift(ctr_a, ctr_b)
    d = cohens_d(ctr_per_query_a, ctr_per_query_b)

    return {
        "ctr_control": ctr_a,
        "ctr_treatment": ctr_b,
        "absolute_lift": lift["absolute_lift"],
        "relative_lift_pct": lift["relative_lift_pct"],
        "cohens_d": d,
    }
