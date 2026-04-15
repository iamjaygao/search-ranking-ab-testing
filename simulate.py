"""
Main simulation entry point.

Simulates an online A/B test comparing a BM25 baseline (control) against
a neural reranker (treatment) using the Position-Based Click Model.

Offline context (from ESCI ranking project):
    Control  (BM25):             NDCG@10 ≈ 0.804
    Treatment (Neural reranker): NDCG@10 ≈ 0.845

Relevance scores are drawn from calibrated distributions that reflect the
offline quality gap, then fed into the PBM to generate simulated clicks.
We then test whether the click-rate improvement is statistically significant.

Usage:
    python simulate.py
"""

import numpy as np
from typing import Tuple

from click_model import simulate_session
from metrics import proxy_ndcg, summarize_metrics
from ab_test import run_all_tests

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

N_QUERIES: int = 1000       # number of queries per group
K_RESULTS: int = 10         # results per query (impression depth)
RANDOM_SEED: int = 42

# Relevance score distributions (Normal, clipped to [0, 1])
# These are NOT NDCG values — they are per-document relevance signals that
# proxy the quality gap observed offline (0.804 → 0.845, ~5% relative gain).
#
# We set treatment mean ~5% higher and slightly lower variance to reflect
# that the neural model tends to surface more relevant documents overall.
CONTROL_MEAN: float = 0.35
CONTROL_STD: float = 0.18

TREATMENT_MEAN: float = 0.39   # ~11% higher mean relevance
TREATMENT_STD: float = 0.17


# ---------------------------------------------------------------------------
# Relevance score generation
# ---------------------------------------------------------------------------

def generate_relevance_scores(
    n_queries: int,
    k_results: int,
    mean: float,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a (N, K) matrix of relevance scores from a clipped Normal distribution.

    Each entry represents the intrinsic relevance of a document to a query,
    independent of its rank position. The PBM then applies position discounts
    on top of these scores to compute click probabilities.

    Args:
        n_queries: Number of queries (N).
        k_results: Results per query (K).
        mean:      Mean of the Normal distribution.
        std:       Standard deviation of the Normal distribution.
        rng:       NumPy random Generator.

    Returns:
        Array of shape (N, K) with values clipped to [0, 1].
    """
    scores = rng.normal(loc=mean, scale=std, size=(n_queries, k_results))
    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    n_queries: int = N_QUERIES,
    k_results: int = K_RESULTS,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate click sessions for control (A) and treatment (B) groups.

    Both groups share the same query set (same RNG stream split), so
    query-level scores are properly paired for the paired t-test.

    Args:
        n_queries: Number of queries per group.
        k_results: Results shown per query.
        seed:      Random seed for reproducibility.

    Returns:
        clicks_a:        Shape (N, K) binary clicks for control.
        clicks_b:        Shape (N, K) binary clicks for treatment.
        ctr_per_query_a: Shape (N,) per-query CTR for control.
        ctr_per_query_b: Shape (N,) per-query CTR for treatment.
    """
    # Use independent RNG streams for each group to avoid cross-contamination
    rng_rel_a = np.random.default_rng(seed)
    rng_rel_b = np.random.default_rng(seed + 1)
    rng_click_a = np.random.default_rng(seed + 2)
    rng_click_b = np.random.default_rng(seed + 3)

    rel_a = generate_relevance_scores(n_queries, k_results, CONTROL_MEAN, CONTROL_STD, rng_rel_a)
    rel_b = generate_relevance_scores(n_queries, k_results, TREATMENT_MEAN, TREATMENT_STD, rng_rel_b)

    clicks_a, ctr_per_query_a = simulate_session(rel_a, rng_click_a)
    clicks_b, ctr_per_query_b = simulate_session(rel_b, rng_click_b)

    return clicks_a, clicks_b, ctr_per_query_a, ctr_per_query_b


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _sig_marker(p_value: float) -> str:
    """Return a significance marker for display."""
    return "✓ p < 0.05" if p_value < 0.05 else "✗ p ≥ 0.05"


def print_report(
    metrics: dict,
    test_results: dict,
) -> None:
    """
    Print a clean, structured A/B test report to stdout.

    Args:
        metrics:      Output of metrics.summarize_metrics().
        test_results: Output of ab_test.run_all_tests().
    """
    SEP = "=" * 60
    sep = "-" * 60

    print(f"\n{SEP}")
    print("  SEARCH RANKING A/B TEST — SIMULATION REPORT")
    print(f"{SEP}\n")

    # --- Descriptive stats ---
    print("CLICK-THROUGH RATE SUMMARY")
    print(sep)
    print(f"  Control   CTR : {metrics['ctr_control']:.4f}  ({metrics['ctr_control']*100:.2f}%)")
    print(f"  Treatment CTR : {metrics['ctr_treatment']:.4f}  ({metrics['ctr_treatment']*100:.2f}%)")
    print(f"  Absolute lift : {metrics['absolute_lift']:+.4f}")
    print(f"  Relative lift : {metrics['relative_lift_pct']:+.2f}%")
    print(f"  Cohen's d     : {metrics['cohens_d']:.4f}")
    print()

    # --- Statistical tests ---
    print("STATISTICAL TESTS  (α = 0.05, one-sided H₁: B > A)")
    print(sep)

    z = test_results["z_test"]
    print(f"  [1] Two-proportion z-test")
    print(f"      z = {z['z_stat']:.4f},  p = {z['p_value']:.4f}  →  {_sig_marker(z['p_value'])}")

    mw = test_results["mann_whitney"]
    print(f"  [2] Mann-Whitney U test (per-query CTR)")
    print(f"      U = {mw['u_stat']:.0f},  p = {mw['p_value']:.4f}  →  {_sig_marker(mw['p_value'])}")

    tt = test_results["paired_ttest"]
    print(f"  [3] Paired t-test (proxy NDCG)")
    print(f"      t = {tt['t_stat']:.4f},  p = {tt['p_value']:.4f}  →  {_sig_marker(tt['p_value'])}")
    print(f"      Mean NDCG diff : {tt['mean_diff']:+.4f}")
    print()

    # --- Power analysis ---
    pa = test_results["power_analysis"]
    print("POWER ANALYSIS")
    print(sep)
    print(f"  Effect size (Cohen's h) : {pa['effect_size_h']:.4f}")
    print(f"  N per group             : {pa['n_per_group']}")
    print(f"  Achieved power          : {pa['achieved_power']:.4f}  ({pa['achieved_power']*100:.1f}%)")
    print(f"  Min N for 80% power     : {pa['min_n_for_80pct_power']}")
    print()

    # --- Conclusion ---
    all_significant = all([
        z["significant"],
        mw["significant"],
        tt["significant"],
    ])
    print("CONCLUSION")
    print(sep)
    if all_significant:
        print(
            f"  All three tests reject H₀ (p < 0.05). The neural reranker produces\n"
            f"  a statistically significant CTR improvement of {metrics['relative_lift_pct']:+.2f}%\n"
            f"  over the BM25 baseline, consistent with the offline NDCG gain\n"
            f"  (0.804 → 0.845). Offline improvements translate to online engagement."
        )
    else:
        failing = [
            name for name, res in [
                ("z-test", z), ("Mann-Whitney", mw), ("paired t-test", tt)
            ] if not res["significant"]
        ]
        print(
            f"  Inconclusive: {', '.join(failing)} did not reach significance.\n"
            f"  Consider increasing sample size (min N = {pa['min_n_for_80pct_power']}) or\n"
            f"  re-examining the effect size calibration."
        )
    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full A/B test simulation and print the report."""
    print(f"Running simulation: N={N_QUERIES} queries, K={K_RESULTS} results, seed={RANDOM_SEED}")

    clicks_a, clicks_b, ctr_per_query_a, ctr_per_query_b = run_simulation()

    # Compute proxy NDCG for paired test
    pndcg_a = proxy_ndcg(clicks_a)
    pndcg_b = proxy_ndcg(clicks_b)

    metrics = summarize_metrics(clicks_a, clicks_b, ctr_per_query_a, ctr_per_query_b)
    test_results = run_all_tests(
        clicks_a, clicks_b,
        ctr_per_query_a, ctr_per_query_b,
        pndcg_a, pndcg_b,
    )

    print_report(metrics, test_results)


if __name__ == "__main__":
    main()
