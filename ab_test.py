"""
Statistical tests and power analysis for the A/B experiment.

Three complementary tests are used to evaluate the experiment from different angles:

1. Two-proportion z-test   — aggregate CTR, simple, interpretable
2. Mann-Whitney U test     — per-query CTR, non-parametric, robust to non-normality
3. Paired t-test           — per-query proxy NDCG, accounts for query-level pairing

Running all three guards against assumptions of any single test and increases
confidence when conclusions agree across methods.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Test 1: Two-proportion z-test
# ---------------------------------------------------------------------------

def two_proportion_ztest(
    clicks_a: np.ndarray,
    clicks_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Test whether aggregate CTRs differ between control and treatment.

    Why: The simplest and most interpretable test for binary click events.
    Assumption: Clicks are i.i.d. Bernoulli trials. With N=1000 queries × K=10
    positions this is a reasonable approximation (CLT applies).

    H0: p_A == p_B  (no difference in click rates)
    H1: p_B > p_A   (one-sided; we expect treatment to be better)

    Args:
        clicks_a: Shape (N, K) binary click array for control.
        clicks_b: Shape (N, K) binary click array for treatment.

    Returns:
        Dictionary with keys: z_stat, p_value, ctr_a, ctr_b, significant.
    """
    total_a = clicks_a.size
    total_b = clicks_b.size
    successes_a = int(clicks_a.sum())
    successes_b = int(clicks_b.sum())

    ctr_a = successes_a / total_a
    ctr_b = successes_b / total_b

    # Pooled proportion under H0
    p_pool = (successes_a + successes_b) / (total_a + total_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))

    if se == 0:
        raise ValueError("Standard error is zero; cannot compute z-test")

    z = (ctr_b - ctr_a) / se
    # One-sided p-value: P(Z > z) under H0
    p_value = float(1 - stats.norm.cdf(z))

    return {
        "test": "Two-proportion z-test",
        "z_stat": float(z),
        "p_value": p_value,
        "ctr_a": ctr_a,
        "ctr_b": ctr_b,
        "significant": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Test 2: Mann–Whitney U test
# ---------------------------------------------------------------------------

def mann_whitney_test(
    ctr_per_query_a: np.ndarray,
    ctr_per_query_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Non-parametric comparison of per-query CTR distributions.

    Why: Per-query CTR distributions are often right-skewed and non-normal
    (many zero-click queries). Mann-Whitney U is rank-based and makes no
    distributional assumptions, making it robust to these violations.

    H0: The two distributions are identical (stochastic equality).
    H1: P(CTR_B > CTR_A) > 0.5  (one-sided)

    Args:
        ctr_per_query_a: Shape (N,) per-query CTR for control.
        ctr_per_query_b: Shape (N,) per-query CTR for treatment.

    Returns:
        Dictionary with keys: u_stat, p_value, significant.
    """
    if ctr_per_query_a.size == 0 or ctr_per_query_b.size == 0:
        raise ValueError("CTR arrays must not be empty")

    u_stat, p_value = stats.mannwhitneyu(
        ctr_per_query_b,
        ctr_per_query_a,
        alternative="greater",  # one-sided: treatment > control
    )

    return {
        "test": "Mann-Whitney U test",
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "significant": float(p_value) < 0.05,
    }


# ---------------------------------------------------------------------------
# Test 3: Paired t-test on proxy NDCG
# ---------------------------------------------------------------------------

def paired_ttest(
    proxy_ndcg_a: np.ndarray,
    proxy_ndcg_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Paired t-test on per-query proxy NDCG scores.

    Why: Since A and B share the same query set (same query IDs, same order),
    the proxy NDCG scores are naturally paired. A paired t-test removes
    between-query variance, increasing statistical power compared to
    an independent-samples test.

    H0: mean(proxy_NDCG_B - proxy_NDCG_A) == 0
    H1: mean(proxy_NDCG_B - proxy_NDCG_A) > 0  (one-sided)

    Args:
        proxy_ndcg_a: Shape (N,) proxy NDCG scores for control.
        proxy_ndcg_b: Shape (N,) proxy NDCG scores for treatment.

    Returns:
        Dictionary with keys: t_stat, p_value, mean_diff, significant.
    """
    if proxy_ndcg_a.shape != proxy_ndcg_b.shape:
        raise ValueError("Paired arrays must have identical shapes")
    if proxy_ndcg_a.size == 0:
        raise ValueError("Proxy NDCG arrays must not be empty")

    differences = proxy_ndcg_b - proxy_ndcg_a
    # One-sided: t-test tests H1: mean(diff) > 0
    t_stat, p_two_sided = stats.ttest_1samp(differences, popmean=0)
    # Convert two-sided p to one-sided (t > 0 direction)
    p_value = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

    return {
        "test": "Paired t-test (proxy NDCG)",
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(differences.mean()),
        "significant": float(p_value) < 0.05,
    }


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def power_analysis(
    ctr_a: float,
    ctr_b: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Lightweight power analysis for the two-proportion z-test.

    Computes:
    1. Achieved power given observed effect and sample size.
    2. Minimum N per group required to reach 80% power.

    Uses Cohen's h as the effect size for proportions:
        h = 2 * arcsin(sqrt(p_B)) - 2 * arcsin(sqrt(p_A))

    Args:
        ctr_a: Aggregate CTR for control.
        ctr_b: Aggregate CTR for treatment.
        n_per_group: Number of queries per group.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary with achieved_power, min_n_for_80pct_power, effect_size_h.
    """
    if ctr_a <= 0 or ctr_b <= 0:
        raise ValueError("CTR values must be positive for power analysis")

    # Cohen's h: effect size for proportions
    h = float(proportion_effectsize(ctr_b, ctr_a))

    analysis = NormalIndPower()

    # Achieved power at observed N
    achieved_power = analysis.solve_power(
        effect_size=h,
        nobs1=n_per_group,
        alpha=alpha,
        alternative="larger",  # one-sided
    )

    # Minimum N for 80% power
    min_n = analysis.solve_power(
        effect_size=h,
        power=0.80,
        alpha=alpha,
        alternative="larger",
    )

    return {
        "effect_size_h": h,
        "n_per_group": n_per_group,
        "alpha": alpha,
        "achieved_power": float(achieved_power),
        "min_n_for_80pct_power": int(np.ceil(min_n)),
    }


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_all_tests(
    clicks_a: np.ndarray,
    clicks_b: np.ndarray,
    ctr_per_query_a: np.ndarray,
    ctr_per_query_b: np.ndarray,
    proxy_ndcg_a: np.ndarray,
    proxy_ndcg_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Run all statistical tests and power analysis and return combined results.

    Args:
        clicks_a:          Shape (N, K) binary clicks for control.
        clicks_b:          Shape (N, K) binary clicks for treatment.
        ctr_per_query_a:   Shape (N,) per-query CTR for control.
        ctr_per_query_b:   Shape (N,) per-query CTR for treatment.
        proxy_ndcg_a:      Shape (N,) proxy NDCG for control.
        proxy_ndcg_b:      Shape (N,) proxy NDCG for treatment.

    Returns:
        Nested dictionary with results for each test and power analysis.
    """
    z_results = two_proportion_ztest(clicks_a, clicks_b)
    mw_results = mann_whitney_test(ctr_per_query_a, ctr_per_query_b)
    tt_results = paired_ttest(proxy_ndcg_a, proxy_ndcg_b)

    # Power analysis operates at impression level (N*K total) to match the
    # z-test, which pools all click/non-click events across positions.
    n_impressions_per_group = clicks_a.size
    power = power_analysis(
        ctr_a=z_results["ctr_a"],
        ctr_b=z_results["ctr_b"],
        n_per_group=n_impressions_per_group,
    )

    return {
        "z_test": z_results,
        "mann_whitney": mw_results,
        "paired_ttest": tt_results,
        "power_analysis": power,
    }
