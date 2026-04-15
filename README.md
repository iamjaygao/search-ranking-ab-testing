# Search Ranking A/B Test Simulation

Simulates an online A/B test to validate whether offline search ranking improvements translate to measurable user engagement gains, using a Position-Based Click Model and rigorous statistical testing.

---

## Background

This project extends a team-based search ranking system built on the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data).

| Group | Model | Offline NDCG@10 |
|-------|-------|----------------|
| A (Control) | BM25 baseline | 0.804 |
| B (Treatment) | Neural reranker | 0.845 |

> This simulation was independently designed and implemented as an extension of the team project.

The core question: **does a ~5% offline NDCG gain translate to a statistically significant online CTR improvement?**

---

## Methodology

### Position-Based Click Model (PBM)

Users don't click results uniformly — links near the top receive far more attention than those further down. The PBM captures this with two independent factors:

```
P(click | rank k) = relevance(document) × examination_probability(k)
```

where examination probability decays logarithmically with rank:

```
v(k) = 1 / log₂(k + 2)
```

This means rank 1 is examined ~63% of the time and rank 10 only ~28%, regardless of relevance — a well-validated empirical finding in IR.

Relevance scores are drawn from calibrated Normal distributions that reflect the quality gap observed offline, then clicks are sampled as Bernoulli trials under the PBM.

### Statistical Tests

Three complementary tests guard against violations of any single method's assumptions:

| Test | Input | Why |
|------|-------|-----|
| **Two-proportion z-test** | Aggregate CTR (all impressions) | Simple, interpretable baseline; valid by CLT at large N |
| **Mann-Whitney U test** | Per-query CTR | Non-parametric; robust to skewed/zero-heavy distributions |
| **Paired t-test** | Per-query proxy NDCG | Exploits query-level pairing to reduce variance and increase power |

All tests use a one-sided alternative (H₁: Treatment > Control) at α = 0.05.

---

## Results

```
============================================================
  SEARCH RANKING A/B TEST — SIMULATION REPORT
============================================================

CLICK-THROUGH RATE SUMMARY
------------------------------------------------------------
  Control   CTR : 0.1329  (13.29%)
  Treatment CTR : 0.1472  (14.72%)
  Absolute lift : +0.0143
  Relative lift : +10.76%
  Cohen's d     : 0.1308

STATISTICAL TESTS  (α = 0.05, one-sided H₁: B > A)
------------------------------------------------------------
  [1] Two-proportion z-test
      z = 2.9137,  p = 0.0018  →  ✓ p < 0.05
  [2] Mann-Whitney U test (per-query CTR)
      U = 536062,  p = 0.0018  →  ✓ p < 0.05
  [3] Paired t-test (proxy NDCG)
      t = 2.9593,  p = 0.0016  →  ✓ p < 0.05
      Mean NDCG diff : +0.0164

POWER ANALYSIS
------------------------------------------------------------
  Effect size (Cohen's h) : 0.0412
  N per group             : 10000
  Achieved power          : 0.8979  (89.8%)
  Min N for 80% power     : 7279

CONCLUSION
------------------------------------------------------------
  All three tests reject H₀ (p < 0.05). The neural reranker produces
  a statistically significant CTR improvement of +10.76%
  over the BM25 baseline, consistent with the offline NDCG gain
  (0.804 → 0.845). These results indicate that offline improvements are associated with measurable gains in simulated online engagement.
============================================================
```

---

## How to Run

```bash
pip install -r requirements.txt
python simulate.py
```

Requires Python 3.9+.

---

## Project Structure

| File | Description |
|------|-------------|
| `click_model.py` | Position-Based Click Model: position bias + click simulation |
| `metrics.py` | Aggregate CTR, lift, Cohen's d, proxy NDCG |
| `ab_test.py` | Two-proportion z-test, Mann-Whitney U, paired t-test, power analysis |
| `simulate.py` | Simulation runner and formatted report |
| `requirements.txt` | Dependencies: numpy, scipy, statsmodels |
| `README.md`         | Project documentation                   |

---

## Key Takeaway

All three statistical tests (p < 0.002) and a well-powered experiment (89.8%) confirm that the neural reranker's offline NDCG improvement (0.804 → 0.845) translates to a statistically significant **+10.76% CTR lift** in simulation. This provides evidence that offline NDCG can serve as a useful proxy for online engagement in this simulated setting. 

---

## Notes on Simulation Design

- Relevance scores are drawn from calibrated distributions — they are **not** NDCG values directly. The PBM produces click probabilities consistent with the observed offline quality gap.
- Query-level scores are paired across A and B (same query set), enabling the paired t-test to eliminate between-query noise.
- Power analysis operates at the impression level (N × K = 10,000) to be consistent with the z-test, which pools all click events across rank positions.
