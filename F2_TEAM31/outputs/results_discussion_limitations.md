# Results, Discussion, and Limitations

*Source files: `outputs/*_full.csv` (80-epoch run, 3 seeds: 7, 21, 42).*
*No result CSVs were modified in producing this document.*

---

## 4  Results

### 4.1  Dataset and Split Statistics

The Abilene dataset covers one week of five-minute traffic snapshots across
12 routers and 30 internal directed links, yielding 1,209 timesteps in total.
Sliding windows of length H = 12 (60 minutes) produce 1,197 training windows,
391 validation windows, and 392 test windows.  Table 1 reports the observed
total-link-load at the last input step of each window (the signal used for
regime classification).

**Table 1. Observed Input Load Ranges by Split (Gbps, across all 30 links)**

| Split | N | Min | P10 | P33 | P50 | P67 | P90 | Max | Mean | Std |
|-------|-----|------|------|------|------|------|-------|-------|------|------|
| Train | 1,197 | 5.43 | 6.73 | 7.87 | 8.44 | 9.20 | 10.18 | 22.12 | 8.54 | 1.53 |
| Val   | 391   | 4.74 | 5.31 | 6.25 | 6.89 | 7.52 | 8.87  | 14.35 | 6.98 | 1.35 |
| Test  | 392   | 4.98 | 5.31 | 5.66 | 5.87 | 6.07 | 6.58  |  7.26 | 5.89 | 0.48 |

The training-defined regime boundaries—P33.3 = **7.87 Gbps** (Low/Medium
boundary) and P66.7 = **9.20 Gbps** (Medium/High boundary)—were computed from
training windows only.  The test set maximum (7.26 Gbps) falls below the
training P33.3 cutoff.  Consequently, all 392 test windows fall in the
training-defined **Low** regime; the Medium and High test regime bands contain
zero windows.  This is a dataset shift: the test week is a lower-traffic
period than the training week, entirely within the low-load tail of the
training distribution.

### 4.2  Alpha Selection on Validation Data

The candidate adjacency mixing weight α was chosen independently for each
training-defined regime (Low, Medium, High) and each random seed using
validation MAE alone; test data played no role in selection.  Table 2 reports
the selected α* and the corresponding validation MAE for each combination.

**Table 2. Selected α* per Regime and Seed (validation MAE criterion)**

| Seed | Regime | Cutoff | α* | Val MAE (Mbps) | Val RMSE (Mbps) |
|------|--------|--------|-----|---------------|----------------|
| 7  | Low    | ≤ 7.87 Gbps        | **0.00** | 21.81 | 31.58 |
| 7  | Medium | (7.87, 9.20] Gbps  | **0.00** | 23.45 | 40.70 |
| 7  | High   | > 9.20 Gbps        | **0.00** | 37.98 | 81.06 |
| 21 | Low    | ≤ 7.87 Gbps        | **0.00** | 21.86 | 31.78 |
| 21 | Medium | (7.87, 9.20] Gbps  | **0.00** | 23.51 | 40.53 |
| 21 | High   | > 9.20 Gbps        | **0.00** | 37.86 | 83.54 |
| 42 | Low    | ≤ 7.87 Gbps        | **0.00** | 21.97 | 31.80 |
| 42 | Medium | (7.87, 9.20] Gbps  | **0.00** | 23.92 | 41.32 |
| 42 | High   | > 9.20 Gbps        | **0.00** | 39.31 | 90.26 |

Across all nine (regime, seed) combinations the policy uniformly selected
α* = 0.00.  The load-aware policy therefore reduces to a **static
Similarity-only Graph-GRU** (α = 0).  It is not adaptive: the physical graph
component is never used.  This is reported factually; model labels in all
subsequent tables reflect this outcome.

### 4.3  Overall Test Performance (Primary Evaluation)

The primary evaluation applies training-defined cutoffs to classify test
windows and reports performance over all 392 test windows, which all fall in
the Low regime.  Results are averaged over three seeds with mean and
sample standard deviation.

**Table 3. Overall Test Performance (Mean ± Std, 3 seeds)**

| Model | MAE (Mbps) | RMSE (Mbps) |
|-------|-----------|------------|
| Similarity-only Graph-GRU (α = 0.00) | **22.28 ± 0.03** | **30.68 ± 0.06** |
| Fixed 50/50 Mixed Graph-GRU (α = 0.50) | 22.91 ± 0.34 | 32.72 ± 0.64 |
| GRU Temporal Baseline | 22.80 ± 1.35 | 33.25 ± 2.54 |
| Physical Graph-GRU (α = 1.00) | 25.47 ± 0.14 | 36.66 ± 0.40 |

*Note: The α*-selected policy row is omitted from Table 3 because its
predictions are numerically identical to the Similarity-only row (α* = 0
everywhere).*

The Similarity-only Graph-GRU achieves the lowest test MAE (22.28 Mbps) and
RMSE (30.68 Mbps).  The Fixed 50/50 model is 2.8% higher in MAE and 6.7%
higher in RMSE.  The GRU temporal baseline is comparable to the 50/50 model
in MAE (22.80 Mbps) but shows higher variability across seeds (std 1.35 vs.
0.34 Mbps for the 50/50 model).  The Physical Graph-GRU (α = 1.00) is the
worst performer across both metrics (MAE 25.47 Mbps, RMSE 36.66 Mbps) in
this experiment.

**Per-regime test results (training-cutoff definition)**

Under the training-defined regime policy only one regime is populated on the
test set:

| Regime (training cutoffs) | N windows | Similarity-only MAE | GRU MAE | Physical MAE |
|--------------------------|-----------|--------------------|---------|----|
| Low (≤ 7.87 Gbps) | 392 | 22.28 ± 0.03 Mbps | 22.80 ± 1.35 Mbps | 25.47 ± 0.14 Mbps |
| Medium (7.87–9.20 Gbps) | **0** | — | — | — |
| High (> 9.20 Gbps) | **0** | — | — | — |

**Medium and High test regime performance could not be evaluated under
the training-cutoff regime definition.**  In this particular split the test
period carries lower observed load than the training period, placing all test
windows below the training-defined Low/Medium boundary.

### 4.4  Retrospective Analysis: Test-Local Tertile Breakdown

To provide descriptive insight into performance variation within the test set
we partition the 392 test windows into tertiles defined by the test
distribution itself (P33 = 5.66 Gbps, P67 = 6.07 Gbps).  These boundaries
are computed from test data and therefore cannot be used prospectively; this
is a purely retrospective, post-hoc analysis.  It does not constitute a
regime evaluation under the training-defined policy and was not used to select
α.

**Table 4. Retrospective Test-Local Tertile Performance (Mean ± Std, 3 seeds)**

| Test-local tertile | N | Similarity-only MAE | Fixed 50/50 MAE | GRU MAE | Physical MAE |
|-------------------|---|--------------------|-----------------|---------|----|
| Low  (≤ 5.66 Gbps) | 131 | 21.39 ± 0.02 Mbps | 24.60 ± 0.59 Mbps | 26.67 ± 2.23 Mbps | 27.26 ± 0.29 Mbps |
| Medium (5.66–6.07 Gbps) | 130 | 21.69 ± 0.03 Mbps | 22.94 ± 0.37 Mbps | 22.16 ± 1.56 Mbps | 26.66 ± 0.15 Mbps |
| High (> 6.07 Gbps) | 131 | 23.75 ± 0.05 Mbps | 21.19 ± 0.07 Mbps | 19.57 ± 0.29 Mbps | 22.50 ± 0.03 Mbps |

Across all three test-local bands the Similarity-only model (α = 0) is
competitive but not uniformly best.  In the test-local High band the GRU
baseline achieves 19.57 Mbps and the 50/50 mix achieves 21.19 Mbps, both
below the Similarity-only result of 23.75 Mbps.  However, because these
tertile boundaries were derived from the test set, this comparison has no
prospective validity and should not be used to motivate α selection or
architecture choices.

---

## 5  Discussion

### 5.1  Why the Similarity Graph Dominates

The validation sweep across α ∈ {0.00, 0.25, 0.50, 0.75, 1.00} showed
monotonically increasing validation MAE as α increased in all three
training-defined regimes and across all three seeds (Table 2 and
`load_aware_validation_regime_scores_full.csv`).  This indicates that, for
Abilene traffic forecasting, correlation-derived link similarity is a more
informative structural prior than physical co-endpoint adjacency.  Physical
links that share a router endpoint are not necessarily correlated in traffic
volume; conversely, links with high Pearson correlation (regardless of
topology) tend to exhibit co-movement that the GRU can exploit when the
graph propagates those signals.

### 5.2  Non-Adaptive Outcome of the Load-Aware Policy

The experiment was designed to test whether the optimal graph-mixing weight
varies with the observed input load regime.  The answer, for this dataset and
this α grid, is no: α* = 0.00 is optimal on the validation set in every
regime.  The load-aware framework therefore produced a static policy, not an
adaptive one.  This is an empirically valid and falsifiable finding—the
framework could in principle have selected different α values per regime had
the validation evidence supported it.

It would be incorrect to describe the resulting model as "load-aware" or
"adaptive" in a report.  The correct description is: a Similarity-only
Graph-GRU (α = 0) whose mixing weight was selected using the validation set
under a regime-conditioned policy that uniformly chose pure similarity
weighting.

An additional caveat applies to the High-load regime: the validation set
contains only 18 windows in the High training-defined regime (> 9.20 Gbps),
compared with 300 Low and 73 Medium windows.  The α* selected for High is
therefore based on limited evidence and may not be reliable.

### 5.3  Generalisation to the Physical Graph-GRU

The Physical Graph-GRU (α = 1.00) performs worst overall (MAE 25.47 Mbps,
RMSE 36.66 Mbps).  The gap to the Similarity-only model is 3.19 Mbps in MAE
(+14%) and 5.98 Mbps in RMSE (+19%).  This reinforces that raw physical
topology, without traffic-correlation weighting, adds noise rather than
signal for this task.  The Fixed 50/50 Mixed model sits between the two
extremes (22.91 Mbps), consistent with partial dilution of the correlation
signal by the physical graph.

### 5.4  Stability Across Seeds

The Similarity-only and α*-selected models exhibit very low seed variance
(MAE std = 0.03 Mbps), as do the Physical (0.14 Mbps) and 50/50 (0.34 Mbps)
models.  The GRU temporal baseline shows higher variability (std 1.35 Mbps in
MAE, 2.54 Mbps in RMSE) in this experiment.  With only three seeds this
difference is noted as an observation rather than a definitive finding.

### 5.5  Train-to-Test Distribution Shift

The mean observed load drops from 8.54 Gbps in training to 5.89 Gbps in
testing—a reduction of 31% in this particular split.  The training P33.3
cutoff (7.87 Gbps) exceeds the test maximum (7.26 Gbps) by 0.61 Gbps.  As a
result, none of the 392 test windows reach the training-defined Medium or
High bands.  Whether this pattern generalises to other splits or longer
observation windows is unknown from this experiment alone.

---

## 6  Limitations

**One-week dataset.**  The Abilene dataset used in this study spans
approximately one week of five-minute measurements.  This narrow temporal
window captures a single traffic pattern cycle and provides limited evidence
for generalisation to different days of the week, seasonal patterns, or
traffic events outside the observed period.  The three-way chronological
split (60/20/20) further reduces the effective size of each partition; the
validation set (391 windows) and test set (392 windows) each cover roughly
33 hours of traffic.

**Three random seeds.**  Reported means and standard deviations are computed
over three seeds (7, 21, 42).  With n = 3 the standard deviation is an
unreliable estimator of true variability and the standard error is large.
Conclusions about seed stability should be treated as indicative rather than
statistically rigorous.

**Train-to-test load shift.**  As documented in Section 4.1, the test period
is systematically lower in traffic than the training period.  The
training-defined regime cutoffs (P33 = 7.87 Gbps, P67 = 9.20 Gbps) place all
392 test windows in the Low regime.  The experiment therefore provides no
evidence about model performance in Medium- or High-load conditions on held-out
data.  Any claim about regime-specific generalisation to Medium or High load
would require a dataset spanning a broader and more representative traffic
range, or a re-partitioning strategy that distributes diverse load conditions
across all three splits.

**Coarse alpha grid.**  The candidate set α ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
uses a step size of 0.25.  A finer grid (e.g., step 0.05) or continuous
optimisation of α could reveal a non-zero optimum that the current search
missed.  The uniform selection of α* = 0.00 may partly reflect the resolution
of the grid rather than a fundamental dominance of the similarity graph.

**Static regime boundaries.**  Regime cutoffs are computed once from the full
training set.  In a streaming deployment the traffic distribution may drift,
making static P33/P67 boundaries stale over time.  Adaptive or rolling
boundary estimates would be needed for a production load-aware routing system.

**Single dataset.**  All results are from the Abilene backbone network.
Abilene has a fixed 12-router, 30-directed-link topology carrying
research-grade traffic.  Performance on commercial ISP networks,
data-centre fabrics, or access networks with different traffic statistics
and topology characteristics is unknown.

---

*All numerical values in this document are sourced from
`outputs/load_aware_model_summary_full.csv`,
`outputs/load_aware_regimes_summary_full.csv`,
`outputs/load_aware_selected_alphas_full.csv`, and
`outputs/load_aware_load_ranges_full.csv`.
No result files were modified.*
