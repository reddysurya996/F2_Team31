# Section 6  Conclusion and Future Work

*Source files: `outputs/*_full.csv` (80-epoch run, 3 seeds: 7, 21, 42).*
*No code or result files were modified in producing this document.*

---

## 6.1  Conclusion

This paper investigated whether adapting the adjacency mixing weight α of a
Graph-GRU model to the observed network load regime improves traffic
forecasting on the Abilene backbone network.  The experiment applied a
regime-conditioned policy: training-data tertiles defined three load bands
(Low ≤ 7.87 Gbps, Medium 7.87–9.20 Gbps, High > 9.20 Gbps), and α was
selected independently for each band using validation-set MAE only, with no
access to test data during selection.

**The policy uniformly selected α* = 0.00 across all three regimes and all
three random seeds.**  The resulting model is therefore a static
Similarity-only Graph-GRU (α = 0), not an adaptive one.  This is an
empirically grounded finding: the framework was capable of selecting
different α values per regime, but the validation evidence did not support
any non-zero mixing weight in this experiment.

On the 392 test windows—all of which fall in the training-defined Low
regime due to an observed load shift between the training and test
periods—the Similarity-only Graph-GRU achieves a test MAE of
**22.28 ± 0.03 Mbps** and RMSE of **30.68 ± 0.06 Mbps** (mean ± std across
three seeds).  The GRU temporal baseline and Fixed 50/50 Mixed Graph-GRU
record MAEs of 22.80 ± 1.35 Mbps and 22.91 ± 0.34 Mbps respectively; the
Physical Graph-GRU (α = 1.00) is the weakest model at 25.47 ± 0.14 Mbps.
The Similarity-only result is narrowly better than the GRU baseline and
the 50/50 mix in this particular split, but the differences are small and
the experiment covers only low-load test conditions; no strong general
claim about superiority across load regimes is supported by this data.

The primary evaluation is confined to the Low test regime because in this
split the test-set maximum (7.26 Gbps) falls 0.61 Gbps below the
training-defined Low/Medium boundary.  **Medium and High test regime
performance could not be evaluated under the training-cutoff policy.**
A retrospective analysis partitioning test windows by their own distribution
(post-hoc tertiles) showed that the Similarity-only model was not uniformly
best across all within-test bands—the GRU baseline achieved a lower MAE in
the test-local High band (19.57 ± 0.29 Mbps vs. 23.75 ± 0.05 Mbps).
Because those tertile boundaries were derived from the test set itself, this
comparison has no prospective validity and is reported for descriptive
purposes only.

The High-load validation regime contained only 18 windows, so the α*
selected for that regime is based on limited evidence.  With three seeds and
one week of data, all quantitative findings should be read as observations
from this particular experimental setting rather than general conclusions.

In summary, the load-regime framework is implemented correctly and is free
of data leakage, but it did not produce an adaptive policy on this dataset.
The static Similarity-only Graph-GRU is the de facto outcome of the
validation-based selection procedure.

---

## 6.2  Future Work

The following directions address the main limitations identified in Section 6
of the Results, Discussion, and Limitations document.

### 6.2.1  Larger and More Diverse Traffic Data

The most direct limitation of this study is the single week of Abilene
data.  Future work should:

- **Extend the observation window** to multiple weeks or months of Abilene
  traffic so that the test period contains windows in the Medium and High
  training-defined regimes.  Without this it is impossible to evaluate
  whether a regime-conditioned α policy provides any benefit over a static
  baseline under genuinely varied load conditions.
- **Use rolling or held-out week splits** rather than a single 60/20/20
  chronological cut.  A rolling evaluation across several non-overlapping
  test weeks would reveal whether the low-load shift observed in this
  split is consistent or incidental.
- **Test on additional networks** with different topologies and traffic
  patterns (e.g., GÉANT, Internet2, or a commercial ISP dataset) to
  assess whether the dominance of similarity-based adjacency is specific
  to the Abilene research network.

### 6.2.2  Genuinely Adaptive Graph-Weight Methods

The current framework selects a single scalar α per regime at training
time.  A natural extension is to make α a continuous, online function of
the observed load rather than a discrete lookup:

- **Hyper-network or attention gating**: Learn a small network that takes
  the observed input load (or a feature vector derived from it) and outputs
  α ∈ [0, 1] at inference time.  This would produce a genuinely adaptive
  mixing weight without requiring pre-defined regime boundaries.
- **Finer α grid or continuous optimisation**: The current candidate set
  {0.00, 0.25, 0.50, 0.75, 1.00} uses a step size of 0.25.  A finer grid
  or gradient-based optimisation of α per regime could reveal a non-zero
  optimal value that the coarse search missed, especially in the
  High-load validation band where only 18 windows were available.
- **Regime boundary learning**: Rather than fixing boundaries at training
  P33 and P67, learn soft or data-driven cluster assignments that may
  capture traffic structure not aligned with simple quantile thresholds.

### 6.2.3  Stronger Baselines and More Seeds

- **Additional baselines**: Including LSTM, Transformer-based sequence
  models, or graph attention networks would help situate the Graph-GRU
  family within a broader comparative landscape.
- **More random seeds**: With only three seeds the reported standard
  deviations are unreliable estimators of true variability.  Running at
  least five to ten seeds would allow tighter confidence intervals and
  more meaningful seed-stability comparisons.

### 6.2.4  Adaptive Regime Boundaries

Static P33/P67 boundaries computed once from training data can become stale
if traffic patterns evolve over time.  A production-ready load-aware system
should maintain rolling or exponentially weighted estimates of the load
distribution so that regime boundaries track the current network state.
Evaluating such a streaming regime classifier over multi-week Abilene traces
is a concrete and tractable next step.

---

*All numerical values in this document are sourced from
`outputs/load_aware_model_summary_full.csv`,
`outputs/load_aware_regimes_summary_full.csv`,
`outputs/load_aware_selected_alphas_full.csv`, and
`outputs/load_aware_load_ranges_full.csv`.
No code or result files were modified.*
