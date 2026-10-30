"""Full-run wrapper for evaluate_load_aware_v2.py.

Runs with the default EPOCHS=80 (unmodified).
All output files use _full suffix so trial (_v2) results are preserved.
Training-only regime cutoffs are used throughout; alpha is selected on
validation MAE only; test data is never used to choose alpha.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate_load_aware_v2 as m

# Confirm epoch count before training starts
print(f"\n>>> Full run: EPOCHS = {m.EPOCHS}  (default, unmodified)")
print(f">>> Seeds: {m.SEEDS}")
print(f">>> Alphas: {m.ALPHAS}")

OUTPUTS = ROOT / "outputs"

# Redirect every output path to _full filenames
m.V2_SELECTED_ALPHAS_CSV  = OUTPUTS / "load_aware_selected_alphas_full.csv"
m.V2_VAL_SCORES_CSV       = OUTPUTS / "load_aware_validation_regime_scores_full.csv"
m.V2_RESULTS_CSV          = OUTPUTS / "load_aware_model_results_full.csv"
m.V2_SUMMARY_CSV          = OUTPUTS / "load_aware_model_summary_full.csv"
m.V2_REGIMES_CSV          = OUTPUTS / "load_aware_regimes_summary_full.csv"
m.V2_COMPARISON_CHART     = OUTPUTS / "load_aware_comparison_full.png"
m.V2_VAL_REGIMES_CHART    = OUTPUTS / "load_aware_val_regimes_full.png"
m.V2_LOAD_RANGES_CSV      = OUTPUTS / "load_aware_load_ranges_full.csv"

m.main()
