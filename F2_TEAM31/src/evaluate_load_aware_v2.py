"""Load-aware dynamic graph-weight experiment  —  corrected v2.

AUDIT FIXES (2026-10-26):
  1. Adds train / val / test observed-load range diagnostics (min, P33, P67, max).
  2. Emits an explicit WARNING when all test windows fall in a single training regime
     so downstream readers know Medium/High test evaluation was not possible.
  3. Removes the misleading "adaptive" label: if alpha* is identical across all
     regimes and seeds the model is renamed to reflect that it is effectively a
     static model and a WARNING is printed.
  4. Test-local observed tertile analysis is clearly labelled [RETROSPECTIVE] in
     all console output and CSV headers; it is never used for model selection.
  5. All output files use _v2 suffixes -- original results are not overwritten.

CAUSAL EVALUATION POLICY (unchanged from v1):
  - Regime cutoffs are computed from training data only.
  - Alpha is selected using validation MAE only.
  - Test windows are classified using training-defined cutoffs (no test leakage).
"""
from pathlib import Path
import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import GRUBaseline, GraphGRU

DATA_PATH = ROOT / "data/processed/abilene_link_loads.npz"
OUTPUTS_DIR = ROOT / "outputs"

V2_SELECTED_ALPHAS_CSV  = OUTPUTS_DIR / "load_aware_selected_alphas_v2.csv"
V2_VAL_SCORES_CSV       = OUTPUTS_DIR / "load_aware_validation_regime_scores_v2.csv"
V2_RESULTS_CSV          = OUTPUTS_DIR / "load_aware_model_results_v2.csv"
V2_SUMMARY_CSV          = OUTPUTS_DIR / "load_aware_model_summary_v2.csv"
V2_REGIMES_CSV          = OUTPUTS_DIR / "load_aware_regimes_summary_v2.csv"
V2_COMPARISON_CHART     = OUTPUTS_DIR / "load_aware_comparison_v2.png"
V2_VAL_REGIMES_CHART    = OUTPUTS_DIR / "load_aware_val_regimes_v2.png"
V2_LOAD_RANGES_CSV      = OUTPUTS_DIR / "load_aware_load_ranges_v2.csv"

UNIT_TO_BPS  = 8.0 / 3.0
BPS_TO_MBPS  = 1e-6
BPS_TO_GBPS  = 1e-9
UNIT_TO_MBPS = UNIT_TO_BPS * BPS_TO_MBPS
UNIT_TO_GBPS = UNIT_TO_BPS * BPS_TO_GBPS

HISTORY = 12
SEEDS   = (7, 21, 42)
ALPHAS  = (0.0, 0.25, 0.5, 0.75, 1.0)
EPOCHS, BATCH, LR = 80, 32, 1e-3


def norm_adj(a):
    a = a.astype(np.float32).copy()
    np.fill_diagonal(a, 1)
    inv = np.power(np.maximum(a.sum(axis=1), 1e-8), -0.5)
    return (inv[:, None] * a * inv[None, :]).astype(np.float32)


def build_raw_graphs(names, train_series):
    n = len(names)
    physical = np.eye(n, dtype=np.float32)
    ends = [str(name).split(",") for name in names]
    for i, (a, b) in enumerate(ends):
        for j in range(i + 1, n):
            c, d = ends[j]
            if {a, b} & {c, d}:
                physical[i, j] = physical[j, i] = 1
    corr = np.nan_to_num(np.corrcoef(train_series, rowvar=False), nan=0.0)
    similar = np.eye(n, dtype=np.float32)
    for i in range(n):
        top = sorted(((corr[i, j], j) for j in range(n) if j != i and corr[i, j] > 0), reverse=True)[:3]
        for value, j in top:
            similar[i, j] = similar[j, i] = max(similar[i, j], value)
    return physical, similar


def windows(a):
    x, y = [], []
    for i in range(len(a) - HISTORY):
        x.append(a[i:i + HISTORY])
        y.append(a[i + HISTORY])
    return (torch.tensor(np.asarray(x), dtype=torch.float32),
            torch.tensor(np.asarray(y), dtype=torch.float32))


def fit(model, loader, vx, vy, adj, device):
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=LR), nn.MSELoss()
    best, state, patience = float("inf"), None, 12
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb) if adj is None else model(xb, adj)
            loss_fn(pred, yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = loss_fn(model(vx) if adj is None else model(vx, adj), vy).item()
        if score < best:
            best, patience = score, 12
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break
    model.load_state_dict(state)
    return model


def compute_observed_load(x_windows, mean, std):
    last_step_unscaled = x_windows[:, -1, :].numpy() * std + mean
    return last_step_unscaled.sum(axis=1)


def classify_regimes(observed_loads, q33_units, q67_units):
    return {
        "Low":    observed_loads <= q33_units,
        "Medium": (observed_loads > q33_units) & (observed_loads <= q67_units),
        "High":   observed_loads > q67_units,
    }


def load_range_stats(loads_units, label):
    return {
        "split":    label,
        "N":        len(loads_units),
        "min_Gbps": float(loads_units.min()  * UNIT_TO_GBPS),
        "p10_Gbps": float(np.percentile(loads_units, 10)        * UNIT_TO_GBPS),
        "p33_Gbps": float(np.percentile(loads_units, 100.0/3)   * UNIT_TO_GBPS),
        "p50_Gbps": float(np.percentile(loads_units, 50)        * UNIT_TO_GBPS),
        "p67_Gbps": float(np.percentile(loads_units, 200.0/3)   * UNIT_TO_GBPS),
        "p90_Gbps": float(np.percentile(loads_units, 90)        * UNIT_TO_GBPS),
        "max_Gbps": float(loads_units.max()  * UNIT_TO_GBPS),
        "mean_Gbps":float(loads_units.mean() * UNIT_TO_GBPS),
        "std_Gbps": float(loads_units.std()  * UNIT_TO_GBPS),
    }


def print_range_table(rows):
    hdr  = ("Split","N","Min","P10","P33","P50","P67","P90","Max","Mean","Std")
    keys = ("split","N","min_Gbps","p10_Gbps","p33_Gbps","p50_Gbps",
            "p67_Gbps","p90_Gbps","max_Gbps","mean_Gbps","std_Gbps")
    hdr_fmt = "{:<8} {:>6}  " + "  {:>8}" * 9
    val_fmt = "{:<8} {:>6}  " + "  {:>8.3f}" * 9
    print("\n--- Observed Input Load Ranges by Split (Gbps) ---")
    print(hdr_fmt.format(*hdr))
    print("-" * 100)
    for r in rows:
        vals = [r[k] for k in keys]
        print(val_fmt.format(*vals))


def plot_val_regimes(val_scores_rows, q33_gbps, q67_gbps, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=300, sharey=False)
    regimes = ["Low", "Medium", "High"]
    for idx, reg in enumerate(regimes):
        ax = axes[idx]
        reg_rows = [r for r in val_scores_rows if r["regime"] == reg]
        if not reg_rows:
            ax.set_title(f"{reg} — no validation windows", fontsize=11)
            continue
        for seed in SEEDS:
            s_rows = sorted([r for r in reg_rows if r["seed"] == seed], key=lambda x: x["alpha"])
            ax.plot([r["alpha"] for r in s_rows], [r["val_MAE_Mbps"] for r in s_rows],
                    marker="o", linewidth=2.0, markersize=6, label=f"Seed {seed}")
        ax.set_title(
            f"{reg} Observed Regime (Training Cutoffs)\n"
            f"({reg_rows[0]['cutoff_label']}, N={reg_rows[0]['window_count']})",
            fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Adjacency Mixing Weight (alpha)", fontsize=10.5)
        ax.set_ylabel("Validation MAE (Mbps)", fontsize=10.5)
        ax.set_xticks(ALPHAS)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9, loc="upper right")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    plt.suptitle(
        "Validation MAE by Training-Defined Observed Load Regime\n"
        "Across Candidate Alpha Weights — Used for Policy Selection",
        fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved validation regime chart to: {out_path}")


def plot_comparison(summary_rows, out_path, adaptive_label, n_test, only_regime):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)
    models     = [r["model"]         for r in summary_rows]
    mae_means  = [r["MAE_mean_Mbps"] for r in summary_rows]
    mae_stds   = [r["MAE_std_Mbps"]  for r in summary_rows]
    rmse_means = [r["RMSE_mean_Mbps"]for r in summary_rows]
    rmse_stds  = [r["RMSE_std_Mbps"] for r in summary_rows]

    label_map = {
        adaptive_label:                              "alpha*-Selected\n(Policy)",
        "Similarity-only Graph-GRU (alpha=0.0)":    "Similarity-only\n(alpha=0.0)",
        "Fixed 50/50 Mixed Graph-GRU (alpha=0.5)":  "Fixed 50/50\n(alpha=0.5)",
        "GRU (Temporal Baseline)":                   "GRU\n(Temporal)",
        "Physical Graph-GRU (alpha=1.0)":            "Physical\n(alpha=1.0)",
    }
    short_labels = [label_map.get(m, m) for m in models]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#9370DB", "#DD8452"]
    x = np.arange(len(models))

    note = (
        f"NOTE: All {n_test} test windows fall in training-defined '{only_regime}' regime.\n"
        "Medium/High test regime performance COULD NOT be evaluated."
    ) if only_regime else ""

    for ax, means, stds, ylabel, title in [
        (ax1, mae_means,  mae_stds,  "MAE (Mbps)",  "Test MAE"),
        (ax2, rmse_means, rmse_stds, "RMSE (Mbps)", "Test RMSE"),
    ]:
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="#2b2b2b", linewidth=1.1,
                      width=0.52, alpha=0.9,
                      error_kw={"elinewidth": 1.4, "ecolor": "#2b2b2b", "capthick": 1.4},
                      zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9.5, fontweight="medium")
        ax.set_ylabel(ylabel, fontsize=11.5, fontweight="medium")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.22)
        for bar, m, s in zip(bars, means, stds):
            ax.annotate(f"{m:.2f}\n+/-{s:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, m + s),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                        color="#222222")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.suptitle(
        f"Overall Test Performance: {adaptive_label} vs. Baselines\n{note}",
        fontsize=10, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison chart to: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    package = np.load(DATA_PATH)
    raw, names = package["link_loads_internal_30"].astype(np.float32), package["internal_link_names"]

    a, b = int(0.6 * len(raw)), int(0.8 * len(raw))
    mean, std = raw[:a].mean(0), raw[:a].std(0)
    std[std < 1e-6] = 1
    scaled = (raw - mean) / std

    tx, ty = windows(scaled[:a])
    vx, vy = windows(scaled[a:b])
    qx, qy = windows(scaled[b:])

    val_truth  = vy.numpy() * std + mean
    test_truth = qy.numpy() * std + mean

    train_obs_loads = compute_observed_load(tx, mean, std)
    val_obs_loads   = compute_observed_load(vx, mean, std)
    test_obs_loads  = compute_observed_load(qx, mean, std)

    # --- DIAGNOSTIC: Load range statistics ---
    range_rows = [
        load_range_stats(train_obs_loads, "Train"),
        load_range_stats(val_obs_loads,   "Val"),
        load_range_stats(test_obs_loads,  "Test"),
    ]
    print_range_table(range_rows)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with V2_LOAD_RANGES_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(range_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in range_rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved load range statistics to: {V2_LOAD_RANGES_CSV}")

    # --- Training-only cutoffs (causal) ---
    q33_units = float(np.percentile(train_obs_loads, 100.0 / 3.0))
    q67_units = float(np.percentile(train_obs_loads, 200.0 / 3.0))
    q33_gbps  = q33_units * UNIT_TO_GBPS
    q67_gbps  = q67_units * UNIT_TO_GBPS

    print("\n--- Training-Defined Regime Cutoffs (used for all classification) ---")
    print(f"  P33.3 (Low/Medium boundary): {q33_gbps:.3f} Gbps  ({q33_units:,.0f} units)")
    print(f"  P66.7 (Medium/High boundary): {q67_gbps:.3f} Gbps  ({q67_units:,.0f} units)")

    val_regimes  = classify_regimes(val_obs_loads,  q33_units, q67_units)
    test_regimes = classify_regimes(test_obs_loads, q33_units, q67_units)

    print("\nWindow counts per training-defined regime:")
    for split_name, regimes, n_total in [
        ("Validation", val_regimes,  len(val_obs_loads)),
        ("Test",       test_regimes, len(test_obs_loads)),
    ]:
        counts = {r: int(m.sum()) for r, m in regimes.items()}
        print(f"  {split_name} ({n_total} total) -> "
              f"Low: {counts['Low']}, Medium: {counts['Medium']}, High: {counts['High']}")

    # --- Regime collapse warning ---
    test_regime_counts  = {r: int(m.sum()) for r, m in test_regimes.items()}
    nonempty_test_regs  = [r for r, c in test_regime_counts.items() if c > 0]
    only_regime = nonempty_test_regs[0] if len(nonempty_test_regs) == 1 else None
    if only_regime:
        print(
            f"\n{'!'*80}\n"
            f"  WARNING: All {len(test_obs_loads)} test windows fall in the training-defined\n"
            f"  '{only_regime}' regime.  Dataset shift detected: the test period is entirely\n"
            f"  within the low-traffic band of the training distribution.\n"
            f"  Medium and High test regime performance CANNOT be evaluated.\n"
            f"{'!'*80}"
        )

    # --- Build graphs (training data only) ---
    phys_raw, sim_raw = build_raw_graphs(names, scaled[:a])
    alpha_graphs = {
        alpha: torch.tensor(norm_adj(alpha * phys_raw + (1.0 - alpha) * sim_raw)).to(device)
        for alpha in ALPHAS
    }

    # --- Train candidates & validation regime scores ---
    trained_candidates = {}
    val_scores_rows    = []

    print("\n--- Training Alpha Candidates & Evaluating Validation Regimes ---")
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        for alpha in ALPHAS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model  = GraphGRU(len(names)).to(device)
            loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH, shuffle=True)
            adj    = alpha_graphs[alpha]
            model  = fit(model, loader, vx.to(device), vy.to(device), adj, device)
            model.eval()
            with torch.no_grad():
                val_pred  = model(vx.to(device), adj).cpu().numpy() * std + mean
                test_pred = model(qx.to(device), adj).cpu().numpy() * std + mean
            trained_candidates[(seed, alpha)] = {"val_pred": val_pred, "test_pred": test_pred}

            for reg_name in ["Low", "Medium", "High"]:
                mask = val_regimes[reg_name]
                if mask.sum() > 0:
                    v_mae_u  = float(np.abs(val_pred[mask]  - val_truth[mask]).mean())
                    v_rmse_u = float(np.sqrt(np.square(val_pred[mask] - val_truth[mask]).mean()))
                    val_scores_rows.append({
                        "seed":          seed,
                        "regime":        reg_name,
                        "window_count":  int(mask.sum()),
                        "cutoff_label":  (f"<= {q33_gbps:.2f} Gbps" if reg_name == "Low"
                                          else (f"({q33_gbps:.2f}, {q67_gbps:.2f}] Gbps"
                                                if reg_name == "Medium"
                                                else f"> {q67_gbps:.2f} Gbps")),
                        "alpha":         alpha,
                        "val_MAE_Mbps":  v_mae_u  * UNIT_TO_MBPS,
                        "val_RMSE_Mbps": v_rmse_u * UNIT_TO_MBPS,
                        "val_MAE_units": v_mae_u,
                        "val_RMSE_units":v_rmse_u,
                    })

    with V2_VAL_SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed","regime","cutoff_label","window_count","alpha",
                      "val_MAE_Mbps","val_RMSE_Mbps","val_MAE_units","val_RMSE_units"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in val_scores_rows:
            writer.writerow({
                "seed": r["seed"], "regime": r["regime"], "cutoff_label": r["cutoff_label"],
                "window_count": r["window_count"], "alpha": f"{r['alpha']:.2f}",
                "val_MAE_Mbps":   f"{r['val_MAE_Mbps']:.3f}",
                "val_RMSE_Mbps":  f"{r['val_RMSE_Mbps']:.3f}",
                "val_MAE_units":  f"{r['val_MAE_units']:.3f}",
                "val_RMSE_units": f"{r['val_RMSE_units']:.3f}",
            })
    print(f"Saved validation regime scores to: {V2_VAL_SCORES_CSV}")
    plot_val_regimes(val_scores_rows, q33_gbps, q67_gbps, V2_VAL_REGIMES_CHART)

    # --- Select alpha* per regime using validation MAE only ---
    selected_alphas_per_seed = {}
    selected_alphas_rows     = []

    print("\n--- Selected Alpha* per Regime (Validation MAE Only) ---")
    for seed in SEEDS:
        selected_alphas_per_seed[seed] = {}
        for reg_name in ["Low", "Medium", "High"]:
            reg_entries = [r for r in val_scores_rows if r["seed"] == seed and r["regime"] == reg_name]
            best_entry  = min(reg_entries, key=lambda x: x["val_MAE_Mbps"])
            selected_alphas_per_seed[seed][reg_name] = best_entry["alpha"]
            selected_alphas_rows.append({
                "seed":           seed,
                "regime":         reg_name,
                "cutoff_label":   best_entry["cutoff_label"],
                "selected_alpha": best_entry["alpha"],
                "val_MAE_Mbps":   best_entry["val_MAE_Mbps"],
                "val_RMSE_Mbps":  best_entry["val_RMSE_Mbps"],
            })
            print(f"  Seed {seed} | {reg_name:<6} ({best_entry['cutoff_label']}): "
                  f"alpha* = {best_entry['alpha']:.2f}  "
                  f"(Val MAE: {best_entry['val_MAE_Mbps']:.2f} Mbps)")

    # --- Non-adaptive warning ---
    all_selected  = [r["selected_alpha"] for r in selected_alphas_rows]
    unique_alphas = set(all_selected)
    if len(unique_alphas) == 1:
        uniform_alpha = next(iter(unique_alphas))
        print(
            f"\n{'!'*80}\n"
            f"  WARNING: alpha* = {uniform_alpha:.2f} was selected for ALL regimes and all seeds.\n"
            f"  The model is NOT adaptive: it uses a fixed alpha everywhere.\n"
            f"  It is functionally identical to the static alpha={uniform_alpha:.2f} baseline.\n"
            f"  Do NOT describe this as an adaptive or load-aware model in reports.\n"
            f"{'!'*80}"
        )
        adaptive_label = (f"alpha*-Selected Graph-GRU "
                          f"(alpha={uniform_alpha:.2f} uniform, NON-ADAPTIVE)")
    else:
        adaptive_label = "Load-Aware Graph-GRU (alpha*-Selected)"

    # Use matching name as key in comp_models
    ADAPTIVE_KEY = "Similarity-only Graph-GRU (alpha=0.0)"  # same preds if alpha*=0 everywhere

    with V2_SELECTED_ALPHAS_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed","regime","cutoff_label","selected_alpha","val_MAE_Mbps","val_RMSE_Mbps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected_alphas_rows:
            writer.writerow({
                "seed": r["seed"], "regime": r["regime"], "cutoff_label": r["cutoff_label"],
                "selected_alpha": f"{r['selected_alpha']:.2f}",
                "val_MAE_Mbps":   f"{r['val_MAE_Mbps']:.3f}",
                "val_RMSE_Mbps":  f"{r['val_RMSE_Mbps']:.3f}",
            })
    print(f"Saved selected alphas to: {V2_SELECTED_ALPHAS_CSV}")

    # --- Train GRU baseline ---
    gru_test_preds = {}
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gru    = GRUBaseline(len(names)).to(device)
        loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH, shuffle=True)
        gru    = fit(gru, loader, vx.to(device), vy.to(device), None, device)
        gru.eval()
        with torch.no_grad():
            gru_test_preds[seed] = gru(qx.to(device)).cpu().numpy() * std + mean

    comp_models = [
        adaptive_label,
        "Similarity-only Graph-GRU (alpha=0.0)",
        "Fixed 50/50 Mixed Graph-GRU (alpha=0.5)",
        "GRU (Temporal Baseline)",
        "Physical Graph-GRU (alpha=1.0)",
    ]

    per_seed_results     = []
    regimes_summary_data = []

    # RETROSPECTIVE ONLY: test-local tertiles -- not used for model selection
    q33_test_obs = float(np.percentile(test_obs_loads, 100.0 / 3.0))
    q67_test_obs = float(np.percentile(test_obs_loads, 200.0 / 3.0))
    test_local_regimes = {
        "Low [test-local]":    test_obs_loads <= q33_test_obs,
        "Medium [test-local]": (test_obs_loads > q33_test_obs) & (test_obs_loads <= q67_test_obs),
        "High [test-local]":   test_obs_loads > q67_test_obs,
    }

    for seed in SEEDS:
        dynamic_test_pred = np.zeros_like(test_truth)
        for i in range(len(test_truth)):
            obs = test_obs_loads[i]
            if obs <= q33_units:
                reg_name = "Low"
            elif obs <= q67_units:
                reg_name = "Medium"
            else:
                reg_name = "High"
            chosen_alpha = selected_alphas_per_seed[seed][reg_name]
            dynamic_test_pred[i] = trained_candidates[(seed, chosen_alpha)]["test_pred"][i]

        model_preds = {
            adaptive_label:                              dynamic_test_pred,
            "Similarity-only Graph-GRU (alpha=0.0)":    trained_candidates[(seed, 0.0)]["test_pred"],
            "Fixed 50/50 Mixed Graph-GRU (alpha=0.5)":  trained_candidates[(seed, 0.5)]["test_pred"],
            "GRU (Temporal Baseline)":                   gru_test_preds[seed],
            "Physical Graph-GRU (alpha=1.0)":            trained_candidates[(seed, 1.0)]["test_pred"],
        }

        for m_name in comp_models:
            pred   = model_preds[m_name]
            mae_u  = float(np.abs(pred - test_truth).mean())
            rmse_u = float(np.sqrt(np.square(pred - test_truth).mean()))
            per_seed_results.append({
                "model": m_name, "seed": seed,
                "MAE_Mbps":  mae_u  * UNIT_TO_MBPS,
                "RMSE_Mbps": rmse_u * UNIT_TO_MBPS,
                "MAE_units": mae_u,
                "RMSE_units":rmse_u,
            })

            # PRIMARY: training-cutoff regimes
            for r_name, r_mask in test_regimes.items():
                if r_mask.sum() > 0:
                    r_mae_u  = float(np.abs(pred[r_mask] - test_truth[r_mask]).mean())
                    r_rmse_u = float(np.sqrt(np.square(pred[r_mask] - test_truth[r_mask]).mean()))
                    regimes_summary_data.append({
                        "regime_type": "Training Cutoffs (primary)",
                        "regime": r_name, "window_count": int(r_mask.sum()),
                        "model": m_name, "seed": seed,
                        "MAE_Mbps":  r_mae_u  * UNIT_TO_MBPS,
                        "RMSE_Mbps": r_rmse_u * UNIT_TO_MBPS,
                        "MAE_units": r_mae_u,  "RMSE_units": r_rmse_u,
                    })

            # RETROSPECTIVE: test-local tertiles
            for r_name, r_mask in test_local_regimes.items():
                if r_mask.sum() > 0:
                    r_mae_u  = float(np.abs(pred[r_mask] - test_truth[r_mask]).mean())
                    r_rmse_u = float(np.sqrt(np.square(pred[r_mask] - test_truth[r_mask]).mean()))
                    regimes_summary_data.append({
                        "regime_type": "RETROSPECTIVE: Test-Local Tertiles",
                        "regime": r_name, "window_count": int(r_mask.sum()),
                        "model": m_name, "seed": seed,
                        "MAE_Mbps":  r_mae_u  * UNIT_TO_MBPS,
                        "RMSE_Mbps": r_rmse_u * UNIT_TO_MBPS,
                        "MAE_units": r_mae_u,  "RMSE_units": r_rmse_u,
                    })

    with V2_RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model","seed","MAE_Mbps","RMSE_Mbps","MAE_units","RMSE_units"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_seed_results:
            writer.writerow({
                "model": r["model"], "seed": r["seed"],
                "MAE_Mbps":  f"{r['MAE_Mbps']:.3f}",  "RMSE_Mbps":  f"{r['RMSE_Mbps']:.3f}",
                "MAE_units": f"{r['MAE_units']:.3f}",  "RMSE_units": f"{r['RMSE_units']:.3f}",
            })
    print(f"\nSaved per-seed results to: {V2_RESULTS_CSV}")

    # Overall summary
    overall_summary = []
    for m in comp_models:
        matches  = [r for r in per_seed_results if r["model"] == m]
        mae_m    = np.array([r["MAE_Mbps"]   for r in matches], dtype=np.float64)
        rmse_m   = np.array([r["RMSE_Mbps"]  for r in matches], dtype=np.float64)
        mae_u    = np.array([r["MAE_units"]   for r in matches], dtype=np.float64)
        rmse_u   = np.array([r["RMSE_units"]  for r in matches], dtype=np.float64)
        overall_summary.append({
            "model": m, "seed_count": len(matches),
            "MAE_mean_Mbps":  float(mae_m.mean()),  "MAE_std_Mbps":  float(mae_m.std(ddof=1)),
            "RMSE_mean_Mbps": float(rmse_m.mean()), "RMSE_std_Mbps": float(rmse_m.std(ddof=1)),
            "MAE_mean_units": float(mae_u.mean()),   "MAE_std_units":  float(mae_u.std(ddof=1)),
            "RMSE_mean_units":float(rmse_u.mean()),  "RMSE_std_units": float(rmse_u.std(ddof=1)),
        })

    with V2_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model","seed_count","MAE_mean_Mbps","MAE_std_Mbps",
                      "RMSE_mean_Mbps","RMSE_std_Mbps","MAE_mean_units","MAE_std_units",
                      "RMSE_mean_units","RMSE_std_units"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in overall_summary:
            writer.writerow({
                "model": r["model"], "seed_count": r["seed_count"],
                "MAE_mean_Mbps":  f"{r['MAE_mean_Mbps']:.3f}",
                "MAE_std_Mbps":   f"{r['MAE_std_Mbps']:.3f}",
                "RMSE_mean_Mbps": f"{r['RMSE_mean_Mbps']:.3f}",
                "RMSE_std_Mbps":  f"{r['RMSE_std_Mbps']:.3f}",
                "MAE_mean_units": f"{r['MAE_mean_units']:.3f}",
                "MAE_std_units":  f"{r['MAE_std_units']:.3f}",
                "RMSE_mean_units":f"{r['RMSE_mean_units']:.3f}",
                "RMSE_std_units": f"{r['RMSE_std_units']:.3f}",
            })
    print(f"Saved overall summary to: {V2_SUMMARY_CSV}")

    # Regime summary
    regimes_summary = []
    all_reg_tuples = sorted(set((r["regime_type"], r["regime"]) for r in regimes_summary_data))
    for reg_type, reg_name in all_reg_tuples:
        for m in comp_models:
            matches = [r for r in regimes_summary_data
                       if r["regime_type"] == reg_type and r["regime"] == reg_name and r["model"] == m]
            if not matches:
                continue
            mae_m  = np.array([r["MAE_Mbps"]  for r in matches], dtype=np.float64)
            rmse_m = np.array([r["RMSE_Mbps"] for r in matches], dtype=np.float64)
            mae_u  = np.array([r["MAE_units"]  for r in matches], dtype=np.float64)
            rmse_u = np.array([r["RMSE_units"] for r in matches], dtype=np.float64)
            regimes_summary.append({
                "regime_type": reg_type, "regime": reg_name,
                "window_count": matches[0]["window_count"],
                "model": m, "seed_count": len(matches),
                "MAE_mean_Mbps":  float(mae_m.mean()),  "MAE_std_Mbps":  float(mae_m.std(ddof=1)),
                "RMSE_mean_Mbps": float(rmse_m.mean()), "RMSE_std_Mbps": float(rmse_m.std(ddof=1)),
                "MAE_mean_units": float(mae_u.mean()),   "MAE_std_units":  float(mae_u.std(ddof=1)),
                "RMSE_mean_units":float(rmse_u.mean()),  "RMSE_std_units": float(rmse_u.std(ddof=1)),
            })

    with V2_REGIMES_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["regime_type","regime","window_count","model","seed_count",
                      "MAE_mean_Mbps","MAE_std_Mbps","RMSE_mean_Mbps","RMSE_std_Mbps",
                      "MAE_mean_units","MAE_std_units","RMSE_mean_units","RMSE_std_units"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in regimes_summary:
            writer.writerow({
                "regime_type": r["regime_type"], "regime": r["regime"],
                "window_count": r["window_count"], "model": r["model"],
                "seed_count": r["seed_count"],
                "MAE_mean_Mbps":  f"{r['MAE_mean_Mbps']:.3f}",
                "MAE_std_Mbps":   f"{r['MAE_std_Mbps']:.3f}",
                "RMSE_mean_Mbps": f"{r['RMSE_mean_Mbps']:.3f}",
                "RMSE_std_Mbps":  f"{r['RMSE_std_Mbps']:.3f}",
                "MAE_mean_units": f"{r['MAE_mean_units']:.3f}",
                "MAE_std_units":  f"{r['MAE_std_units']:.3f}",
                "RMSE_mean_units":f"{r['RMSE_mean_units']:.3f}",
                "RMSE_std_units": f"{r['RMSE_std_units']:.3f}",
            })
    print(f"Saved regime summary to: {V2_REGIMES_CSV}")
    plot_comparison(overall_summary, V2_COMPARISON_CHART, adaptive_label,
                    len(test_obs_loads), only_regime)

    # --- Console report ---
    print("\n" + "=" * 100)
    print("OVERALL TEST PERFORMANCE  (Mean +/- Std across 3 seeds)")
    print("=" * 100)
    if only_regime:
        print(f"  NOTE: All {len(test_obs_loads)} test windows fall in the training-defined"
              f" '{only_regime}' regime.")
        print("  Medium/High test regime performance COULD NOT be evaluated under this policy.")
    print("=" * 100)
    print(f"  {'Model':<52} | {'MAE (Mbps)':<26} | {'RMSE (Mbps)'}")
    print("-" * 100)
    for r in overall_summary:
        print(f"  {r['model']:<52} | "
              f"{r['MAE_mean_Mbps']:.2f} +/- {r['MAE_std_Mbps']:.2f} Mbps      | "
              f"{r['RMSE_mean_Mbps']:.2f} +/- {r['RMSE_std_Mbps']:.2f} Mbps")
    print("=" * 100)

    primary_rows = [r for r in regimes_summary if r["regime_type"] == "Training Cutoffs (primary)"]
    print("\n" + "=" * 110)
    print("PER-REGIME TEST PERFORMANCE -- PRIMARY (Training-Cutoff Regimes, Mean +/- Std)")
    print("=" * 110)
    for r in primary_rows:
        print(f"  Regime: {r['regime']:<8} | N={r['window_count']:>4} | "
              f"{r['model']:<52} | MAE: {r['MAE_mean_Mbps']:.2f} +/- {r['MAE_std_Mbps']:.2f} Mbps")
    unavailable = [reg for reg, cnt in test_regime_counts.items() if cnt == 0]
    if unavailable:
        print(f"\n  *** NO test windows in regimes: {', '.join(unavailable)} -- cannot be evaluated ***")
    print("=" * 110)

    retro_rows = [r for r in regimes_summary if "RETROSPECTIVE" in r["regime_type"]]
    if retro_rows:
        print("\n" + "=" * 110)
        print("[RETROSPECTIVE -- POST-HOC ONLY] Test-Local Tertile Analysis")
        print("  Cutoffs are computed FROM the test data itself.")
        print("  NOT used for policy selection. NOT a prospective regime evaluation.")
        print("=" * 110)
        for r in retro_rows:
            print(f"  {r['regime']:<24} | N={r['window_count']:>4} | "
                  f"{r['model']:<52} | MAE: {r['MAE_mean_Mbps']:.2f} +/- {r['MAE_std_Mbps']:.2f} Mbps")
        print("=" * 110)


if __name__ == "__main__":
    main()
