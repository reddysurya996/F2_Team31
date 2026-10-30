"""Evaluate physical-graph mixing weight alpha in [0, 0.25, 0.5, 0.75, 1.0].

Selects alpha per seed using validation MAE only (never test data),
then evaluates the selected model on the test set and compares with baselines.
"""
from pathlib import Path
import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Ensure models can be imported whether run from root or src/
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import GRUBaseline, GraphGRU

DATA_PATH = ROOT / "data/processed/abilene_link_loads.npz"
OUTPUTS_DIR = ROOT / "outputs"

# New separate filenames to preserve existing results
ALPHA_GRID_CSV = OUTPUTS_DIR / "alpha_grid_validation_results.csv"
ALPHA_SELECTED_RESULTS_CSV = OUTPUTS_DIR / "alpha_selected_model_results.csv"
ALPHA_SELECTED_SUMMARY_CSV = OUTPUTS_DIR / "alpha_selected_model_summary.csv"
ALPHA_SELECTED_SCENARIOS_CSV = OUTPUTS_DIR / "alpha_selected_scenarios_summary.csv"
ALPHA_VAL_CURVE_CHART = OUTPUTS_DIR / "alpha_tuning_validation_curve.png"
ALPHA_COMPARISON_CHART = OUTPUTS_DIR / "alpha_selected_comparison.png"
ALPHA_SCENARIOS_CHART = OUTPUTS_DIR / "alpha_selected_scenarios.png"

# Unit conversion constants: 1 Abilene unit = 8/3 bps
UNIT_TO_BPS = 8.0 / 3.0
BPS_TO_MBPS = 1e-6
BPS_TO_GBPS = 1e-9
UNIT_TO_MBPS = UNIT_TO_BPS * BPS_TO_MBPS    # 1 unit = (8/3) * 1e-6 Mbps
UNIT_TO_GBPS = UNIT_TO_BPS * BPS_TO_GBPS    # 1 unit = (8/3) * 1e-9 Gbps

HISTORY = 12                 # 60 minutes at five-minute intervals
SEEDS = (7, 21, 42)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPOCHS, BATCH, LR = 80, 32, 1e-3


def norm_adj(a):
    a = a.astype(np.float32).copy()
    np.fill_diagonal(a, 1)
    inv = np.power(np.maximum(a.sum(axis=1), 1e-8), -0.5)
    return (inv[:, None] * a * inv[None, :]).astype(np.float32)


def build_raw_graphs(names, train_series):
    """Build unnormalized physical and similarity adjacency matrices from training data only."""
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
    return torch.tensor(np.asarray(x), dtype=torch.float32), torch.tensor(np.asarray(y), dtype=torch.float32)


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


def assign_load_scenarios(test_truth):
    """Tertile partition of test forecast windows based on total internal link traffic."""
    total_link_load_units = test_truth.sum(axis=1)
    q33_units = float(np.percentile(total_link_load_units, 100.0 / 3.0))
    q67_units = float(np.percentile(total_link_load_units, 200.0 / 3.0))

    q33_gbps = q33_units * UNIT_TO_GBPS
    q67_gbps = q67_units * UNIT_TO_GBPS

    low_mask = total_link_load_units <= q33_units
    med_mask = (total_link_load_units > q33_units) & (total_link_load_units <= q67_units)
    high_mask = total_link_load_units > q67_units

    scenarios = {
        "Low": {
            "mask": low_mask,
            "count": int(low_mask.sum()),
            "cutoff_label": f"<= {q33_gbps:.2f} Gbps",
            "axis_label": f"Low Load\n(<= {q33_gbps:.2f} Gbps)"
        },
        "Medium": {
            "mask": med_mask,
            "count": int(med_mask.sum()),
            "cutoff_label": f"({q33_gbps:.2f}, {q67_gbps:.2f}] Gbps",
            "axis_label": f"Medium Load\n({q33_gbps:.2f} - {q67_gbps:.2f} Gbps)"
        },
        "High": {
            "mask": high_mask,
            "count": int(high_mask.sum()),
            "cutoff_label": f"> {q67_gbps:.2f} Gbps",
            "axis_label": f"High Load\n(> {q67_gbps:.2f} Gbps)"
        }
    }
    return scenarios, q33_gbps, q67_gbps


def plot_validation_curves(grid_results, selected_alphas, out_path):
    """Plot Validation MAE vs Alpha curves per seed and highlight selected optimal alpha."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    seed_colors = {7: "#4C72B0", 21: "#DD8452", 42: "#55A868"}

    for seed in SEEDS:
        seed_data = [r for r in grid_results if r["seed"] == seed]
        seed_data.sort(key=lambda x: x["alpha"])
        alphas = [r["alpha"] for r in seed_data]
        val_maes = [r["val_MAE_Mbps"] for r in seed_data]

        best_alpha = selected_alphas[seed]["alpha"]
        best_mae = selected_alphas[seed]["val_MAE_Mbps"]

        ax.plot(alphas, val_maes, marker="o", linewidth=2.0, markersize=7,
                color=seed_colors[seed], label=f"Seed {seed} (Selected α = {best_alpha})")
        # Highlight best point
        ax.scatter([best_alpha], [best_mae], s=140, facecolors="none",
                   edgecolors=seed_colors[seed], linewidth=2.5, zorder=5)

    ax.set_xlabel("Physical Graph Weight α  [A = α * Physical + (1 - α) * Similarity]", fontsize=11.5, fontweight="medium")
    ax.set_ylabel("Validation MAE (Mbps)", fontsize=11.5, fontweight="medium")
    ax.set_title("Validation MAE vs. Mixing Weight α Across Seeds\n(Circles mark validation-selected α* per seed)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels([f"{a:.2f}\n" + ("(Sim Only)" if a == 0 else "(Phys Only)" if a == 1 else "") for a in ALPHAS], fontsize=10)
    ax.legend(frameon=True, fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved validation tuning curve to: {out_path}")


def plot_comparison_chart(summary_rows, metric_name, out_path, colors):
    """Bar chart comparing GRU, Physical (a=1), Similarity (a=0), Fixed Mixed (a=0.5), and Selected Alpha."""
    display_names = [
        "GRU\n(Baseline)",
        "Physical\n(α=1.0)",
        "Similarity\n(α=0.0)",
        "Fixed Mixed\n(α=0.5)",
        "Val-Selected α*\n(Selected α*=0.0)",
    ]
    means = [r[f"{metric_name}_mean_Mbps"] for r in summary_rows]
    stds = [r[f"{metric_name}_std_Mbps"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    x = np.arange(len(display_names))

    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, edgecolor="#2b2b2b", linewidth=1.2, width=0.55, alpha=0.9,
        error_kw={"elinewidth": 1.5, "ecolor": "#2b2b2b", "capthick": 1.5},
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10, fontweight="medium")
    ax.set_ylabel(f"{metric_name} (Mbps)", fontsize=12, fontweight="medium")
    ax.set_title(f"Test {metric_name} Comparison with Validation-Selected α* (Mean ± Std in Mbps)", fontsize=12.5, fontweight="bold", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    max_val = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, max_val * 1.20)

    for bar, m, s in zip(bars, means, stds):
        top = m + s
        ax.annotate(
            f"{m:.2f}\n± {s:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, top),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#222222"
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {metric_name} comparison chart to: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data and establish splits strictly matching train.py
    package = np.load(DATA_PATH)
    raw, names = package["link_loads_internal_30"].astype(np.float32), package["internal_link_names"]

    a, b = int(0.6 * len(raw)), int(0.8 * len(raw))
    mean, std = raw[:a].mean(0), raw[:a].std(0)
    std[std < 1e-6] = 1
    scaled = (raw - mean) / std

    tx, ty = windows(scaled[:a])
    vx, vy = windows(scaled[a:b])
    qx, qy = windows(scaled[b:])

    # Unscaled ground truths
    val_truth = vy.numpy() * std + mean
    test_truth = qy.numpy() * std + mean
    scenarios, q33_gbps, q67_gbps = assign_load_scenarios(test_truth)

    # Build unnormalized graphs strictly from training series
    phys_raw, sim_raw = build_raw_graphs(names, scaled[:a])

    # Pre-build normalized adjacency tensors for each alpha
    alpha_graphs = {}
    for alpha in ALPHAS:
        adj_dense = alpha * phys_raw + (1.0 - alpha) * sim_raw
        alpha_graphs[alpha] = torch.tensor(norm_adj(adj_dense)).to(device)

    print("\n--- Physical Weight (Alpha) Grid Experiment ---")
    print(f"Alphas tested: {ALPHAS}")
    print(f"Seeds: {SEEDS}")
    print("Rule: Alpha is selected per seed strictly using Validation MAE.")

    # 2. Run alpha grid search across seeds
    grid_rows = []
    trained_models = {}  # (seed, alpha) -> (model, val_mae_mbps, test_pred)

    for seed in SEEDS:
        print(f"\nEvaluating Seed {seed}:")
        for alpha in ALPHAS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = GraphGRU(len(names)).to(device)
            loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH, shuffle=True)
            adj = alpha_graphs[alpha]

            model = fit(model, loader, vx.to(device), vy.to(device), adj, device)
            model.eval()

            with torch.no_grad():
                val_pred = model(vx.to(device), adj).cpu().numpy() * std + mean
                test_pred = model(qx.to(device), adj).cpu().numpy() * std + mean

            # Validation metrics (USED FOR SELECTION)
            val_mae_units = float(np.abs(val_pred - val_truth).mean())
            val_rmse_units = float(np.sqrt(np.square(val_pred - val_truth).mean()))
            val_mae_mbps = val_mae_units * UNIT_TO_MBPS
            val_rmse_mbps = val_rmse_units * UNIT_TO_MBPS

            # Test metrics (EVALUATION ONLY)
            test_mae_units = float(np.abs(test_pred - test_truth).mean())
            test_rmse_units = float(np.sqrt(np.square(test_pred - test_truth).mean()))
            test_mae_mbps = test_mae_units * UNIT_TO_MBPS
            test_rmse_mbps = test_rmse_units * UNIT_TO_MBPS

            grid_rows.append({
                "seed": seed,
                "alpha": alpha,
                "val_MAE_Mbps": val_mae_mbps,
                "val_RMSE_Mbps": val_rmse_mbps,
                "val_MAE_units": val_mae_units,
                "val_RMSE_units": val_rmse_units,
                "test_MAE_Mbps": test_mae_mbps,
                "test_RMSE_Mbps": test_rmse_mbps,
                "test_MAE_units": test_mae_units,
                "test_RMSE_units": test_rmse_units,
            })
            trained_models[(seed, alpha)] = {
                "val_mae_mbps": val_mae_mbps,
                "test_pred": test_pred,
            }
            print(f"  alpha={alpha:4.2f} -> Val MAE: {val_mae_mbps:5.2f} Mbps, Test MAE: {test_mae_mbps:5.2f} Mbps")

    # 3. Select alpha per seed using Validation MAE ONLY
    selected_alphas = {}
    print("\n--- Validation Selection Results (Per Seed) ---")
    for seed in SEEDS:
        seed_entries = [r for r in grid_rows if r["seed"] == seed]
        best_entry = min(seed_entries, key=lambda x: x["val_MAE_Mbps"])
        selected_alphas[seed] = best_entry
        print(f"  Seed {seed}: Selected α* = {best_entry['alpha']:.2f} (Val MAE = {best_entry['val_MAE_Mbps']:.2f} Mbps, Test MAE = {best_entry['test_MAE_Mbps']:.2f} Mbps)")

    # 4. Save grid validation results CSV
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with ALPHA_GRID_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seed", "alpha", "val_MAE_Mbps", "val_RMSE_Mbps",
            "test_MAE_Mbps", "test_RMSE_Mbps",
            "val_MAE_units", "val_RMSE_units",
            "test_MAE_units", "test_RMSE_units"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in grid_rows:
            writer.writerow({
                "seed": r["seed"],
                "alpha": f"{r['alpha']:.2f}",
                "val_MAE_Mbps": f"{r['val_MAE_Mbps']:.3f}",
                "val_RMSE_Mbps": f"{r['val_RMSE_Mbps']:.3f}",
                "test_MAE_Mbps": f"{r['test_MAE_Mbps']:.3f}",
                "test_RMSE_Mbps": f"{r['test_RMSE_Mbps']:.3f}",
                "val_MAE_units": f"{r['val_MAE_units']:.3f}",
                "val_RMSE_units": f"{r['val_RMSE_units']:.3f}",
                "test_MAE_units": f"{r['test_MAE_units']:.3f}",
                "test_RMSE_units": f"{r['test_RMSE_units']:.3f}",
            })
    print(f"\nSaved full alpha grid results to: {ALPHA_GRID_CSV}")

    # Plot validation curves
    plot_validation_curves(grid_rows, selected_alphas, ALPHA_VAL_CURVE_CHART)

    # 5. Evaluate GRU baseline on test set to compare with selected alpha
    gru_test_preds = {}
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gru = GRUBaseline(len(names)).to(device)
        loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH, shuffle=True)
        gru = fit(gru, loader, vx.to(device), vy.to(device), None, device)
        gru.eval()
        with torch.no_grad():
            gru_test_preds[seed] = gru(qx.to(device)).cpu().numpy() * std + mean

    # 6. Build per-seed results for 5 comparison models:
    #    - GRU
    #    - Physical Graph-GRU (alpha = 1.0)
    #    - Similarity Graph-GRU (alpha = 0.0)
    #    - Fixed Mixed Graph-GRU (alpha = 0.5)
    #    - Selected Mixed Graph-GRU (alpha = alpha*(seed))
    comp_models = [
        "GRU",
        "Physical Graph-GRU (α=1.0)",
        "Similarity Graph-GRU (α=0.0)",
        "Fixed Mixed Graph-GRU (α=0.5)",
        "Selected Mixed Graph-GRU (Val-Selected α*)",
    ]

    selected_results_rows = []
    scenarios_results_rows = []

    for seed in SEEDS:
        best_alpha = selected_alphas[seed]["alpha"]
        model_preds = {
            "GRU": gru_test_preds[seed],
            "Physical Graph-GRU (α=1.0)": trained_models[(seed, 1.0)]["test_pred"],
            "Similarity Graph-GRU (α=0.0)": trained_models[(seed, 0.0)]["test_pred"],
            "Fixed Mixed Graph-GRU (α=0.5)": trained_models[(seed, 0.5)]["test_pred"],
            "Selected Mixed Graph-GRU (Val-Selected α*)": trained_models[(seed, best_alpha)]["test_pred"],
        }

        for m_name in comp_models:
            pred = model_preds[m_name]
            mae_units = float(np.abs(pred - test_truth).mean())
            rmse_units = float(np.sqrt(np.square(pred - test_truth).mean()))
            mae_mbps = mae_units * UNIT_TO_MBPS
            rmse_mbps = rmse_units * UNIT_TO_MBPS

            selected_results_rows.append({
                "model": m_name,
                "seed": seed,
                "selected_alpha": f"{best_alpha:.2f}" if "Val-Selected" in m_name else ("N/A" if m_name == "GRU" else ("1.00" if "1.0" in m_name else ("0.00" if "0.0" in m_name else "0.50"))),
                "MAE_Mbps": mae_mbps,
                "RMSE_Mbps": rmse_mbps,
                "MAE_units": mae_units,
                "RMSE_units": rmse_units,
            })

            # Scenario breakdown
            for s_name, s_info in scenarios.items():
                mask = s_info["mask"]
                s_mae_u = float(np.abs(pred[mask] - test_truth[mask]).mean())
                s_rmse_u = float(np.sqrt(np.square(pred[mask] - test_truth[mask]).mean()))
                scenarios_results_rows.append({
                    "scenario": s_name,
                    "threshold_label": s_info["cutoff_label"],
                    "model": m_name,
                    "seed": seed,
                    "MAE_Mbps": s_mae_u * UNIT_TO_MBPS,
                    "RMSE_Mbps": s_rmse_u * UNIT_TO_MBPS,
                    "MAE_units": s_mae_u,
                    "RMSE_units": s_rmse_u,
                })

    # Save selected per-seed results CSV
    with ALPHA_SELECTED_RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "seed", "selected_alpha", "MAE_Mbps", "RMSE_Mbps", "MAE_units", "RMSE_units"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected_results_rows:
            writer.writerow({
                "model": r["model"],
                "seed": r["seed"],
                "selected_alpha": r["selected_alpha"],
                "MAE_Mbps": f"{r['MAE_Mbps']:.3f}",
                "RMSE_Mbps": f"{r['RMSE_Mbps']:.3f}",
                "MAE_units": f"{r['MAE_units']:.3f}",
                "RMSE_units": f"{r['RMSE_units']:.3f}",
            })
    print(f"Saved selected per-seed results to: {ALPHA_SELECTED_RESULTS_CSV}")

    # 7. Compute overall summary table
    overall_summary = []
    for m in comp_models:
        matches = [r for r in selected_results_rows if r["model"] == m]
        mae_m = np.array([r["MAE_Mbps"] for r in matches], dtype=np.float64)
        rmse_m = np.array([r["RMSE_Mbps"] for r in matches], dtype=np.float64)
        mae_u = np.array([r["MAE_units"] for r in matches], dtype=np.float64)
        rmse_u = np.array([r["RMSE_units"] for r in matches], dtype=np.float64)

        overall_summary.append({
            "model": m,
            "seed_count": len(matches),
            "MAE_mean_Mbps": float(mae_m.mean()),
            "MAE_std_Mbps": float(mae_m.std(ddof=1)),
            "RMSE_mean_Mbps": float(rmse_m.mean()),
            "RMSE_std_Mbps": float(rmse_m.std(ddof=1)),
            "MAE_mean_units": float(mae_u.mean()),
            "MAE_std_units": float(mae_u.std(ddof=1)),
            "RMSE_mean_units": float(rmse_u.mean()),
            "RMSE_std_units": float(rmse_u.std(ddof=1)),
        })

    with ALPHA_SELECTED_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "seed_count", "MAE_mean_Mbps", "MAE_std_Mbps",
            "RMSE_mean_Mbps", "RMSE_std_Mbps", "MAE_mean_units", "MAE_std_units",
            "RMSE_mean_units", "RMSE_std_units"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in overall_summary:
            writer.writerow({
                "model": r["model"],
                "seed_count": r["seed_count"],
                "MAE_mean_Mbps": f"{r['MAE_mean_Mbps']:.3f}",
                "MAE_std_Mbps": f"{r['MAE_std_Mbps']:.3f}",
                "RMSE_mean_Mbps": f"{r['RMSE_mean_Mbps']:.3f}",
                "RMSE_std_Mbps": f"{r['RMSE_std_Mbps']:.3f}",
                "MAE_mean_units": f"{r['MAE_mean_units']:.3f}",
                "MAE_std_units": f"{r['MAE_std_units']:.3f}",
                "RMSE_mean_units": f"{r['RMSE_mean_units']:.3f}",
                "RMSE_std_units": f"{r['RMSE_std_units']:.3f}",
            })
    print(f"Saved selected overall summary to: {ALPHA_SELECTED_SUMMARY_CSV}")

    # 8. Compute scenario summary table
    scenario_summary = []
    for s_name in ["Low", "Medium", "High"]:
        for m in comp_models:
            matches = [r for r in scenarios_results_rows if r["scenario"] == s_name and r["model"] == m]
            mae_m = np.array([r["MAE_Mbps"] for r in matches], dtype=np.float64)
            rmse_m = np.array([r["RMSE_Mbps"] for r in matches], dtype=np.float64)
            mae_u = np.array([r["MAE_units"] for r in matches], dtype=np.float64)
            rmse_u = np.array([r["RMSE_units"] for r in matches], dtype=np.float64)

            scenario_summary.append({
                "scenario": s_name,
                "total_load_threshold_Gbps": matches[0]["threshold_label"],
                "model": m,
                "seed_count": len(matches),
                "MAE_mean_Mbps": float(mae_m.mean()),
                "MAE_std_Mbps": float(mae_m.std(ddof=1)),
                "RMSE_mean_Mbps": float(rmse_m.mean()),
                "RMSE_std_Mbps": float(rmse_m.std(ddof=1)),
                "MAE_mean_units": float(mae_u.mean()),
                "MAE_std_units": float(mae_u.std(ddof=1)),
                "RMSE_mean_units": float(rmse_u.mean()),
                "RMSE_std_units": float(rmse_u.std(ddof=1)),
            })

    with ALPHA_SELECTED_SCENARIOS_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario", "total_load_threshold_Gbps", "model", "seed_count",
            "MAE_mean_Mbps", "MAE_std_Mbps", "RMSE_mean_Mbps", "RMSE_std_Mbps",
            "MAE_mean_units", "MAE_std_units", "RMSE_mean_units", "RMSE_std_units"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scenario_summary:
            writer.writerow({
                "scenario": r["scenario"],
                "total_load_threshold_Gbps": r["total_load_threshold_Gbps"],
                "model": r["model"],
                "seed_count": r["seed_count"],
                "MAE_mean_Mbps": f"{r['MAE_mean_Mbps']:.3f}",
                "MAE_std_Mbps": f"{r['MAE_std_Mbps']:.3f}",
                "RMSE_mean_Mbps": f"{r['RMSE_mean_Mbps']:.3f}",
                "RMSE_std_Mbps": f"{r['RMSE_std_Mbps']:.3f}",
                "MAE_mean_units": f"{r['MAE_mean_units']:.3f}",
                "MAE_std_units": f"{r['MAE_std_units']:.3f}",
                "RMSE_mean_units": f"{r['RMSE_mean_units']:.3f}",
                "RMSE_std_units": f"{r['RMSE_std_units']:.3f}",
            })
    print(f"Saved selected scenarios summary to: {ALPHA_SELECTED_SCENARIOS_CSV}")

    # 9. Plot charts
    palette = ["#4C72B0", "#DD8452", "#9370DB", "#55A868", "#C44E52"]
    plot_comparison_chart(overall_summary, "MAE", ALPHA_COMPARISON_CHART, palette)

    # 10. Display formatted tables to stdout
    print("\n" + "=" * 105)
    print("OVERALL PERFORMANCE SUMMARY (Validation-Selected α vs. Baselines)")
    print("=" * 105)
    print(f"{'Model':<44} | {'MAE (Mean ± Std, Mbps)':<24} | {'RMSE (Mean ± Std, Mbps)':<24}")
    print("-" * 105)
    for r in overall_summary:
        mae_str = f"{r['MAE_mean_Mbps']:.2f} ± {r['MAE_std_Mbps']:.2f} Mbps"
        rmse_str = f"{r['RMSE_mean_Mbps']:.2f} ± {r['RMSE_std_Mbps']:.2f} Mbps"
        print(f"{r['model']:<44} | {mae_str:<24} | {rmse_str:<24}")
    print("=" * 105)

    print("\n" + "=" * 115)
    print("LOAD SCENARIOS SUMMARY (Validation-Selected α vs. Baselines)")
    print("=" * 115)
    print(f"{'Scenario':<8} | {'Load Threshold':<16} | {'Model':<42} | {'MAE (Mbps)':<20} | {'RMSE (Mbps)':<20}")
    print("-" * 115)
    for r in scenario_summary:
        mae_str = f"{r['MAE_mean_Mbps']:.2f} ± {r['MAE_std_Mbps']:.2f} Mbps"
        rmse_str = f"{r['RMSE_mean_Mbps']:.2f} ± {r['RMSE_std_Mbps']:.2f} Mbps"
        print(f"{r['scenario']:<8} | {r['total_load_threshold_Gbps']:<16} | {r['model']:<42} | {mae_str:<20} | {rmse_str:<20}")
    print("=" * 115)


if __name__ == "__main__":
    main()
