"""Run ablation study comparing GRU, Physical Graph-GRU, Similarity Graph-GRU, and Mixed Graph-GRU."""
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

# Separate ablation filenames to strictly preserve existing outputs
ABLATION_OVERALL_RESULTS_CSV = OUTPUTS_DIR / "ablation_model_results.csv"
ABLATION_OVERALL_SUMMARY_CSV = OUTPUTS_DIR / "ablation_model_summary.csv"
ABLATION_SCENARIOS_RESULTS_CSV = OUTPUTS_DIR / "ablation_load_scenarios_results.csv"
ABLATION_SCENARIOS_SUMMARY_CSV = OUTPUTS_DIR / "ablation_load_scenarios_summary.csv"
ABLATION_OVERALL_MAE_CHART = OUTPUTS_DIR / "ablation_mae_comparison.png"
ABLATION_OVERALL_RMSE_CHART = OUTPUTS_DIR / "ablation_rmse_comparison.png"
ABLATION_SCENARIOS_MAE_CHART = OUTPUTS_DIR / "ablation_scenarios_mae.png"
ABLATION_SCENARIOS_RMSE_CHART = OUTPUTS_DIR / "ablation_scenarios_rmse.png"

# Unit conversion constants: 1 Abilene unit = 8/3 bps
UNIT_TO_BPS = 8.0 / 3.0
BPS_TO_MBPS = 1e-6
BPS_TO_GBPS = 1e-9
UNIT_TO_MBPS = UNIT_TO_BPS * BPS_TO_MBPS    # 1 unit = (8/3) * 1e-6 Mbps
UNIT_TO_GBPS = UNIT_TO_BPS * BPS_TO_GBPS    # 1 unit = (8/3) * 1e-9 Gbps

HISTORY = 12                 # 60 minutes at five-minute intervals
SEEDS = (7, 21, 42)
EPOCHS, BATCH, LR = 80, 32, 1e-3


def norm_adj(a):
    a = a.astype(np.float32).copy()
    np.fill_diagonal(a, 1)
    inv = np.power(np.maximum(a.sum(axis=1), 1e-8), -0.5)
    return (inv[:, None] * a * inv[None, :]).astype(np.float32)


def make_graphs_with_ablation(names, train_series):
    """
    Build physical graph, similarity graph (training series only), and mixed graph.
    Returns normalized adjacency tensors for:
      1. physical
      2. similarity (ablation)
      3. mixed (0.5 * physical + 0.5 * similar)
    """
    n = len(names)
    physical = np.eye(n, dtype=np.float32)
    ends = [str(name).split(",") for name in names]
    for i, (a, b) in enumerate(ends):
        for j in range(i + 1, n):
            c, d = ends[j]
            if {a, b} & {c, d}:
                physical[i, j] = physical[j, i] = 1

    # Similarity graph built strictly using training series correlation
    corr = np.nan_to_num(np.corrcoef(train_series, rowvar=False), nan=0.0)
    similar = np.eye(n, dtype=np.float32)
    for i in range(n):
        top = sorted(((corr[i, j], j) for j in range(n) if j != i and corr[i, j] > 0), reverse=True)[:3]
        for value, j in top:
            similar[i, j] = similar[j, i] = max(similar[i, j], value)

    mixed = 0.5 * physical + 0.5 * similar

    return {
        "physical": torch.tensor(norm_adj(physical)),
        "similarity": torch.tensor(norm_adj(similar)),
        "mixed": torch.tensor(norm_adj(mixed)),
    }


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


def plot_overall_bar_chart(models, means, stds, metric_name, out_path, colors):
    """Plot overall 4-model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(models))

    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, edgecolor="#2b2b2b", linewidth=1.2, width=0.55, alpha=0.9,
        error_kw={"elinewidth": 1.5, "ecolor": "#2b2b2b", "capthick": 1.5},
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight="medium")
    ax.set_ylabel(f"{metric_name} (Mbps)", fontsize=12, fontweight="medium")
    ax.set_title(f"Overall Test {metric_name} Ablation Comparison (Mean ± Std in Mbps)", fontsize=13, fontweight="bold", pad=14)
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
    print(f"Saved overall {metric_name} ablation chart to: {out_path}")


def plot_scenarios_ablation_chart(summary_rows, metric_name, out_path, scenarios_info, models, colors):
    """Plot grouped bar chart comparing all 4 models across Low, Medium, and High scenarios."""
    scenarios = ["Low", "Medium", "High"]
    mean_map = {(r["scenario"], r["model"]): r[f"{metric_name}_mean_Mbps"] for r in summary_rows}
    std_map = {(r["scenario"], r["model"]): r[f"{metric_name}_std_Mbps"] for r in summary_rows}

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(scenarios))
    bar_width = 0.18

    max_val = 0.0
    for i, model in enumerate(models):
        means = [mean_map[(s, model)] for s in scenarios]
        stds = [std_map[(s, model)] for s in scenarios]
        offsets = x + (i - 1.5) * bar_width
        bars = ax.bar(
            offsets, means, yerr=stds, width=bar_width,
            label=model, color=colors[i], edgecolor="#2b2b2b",
            linewidth=1.0, alpha=0.9, capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#2b2b2b", "capthick": 1.2},
            zorder=3
        )
        for bar, m, s in zip(bars, means, stds):
            top = m + s
            if top > max_val:
                max_val = top
            ax.annotate(
                f"{m:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, top),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#222222"
            )

    x_labels = [scenarios_info[s]["axis_label"] for s in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="medium")
    ax.set_ylabel(f"{metric_name} (Mbps)", fontsize=12, fontweight="medium")
    ax.set_title(
        f"Test {metric_name} Across Load Scenarios with Graph Ablations (Mean ± Std in Mbps)\n"
        f"[Physical vs. Similarity vs. Mixed vs. GRU Baseline]",
        fontsize=12.5, fontweight="bold", pad=12
    )
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max_val * 1.22)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved scenario {metric_name} ablation chart to: {out_path}")


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

    # Build graphs strictly using training series correlation
    graphs = make_graphs_with_ablation(names, scaled[:a])
    device_graphs = {k: g.to(device) for k, g in graphs.items()}

    # Specs for all four models including the similarity-only ablation
    specs = [
        ("GRU", GRUBaseline, None),
        ("Physical Graph-GRU", GraphGRU, device_graphs["physical"]),
        ("Similarity Graph-GRU", GraphGRU, device_graphs["similarity"]),
        ("Proposed Mixed Graph-GRU", GraphGRU, device_graphs["mixed"]),
    ]

    # Ground truth for test set (in dataset units)
    truth = qy.numpy() * std + mean
    scenarios, q33_gbps, q67_gbps = assign_load_scenarios(truth)

    print("\n--- Ablation Experiment Setup ---")
    print("Models under evaluation:")
    print("  1. GRU (temporal baseline, no graph)")
    print("  2. Physical Graph-GRU (physical-topology graph only)")
    print("  3. Similarity Graph-GRU (similarity-correlation graph only [ablation])")
    print("  4. Proposed Mixed Graph-GRU (0.5 * physical + 0.5 * similarity)")
    print(f"Seeds: {SEEDS}")
    print(f"Test load cutoffs: Low <= {q33_gbps:.2f} Gbps | Med ({q33_gbps:.2f}, {q67_gbps:.2f}] Gbps | High > {q67_gbps:.2f} Gbps")

    overall_per_seed = []
    scenarios_per_seed = []

    for label, model_class, adj in specs:
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = model_class(len(names)).to(device)
            loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH, shuffle=True)

            model = fit(model, loader, vx.to(device), vy.to(device), adj, device)
            model.eval()

            with torch.no_grad():
                pred = model(qx.to(device)) if adj is None else model(qx.to(device), adj)
            pred = pred.cpu().numpy() * std + mean

            # Overall metrics
            overall_mae_units = float(np.abs(pred - truth).mean())
            overall_rmse_units = float(np.sqrt(np.square(pred - truth).mean()))
            overall_mae_mbps = overall_mae_units * UNIT_TO_MBPS
            overall_rmse_mbps = overall_rmse_units * UNIT_TO_MBPS

            overall_per_seed.append({
                "model": label,
                "seed": seed,
                "MAE_Mbps": overall_mae_mbps,
                "RMSE_Mbps": overall_rmse_mbps,
                "MAE_units": overall_mae_units,
                "RMSE_units": overall_rmse_units,
            })
            print(f"[Overall] {label}, seed {seed}: MAE={overall_mae_mbps:.2f} Mbps ({overall_mae_units:.0f} units), RMSE={overall_rmse_mbps:.2f} Mbps ({overall_rmse_units:.0f} units)")

            # Scenario metrics
            for s_name, s_info in scenarios.items():
                mask = s_info["mask"]
                pred_b, truth_b = pred[mask], truth[mask]
                s_mae_units = float(np.abs(pred_b - truth_b).mean())
                s_rmse_units = float(np.sqrt(np.square(pred_b - truth_b).mean()))
                s_mae_mbps = s_mae_units * UNIT_TO_MBPS
                s_rmse_mbps = s_rmse_units * UNIT_TO_MBPS

                scenarios_per_seed.append({
                    "scenario": s_name,
                    "threshold_label": s_info["cutoff_label"],
                    "model": label,
                    "seed": seed,
                    "window_count": s_info["count"],
                    "MAE_Mbps": s_mae_mbps,
                    "RMSE_Mbps": s_rmse_mbps,
                    "MAE_units": s_mae_units,
                    "RMSE_units": s_rmse_units,
                })

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save overall per-seed CSV
    with ABLATION_OVERALL_RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "seed", "MAE_Mbps", "RMSE_Mbps", "MAE_units", "RMSE_units"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in overall_per_seed:
            writer.writerow({
                "model": r["model"],
                "seed": r["seed"],
                "MAE_Mbps": f"{r['MAE_Mbps']:.3f}",
                "RMSE_Mbps": f"{r['RMSE_Mbps']:.3f}",
                "MAE_units": f"{r['MAE_units']:.3f}",
                "RMSE_units": f"{r['RMSE_units']:.3f}",
            })
    print(f"\nSaved overall ablation results to: {ABLATION_OVERALL_RESULTS_CSV}")

    # 2. Compute and save overall summary CSV
    overall_summary = []
    model_names = [s[0] for s in specs]
    for m in model_names:
        matches = [r for r in overall_per_seed if r["model"] == m]
        mae_mbps = np.array([r["MAE_Mbps"] for r in matches], dtype=np.float64)
        rmse_mbps = np.array([r["RMSE_Mbps"] for r in matches], dtype=np.float64)
        mae_units = np.array([r["MAE_units"] for r in matches], dtype=np.float64)
        rmse_units = np.array([r["RMSE_units"] for r in matches], dtype=np.float64)

        overall_summary.append({
            "model": m,
            "seed_count": len(matches),
            "MAE_mean_Mbps": float(mae_mbps.mean()),
            "MAE_std_Mbps": float(mae_mbps.std(ddof=1)),
            "RMSE_mean_Mbps": float(rmse_mbps.mean()),
            "RMSE_std_Mbps": float(rmse_mbps.std(ddof=1)),
            "MAE_mean_units": float(mae_units.mean()),
            "MAE_std_units": float(mae_units.std(ddof=1)),
            "RMSE_mean_units": float(rmse_units.mean()),
            "RMSE_std_units": float(rmse_units.std(ddof=1)),
        })

    with ABLATION_OVERALL_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
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
    print(f"Saved overall ablation summary to: {ABLATION_OVERALL_SUMMARY_CSV}")

    # 3. Save scenarios per-seed CSV
    with ABLATION_SCENARIOS_RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario", "threshold_label", "model", "seed", "window_count",
            "MAE_Mbps", "RMSE_Mbps", "MAE_units", "RMSE_units"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scenarios_per_seed:
            writer.writerow({
                "scenario": r["scenario"],
                "threshold_label": r["threshold_label"],
                "model": r["model"],
                "seed": r["seed"],
                "window_count": r["window_count"],
                "MAE_Mbps": f"{r['MAE_Mbps']:.3f}",
                "RMSE_Mbps": f"{r['RMSE_Mbps']:.3f}",
                "MAE_units": f"{r['MAE_units']:.3f}",
                "RMSE_units": f"{r['RMSE_units']:.3f}",
            })
    print(f"Saved scenario ablation results to: {ABLATION_SCENARIOS_RESULTS_CSV}")

    # 4. Compute and save scenarios summary CSV
    scenarios_summary = []
    for s_name in ["Low", "Medium", "High"]:
        for m_name in model_names:
            matches = [r for r in scenarios_per_seed if r["scenario"] == s_name and r["model"] == m_name]
            mae_mbps = np.array([r["MAE_Mbps"] for r in matches], dtype=np.float64)
            rmse_mbps = np.array([r["RMSE_Mbps"] for r in matches], dtype=np.float64)
            mae_units = np.array([r["MAE_units"] for r in matches], dtype=np.float64)
            rmse_units = np.array([r["RMSE_units"] for r in matches], dtype=np.float64)

            scenarios_summary.append({
                "scenario": s_name,
                "total_load_threshold_Gbps": matches[0]["threshold_label"],
                "model": m_name,
                "seed_count": len(matches),
                "window_count": matches[0]["window_count"],
                "MAE_mean_Mbps": float(mae_mbps.mean()),
                "MAE_std_Mbps": float(mae_mbps.std(ddof=1)),
                "RMSE_mean_Mbps": float(rmse_mbps.mean()),
                "RMSE_std_Mbps": float(rmse_mbps.std(ddof=1)),
                "MAE_mean_units": float(mae_units.mean()),
                "MAE_std_units": float(mae_units.std(ddof=1)),
                "RMSE_mean_units": float(rmse_units.mean()),
                "RMSE_std_units": float(rmse_units.std(ddof=1)),
            })

    with ABLATION_SCENARIOS_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario", "total_load_threshold_Gbps", "model", "seed_count", "window_count",
            "MAE_mean_Mbps", "MAE_std_Mbps", "RMSE_mean_Mbps", "RMSE_std_Mbps",
            "MAE_mean_units", "MAE_std_units", "RMSE_mean_units", "RMSE_std_units"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scenarios_summary:
            writer.writerow({
                "scenario": r["scenario"],
                "total_load_threshold_Gbps": r["total_load_threshold_Gbps"],
                "model": r["model"],
                "seed_count": r["seed_count"],
                "window_count": r["window_count"],
                "MAE_mean_Mbps": f"{r['MAE_mean_Mbps']:.3f}",
                "MAE_std_Mbps": f"{r['MAE_std_Mbps']:.3f}",
                "RMSE_mean_Mbps": f"{r['RMSE_mean_Mbps']:.3f}",
                "RMSE_std_Mbps": f"{r['RMSE_std_Mbps']:.3f}",
                "MAE_mean_units": f"{r['MAE_mean_units']:.3f}",
                "MAE_std_units": f"{r['MAE_std_units']:.3f}",
                "RMSE_mean_units": f"{r['RMSE_mean_units']:.3f}",
                "RMSE_std_units": f"{r['RMSE_std_units']:.3f}",
            })
    print(f"Saved scenario ablation summary to: {ABLATION_SCENARIOS_SUMMARY_CSV}")

    # 5. Visualizations
    palette = ["#4C72B0", "#DD8452", "#9370DB", "#55A868"]  # Blue, Orange, Purple, Green
    plot_overall_bar_chart(
        model_names,
        [r["MAE_mean_Mbps"] for r in overall_summary],
        [r["MAE_std_Mbps"] for r in overall_summary],
        "MAE", ABLATION_OVERALL_MAE_CHART, palette
    )
    plot_overall_bar_chart(
        model_names,
        [r["RMSE_mean_Mbps"] for r in overall_summary],
        [r["RMSE_std_Mbps"] for r in overall_summary],
        "RMSE", ABLATION_OVERALL_RMSE_CHART, palette
    )
    plot_scenarios_ablation_chart(
        scenarios_summary, "MAE", ABLATION_SCENARIOS_MAE_CHART, scenarios, model_names, palette
    )
    plot_scenarios_ablation_chart(
        scenarios_summary, "RMSE", ABLATION_SCENARIOS_RMSE_CHART, scenarios, model_names, palette
    )

    # 6. Display formatted tables to stdout
    print("\n" + "=" * 92)
    print("OVERALL ABLATION SUMMARY (Mean ± Std across 3 seeds)")
    print("=" * 92)
    print(f"{'Model':<26} | {'MAE (Mbps)':<22} | {'RMSE (Mbps)':<22} | {'MAE (Units)':<14}")
    print("-" * 92)
    for r in overall_summary:
        mae_str = f"{r['MAE_mean_Mbps']:.2f} ± {r['MAE_std_Mbps']:.2f} Mbps"
        rmse_str = f"{r['RMSE_mean_Mbps']:.2f} ± {r['RMSE_std_Mbps']:.2f} Mbps"
        print(f"{r['model']:<26} | {mae_str:<22} | {rmse_str:<22} | {r['MAE_mean_units']:,.0f}")
    print("=" * 92)

    print("\n" + "=" * 102)
    print("LOAD SCENARIO ABLATION SUMMARY (Mean ± Std across 3 seeds)")
    print("=" * 102)
    print(f"{'Scenario':<8} | {'Load Threshold':<16} | {'Model':<26} | {'MAE (Mean ± Std, Mbps)':<22} | {'RMSE (Mean ± Std, Mbps)':<22}")
    print("-" * 102)
    for r in scenarios_summary:
        mae_str = f"{r['MAE_mean_Mbps']:.2f} ± {r['MAE_std_Mbps']:.2f} Mbps"
        rmse_str = f"{r['RMSE_mean_Mbps']:.2f} ± {r['RMSE_std_Mbps']:.2f} Mbps"
        print(f"{r['scenario']:<8} | {r['total_load_threshold_Gbps']:<16} | {r['model']:<26} | {mae_str:<22} | {rmse_str:<22}")
    print("=" * 102)


if __name__ == "__main__":
    main()
