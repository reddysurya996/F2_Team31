"""Evaluate model performance across low, medium, and high network load scenarios with unit conversions."""
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
PER_SEED_CSV = OUTPUTS_DIR / "load_scenarios_results.csv"
SUMMARY_CSV = OUTPUTS_DIR / "load_scenarios_summary.csv"
MAE_CHART = OUTPUTS_DIR / "load_scenarios_mae.png"
RMSE_CHART = OUTPUTS_DIR / "load_scenarios_rmse.png"

# Unit conversion constants:
# 1 Abilene dataset unit = 8/3 bps (100 bytes / 300s = 800/300 bps)
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


def make_graphs(names, train_series):
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
    return [torch.tensor(norm_adj(a)) for a in (physical, 0.5 * physical + 0.5 * similar)]


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
    """
    Assign test forecast windows into lower, middle, and upper thirds
    based on the total actual traffic load across all 30 internal links.
    Training and evaluation remain in dataset units; Gbps thresholds are computed for labeling.
    """
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
            "range_units": (float(total_link_load_units[low_mask].min()), float(total_link_load_units[low_mask].max())),
            "range_gbps": (float(total_link_load_units[low_mask].min()) * UNIT_TO_GBPS, float(total_link_load_units[low_mask].max()) * UNIT_TO_GBPS),
            "cutoff_label": f"<= {q33_gbps:.2f} Gbps",
            "axis_label": f"Low Load\n(<= {q33_gbps:.2f} Gbps)"
        },
        "Medium": {
            "mask": med_mask,
            "count": int(med_mask.sum()),
            "range_units": (float(total_link_load_units[med_mask].min()), float(total_link_load_units[med_mask].max())),
            "range_gbps": (float(total_link_load_units[med_mask].min()) * UNIT_TO_GBPS, float(total_link_load_units[med_mask].max()) * UNIT_TO_GBPS),
            "cutoff_label": f"({q33_gbps:.2f}, {q67_gbps:.2f}] Gbps",
            "axis_label": f"Medium Load\n({q33_gbps:.2f} - {q67_gbps:.2f} Gbps)"
        },
        "High": {
            "mask": high_mask,
            "count": int(high_mask.sum()),
            "range_units": (float(total_link_load_units[high_mask].min()), float(total_link_load_units[high_mask].max())),
            "range_gbps": (float(total_link_load_units[high_mask].min()) * UNIT_TO_GBPS, float(total_link_load_units[high_mask].max()) * UNIT_TO_GBPS),
            "cutoff_label": f"> {q67_gbps:.2f} Gbps",
            "axis_label": f"High Load\n(> {q67_gbps:.2f} Gbps)"
        }
    }
    return scenarios, total_link_load_units, q33_units, q67_units, q33_gbps, q67_gbps


def plot_scenario_comparison(summary_rows, metric_name, out_path, scenarios_info):
    """Plot grouped bar chart comparing models across Low, Medium, and High load scenarios in Mbps."""
    scenarios = ["Low", "Medium", "High"]
    models = ["GRU", "Physical Graph-GRU", "Proposed Mixed Graph-GRU"]
    palette = ["#4C72B0", "#DD8452", "#55A868"]

    mean_map = {(r["scenario"], r["model"]): r[f"{metric_name}_mean_Mbps"] for r in summary_rows}
    std_map = {(r["scenario"], r["model"]): r[f"{metric_name}_std_Mbps"] for r in summary_rows}

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
    x = np.arange(len(scenarios))
    bar_width = 0.25

    max_val = 0.0
    for i, model in enumerate(models):
        means = [mean_map[(s, model)] for s in scenarios]
        stds = [std_map[(s, model)] for s in scenarios]
        offsets = x + (i - 1) * bar_width
        bars = ax.bar(
            offsets, means, yerr=stds, width=bar_width,
            label=model, color=palette[i], edgecolor="#2b2b2b",
            linewidth=1.0, alpha=0.9, capsize=5,
            error_kw={"elinewidth": 1.3, "ecolor": "#2b2b2b", "capthick": 1.3},
            zorder=3
        )
        for bar, m, s in zip(bars, means, stds):
            top = m + s
            if top > max_val:
                max_val = top
            ax.annotate(
                f"{m:.2f}\n± {s:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#222222"
            )

    x_labels = [scenarios_info[s]["axis_label"] for s in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="medium")
    ax.set_ylabel(f"{metric_name} (Mbps)", fontsize=12, fontweight="medium")
    ax.set_title(
        f"Test {metric_name} Across Traffic Load Scenarios (Mean ± Std in Mbps)\n"
        f"[Converted via 1 dataset unit = 8/3 bps; Load thresholds in Gbps]",
        fontsize=12.5, fontweight="bold", pad=12
    )
    ax.legend(frameon=True, fontsize=10, loc="upper right")
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
    print(f"Saved {metric_name} scenario chart (Mbps) to: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("\nUnit Conversion Formula:")
    print("  1 dataset unit = 8/3 bps ≈ 2.66667 bps (100 bytes / 300 s)")
    print("  Value in Mbps  = Value (units) * (8 / 3) * 10^-6")
    print("  Total in Gbps  = Total (units) * (8 / 3) * 10^-9")

    # 1. Load data and establish splits strictly matching train.py
    package = np.load(DATA_PATH)
    raw, names = package["link_loads_internal_30"].astype(np.float32), package["internal_link_names"]

    a, b = int(0.6 * len(raw)), int(0.8 * len(raw))
    # Scaling parameters computed from training set only
    mean, std = raw[:a].mean(0), raw[:a].std(0)
    std[std < 1e-6] = 1
    scaled = (raw - mean) / std

    tx, ty = windows(scaled[:a])
    vx, vy = windows(scaled[a:b])
    qx, qy = windows(scaled[b:])

    # Physical and similarity graphs built using training series only
    physical, mixed = make_graphs(names, scaled[:a])
    specs = [
        ("GRU", GRUBaseline, None),
        ("Physical Graph-GRU", GraphGRU, physical),
        ("Proposed Mixed Graph-GRU", GraphGRU, mixed)
    ]
    device_graphs = {id(g): g.to(device) for g in (physical, mixed)}

    # Unscaled test ground truth (original units)
    truth = qy.numpy() * std + mean
    scenarios, total_load_units, q33_units, q67_units, q33_gbps, q67_gbps = assign_load_scenarios(truth)

    print("\n--- Test Set Load Scenario Assignment ---")
    print(f"Total test windows: {len(truth)}")
    print(f"Network Load Thresholds (total across 30 internal links):")
    print(f"  Lower 33.3% cutoff (P33.3): {q33_gbps:.3f} Gbps ({q33_units:,.0f} dataset units)")
    print(f"  Upper 66.7% cutoff (P66.7): {q67_gbps:.3f} Gbps ({q67_units:,.0f} dataset units)")
    for s_name, s_info in scenarios.items():
        print(f"  {s_name} load: {s_info['count']} windows | {s_info['cutoff_label']} "
              f"(Range: [{s_info['range_gbps'][0]:.2f}, {s_info['range_gbps'][1]:.2f}] Gbps)")

    # 2. Run training and evaluate per seed and load scenario (in dataset units)
    per_seed_rows = []
    for label, model_class, graph in specs:
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = model_class(len(names)).to(device)
            loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH, shuffle=True)
            adj = None if graph is None else device_graphs[id(graph)]

            model = fit(model, loader, vx.to(device), vy.to(device), adj, device)
            model.eval()

            with torch.no_grad():
                pred = model(qx.to(device)) if adj is None else model(qx.to(device), adj)
            pred = pred.cpu().numpy() * std + mean

            # Calculate metrics per scenario band in dataset units, then convert to Mbps
            for s_name, s_info in scenarios.items():
                mask = s_info["mask"]
                pred_band = pred[mask]
                truth_band = truth[mask]

                band_mae_units = float(np.abs(pred_band - truth_band).mean())
                band_rmse_units = float(np.sqrt(np.square(pred_band - truth_band).mean()))

                band_mae_mbps = band_mae_units * UNIT_TO_MBPS
                band_rmse_mbps = band_rmse_units * UNIT_TO_MBPS

                per_seed_rows.append({
                    "scenario": s_name,
                    "model": label,
                    "seed": seed,
                    "window_count": s_info["count"],
                    "threshold_label": s_info["cutoff_label"],
                    "MAE_Mbps": band_mae_mbps,
                    "RMSE_Mbps": band_rmse_mbps,
                    "MAE_units": band_mae_units,
                    "RMSE_units": band_rmse_units
                })
                print(f"[{s_name} Load ({s_info['cutoff_label']})] {label}, seed {seed}: "
                      f"MAE={band_mae_mbps:.2f} Mbps ({band_mae_units:.0f} units), "
                      f"RMSE={band_rmse_mbps:.2f} Mbps ({band_rmse_units:.0f} units)")

    # 3. Save per-seed results CSV
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with PER_SEED_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario", "threshold_label", "model", "seed", "window_count",
            "MAE_Mbps", "RMSE_Mbps", "MAE_units", "RMSE_units"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_seed_rows:
            writer.writerow({
                "scenario": r["scenario"],
                "threshold_label": r["threshold_label"],
                "model": r["model"],
                "seed": r["seed"],
                "window_count": r["window_count"],
                "MAE_Mbps": f"{r['MAE_Mbps']:.3f}",
                "RMSE_Mbps": f"{r['RMSE_Mbps']:.3f}",
                "MAE_units": f"{r['MAE_units']:.3f}",
                "RMSE_units": f"{r['RMSE_units']:.3f}"
            })
    print(f"\nSaved per-seed scenario results to: {PER_SEED_CSV}")

    # 4. Compute and save summary table across seeds
    summary_rows = []
    scenario_order = ["Low", "Medium", "High"]
    model_order = [s[0] for s in specs]

    for s_name in scenario_order:
        for m_name in model_order:
            matches = [r for r in per_seed_rows if r["scenario"] == s_name and r["model"] == m_name]
            mae_mbps = np.array([r["MAE_Mbps"] for r in matches], dtype=np.float64)
            rmse_mbps = np.array([r["RMSE_Mbps"] for r in matches], dtype=np.float64)
            mae_units = np.array([r["MAE_units"] for r in matches], dtype=np.float64)
            rmse_units = np.array([r["RMSE_units"] for r in matches], dtype=np.float64)

            summary_rows.append({
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
                "RMSE_std_units": float(rmse_units.std(ddof=1))
            })

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario", "total_load_threshold_Gbps", "model", "seed_count", "window_count",
            "MAE_mean_Mbps", "MAE_std_Mbps", "RMSE_mean_Mbps", "RMSE_std_Mbps",
            "MAE_mean_units", "MAE_std_units", "RMSE_mean_units", "RMSE_std_units"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
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
                "RMSE_std_units": f"{r['RMSE_std_units']:.3f}"
            })
    print(f"Saved scenario summary table to: {SUMMARY_CSV}")

    # 5. Generate comparison bar charts with converted units
    plot_scenario_comparison(summary_rows, "MAE", MAE_CHART, scenarios)
    plot_scenario_comparison(summary_rows, "RMSE", RMSE_CHART, scenarios)

    # 6. Display formatted summary table to console
    print("\n" + "=" * 98)
    print(f"{'Scenario':<8} | {'Load Threshold':<16} | {'Model':<25} | {'MAE (Mean ± Std, Mbps)':<22} | {'RMSE (Mean ± Std, Mbps)':<22}")
    print("-" * 98)
    for r in summary_rows:
        mae_str = f"{r['MAE_mean_Mbps']:.2f} ± {r['MAE_std_Mbps']:.2f} Mbps"
        rmse_str = f"{r['RMSE_mean_Mbps']:.2f} ± {r['RMSE_std_Mbps']:.2f} Mbps"
        print(f"{r['scenario']:<8} | {r['total_load_threshold_Gbps']:<16} | {r['model']:<25} | {mae_str:<22} | {rmse_str:<22}")
    print("=" * 98)


if __name__ == "__main__":
    main()
