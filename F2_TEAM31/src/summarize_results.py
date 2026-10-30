"""Summarize multi-seed experimental results and generate comparison bar charts."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "outputs" / "model_results.csv"
SUMMARY_FILE = ROOT / "outputs" / "model_summary.csv"
MAE_CHART_FILE = ROOT / "outputs" / "mae_comparison.png"
RMSE_CHART_FILE = ROOT / "outputs" / "rmse_comparison.png"


def load_results(path: Path):
    """Load model results grouped by model name."""
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")

    data = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"].strip()
            if model not in data:
                data[model] = {"seeds": [], "MAE": [], "RMSE": []}
            data[model]["seeds"].append(int(row["seed"]))
            data[model]["MAE"].append(float(row["MAE"]))
            data[model]["RMSE"].append(float(row["RMSE"]))
    return data


def compute_summary(data: dict):
    """Calculate mean and standard deviation (sample std, ddof=1) for each model."""
    summary_rows = []
    for model, values in data.items():
        mae_arr = np.array(values["MAE"], dtype=np.float64)
        rmse_arr = np.array(values["RMSE"], dtype=np.float64)

        ddof = 1 if len(mae_arr) > 1 else 0
        mae_mean = float(mae_arr.mean())
        mae_std = float(mae_arr.std(ddof=ddof))
        rmse_mean = float(rmse_arr.mean())
        rmse_std = float(rmse_arr.std(ddof=ddof))

        summary_rows.append({
            "model": model,
            "seed_count": len(values["seeds"]),
            "MAE_mean": mae_mean,
            "MAE_std": mae_std,
            "RMSE_mean": rmse_mean,
            "RMSE_std": rmse_std,
        })
    return summary_rows


def save_summary_csv(summary_rows: list, out_path: Path):
    """Save the aggregated summary to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "seed_count", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "model": row["model"],
                "seed_count": row["seed_count"],
                "MAE_mean": f"{row['MAE_mean']:.3f}",
                "MAE_std": f"{row['MAE_std']:.3f}",
                "RMSE_mean": f"{row['RMSE_mean']:.3f}",
                "RMSE_std": f"{row['RMSE_std']:.3f}",
            })
    print(f"Saved summary CSV to: {out_path}")


def plot_metric_bar_chart(models, means, stds, metric_name, out_path, color_palette=None):
    """Generate a clean, publication-ready bar chart with standard deviation error bars."""
    if color_palette is None:
        color_palette = ["#4C72B0", "#DD8452", "#55A868"]

    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    x_positions = np.arange(len(models))
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="#2b2b2b",
        linewidth=1.2,
        width=0.55,
        alpha=0.9,
        error_kw={"elinewidth": 1.5, "ecolor": "#2b2b2b", "capthick": 1.5},
        zorder=3
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, fontsize=11, fontweight="medium")
    ax.set_ylabel(f"{metric_name} (Traffic Flow Units)", fontsize=12, fontweight="medium")
    ax.set_title(f"{metric_name} by Model across Seeds (Mean ± Std Dev)", fontsize=13, fontweight="bold", pad=14)

    # Grid lines behind bars
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Add numeric annotations above each bar
    for bar, mean_val, std_val in zip(bars, means, stds):
        y_pos = mean_val + std_val
        ax.annotate(
            f"{mean_val:,.0f}\n± {std_val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#222222"
        )

    # Leave headroom for labels
    max_top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, max_top * 1.18)

    # Formatting y-axis with comma separators
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))

    # Style borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {metric_name} bar chart to: {out_path}")


def main():
    print(f"Reading results from: {RESULTS_FILE}")
    data = load_results(RESULTS_FILE)

    summary_rows = compute_summary(data)
    save_summary_csv(summary_rows, SUMMARY_FILE)

    models = [r["model"] for r in summary_rows]
    mae_means = [r["MAE_mean"] for r in summary_rows]
    mae_stds = [r["MAE_std"] for r in summary_rows]
    rmse_means = [r["RMSE_mean"] for r in summary_rows]
    rmse_stds = [r["RMSE_std"] for r in summary_rows]

    # Generate separate bar charts for MAE and RMSE
    plot_metric_bar_chart(models, mae_means, mae_stds, "MAE", MAE_CHART_FILE)
    plot_metric_bar_chart(models, rmse_means, rmse_stds, "RMSE", RMSE_CHART_FILE)

    # Print summary table to stdout
    print("\n" + "=" * 78)
    print(f"{'Model':<28} | {'MAE (Mean ± Std)':<22} | {'RMSE (Mean ± Std)':<22}")
    print("-" * 78)
    for r in summary_rows:
        mae_str = f"{r['MAE_mean']:,.1f} ± {r['MAE_std']:,.1f}"
        rmse_str = f"{r['RMSE_mean']:,.1f} ± {r['RMSE_std']:,.1f}"
        print(f"{r['model']:<28} | {mae_str:<22} | {rmse_str:<22}")
    print("=" * 78)


if __name__ == "__main__":
    main()
