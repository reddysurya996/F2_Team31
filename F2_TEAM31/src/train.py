"""Run GRU, physical-graph, and proposed mixed-graph comparisons."""
from pathlib import Path
import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from models import GRUBaseline, GraphGRU

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/abilene_link_loads.npz"
RESULTS = ROOT / "outputs/model_results.csv"
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    package = np.load(DATA)
    raw, names = package["link_loads_internal_30"].astype(np.float32), package["internal_link_names"]
    a, b = int(.6 * len(raw)), int(.8 * len(raw))
    mean, std = raw[:a].mean(0), raw[:a].std(0)
    std[std < 1e-6] = 1
    scaled = (raw - mean) / std
    tx, ty = windows(scaled[:a])
    vx, vy = windows(scaled[a:b])
    qx, qy = windows(scaled[b:])
    physical, mixed = make_graphs(names, scaled[:a])
    specs = [("GRU", GRUBaseline, None), ("Physical Graph-GRU", GraphGRU, physical),
             ("Proposed Mixed Graph-GRU", GraphGRU, mixed)]
    device_graphs = {id(g): g.to(device) for g in (physical, mixed)}
    rows = []
    for label, model_class, graph in specs:
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = model_class(len(names)).to(device)
            loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH, shuffle=True)
            adj = None if graph is None else device_graphs[id(graph)]
            model = fit(model, loader, vx.to(device), vy.to(device), adj, device)
            model.eval()
            with torch.no_grad():
                pred = model(qx.to(device)) if adj is None else model(qx.to(device), adj)
            pred = pred.cpu().numpy() * std + mean
            truth = qy.numpy() * std + mean
            rows.append({"model": label, "seed": seed,
                         "MAE": float(np.abs(pred - truth).mean()),
                         "RMSE": float(np.sqrt(np.square(pred - truth).mean()))})
            print(f"{label}, seed {seed}: MAE={rows[-1]['MAE']:.3f}, RMSE={rows[-1]['RMSE']:.3f}")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "MAE", "RMSE"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved per-seed results to {RESULTS}")


if __name__ == "__main__":
    main()
