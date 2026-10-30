"""Convert Abilene OD traffic matrices into directed link-load series."""
from pathlib import Path
import gzip
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "abilene"
OUT = ROOT / "data" / "processed" / "abilene_link_loads.npz"


def read_rows(path, nfields):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fields = line.split()
            if len(fields) != nfields:
                raise ValueError(f"Unexpected row in {path}: {line}")
            rows.append(fields)
    return rows


def main():
    demands = read_rows(RAW / "demands", 2)
    links = read_rows(RAW / "links", 3)
    if len(demands) != 144 or len(links) != 54:
        raise ValueError("Expected 144 OD pairs and 54 links")
    with gzip.open(RAW / "X01.gz", "rt") as f:
        stored = np.loadtxt(f, dtype=np.float64)
    if stored.ndim != 2 or stored.shape[1] != 720:
        raise ValueError(f"Expected 720 columns per time step; got {stored.shape}")
    # The public file stores five values per OD pair; the measured value is first.
    od = stored.reshape(-1, 144, 5)[:, :, 0]
    link_names = [r[0] for r in links]
    demand_names = [r[0] for r in demands]
    routing = np.zeros((54, 144), dtype=np.float64)
    for name, demand, li, di, frac in read_rows(RAW / "A", 5):
        li, di = int(li) - 1, int(di) - 1
        if link_names[li] != name or demand_names[di] != demand:
            raise ValueError("Routing indices disagree with metadata")
        routing[li, di] += float(frac)
    loads = od @ routing.T
    link_types = np.array([int(r[2]) for r in links])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, od_flows=od, link_loads_all_54=loads,
        link_loads_internal_30=loads[:, link_types == 0],
        link_names=np.array(link_names),
        internal_link_names=np.array(link_names)[link_types == 0],
    )
    print(f"OD={od.shape}; all links={loads.shape}; internal links={loads[:, link_types == 0].shape}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
