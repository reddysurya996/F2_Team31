# Spatio-Temporal GNN for Network Traffic Forecasting

This project predicts traffic on Abilene's internal directed links five minutes ahead. It compares a temporal GRU, a physical-topology Graph-GRU, and a proposed Graph-GRU that combines physical adjacency with training-data traffic similarity.

## Contents

- `data/raw/abilene/`: public Abilene data and routing metadata.
- `src/prepare_abilene.py`: builds OD and link-load series.
- `src/models.py`: GRU and Graph-GRU model classes.
- `src/train.py`: chronological train/validation/test experiment, three seeds, MAE/RMSE.
- `requirements.txt`: Python dependencies.
- `outputs/`: generated dataset and experiment results.

## Run on macOS/Linux

Open this folder in Antigravity. In its terminal, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/prepare_abilene.py
python src/train.py
```

On Windows, replace activation with `.venv\\Scripts\\activate`.

The training script uses the 30 internal directed links, 12 previous samples (one hour), and a one-step (five-minute) forecast. It splits the time series chronologically 60/20/20, computes scaling and similarity using training data only, and reports each of three seeded runs to `outputs/model_results.csv`. Compare mean and standard deviation across seeds before reporting conclusions. This is a one-week dataset, so results should be described as a course-scale evaluation rather than broad evidence of generalization.

Dataset source: [Abilene traffic matrix data](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/).
