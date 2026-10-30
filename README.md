# Spatio-Temporal GNN for Network Traffic Forecasting

**Course:** BCSE308P – Computer Networks Lab  
**Semester:** Fall 2026–27  
**Slot / Team:** F2 / Team 31

## Team

| Register Number | Student Name |
|---|---|
| 24BAI0171 | M. Charan Sai |
| 24BCE0677 | Surya R |
| 24BCE2552 | A. Ram Likhith |

## Project Overview

This project forecasts traffic on the internal directed links of the Abilene backbone network. It uses the previous 12 five-minute observations—one hour of history—to predict traffic one step ahead, or five minutes into the future.

The project compares temporal forecasting with graph-based forecasting. It evaluates whether link-to-link relationships based on physical network connectivity, historical traffic similarity, or a fixed mixture of both produce more accurate predictions.

This is a course-scale experimental comparison. The Graph-GRU implementation is **not an exact reproduction of DCRNN**.

## Problem and Motivation

Traffic loads on network links can vary over time and can be related across links. A standard recurrent model learns from each time step but does not explicitly represent the relationships between links. A physical graph represents links that share a router endpoint, but physical proximity may not fully reflect which links carry traffic that changes together.

This project compares physical connectivity with a traffic-similarity graph built from historical link loads. The goal is to measure how each graph choice affects short-term forecasting accuracy on the Abilene dataset.

## Objectives

- Convert Abilene origin–destination traffic measurements into link-load time series.
- Compare a GRU without graph information against Graph-GRU models using different adjacency graphs.
- Build the traffic-similarity graph from training data only.
- Select graph mixing weights using validation data, keeping test data out of model selection.
- Report results across three random seeds using mean and standard deviation.
- Examine load-specific results while clearly distinguishing test-set descriptions from prospective evaluation.

## Dataset

The project uses the public Abilene traffic matrix data and routing metadata:

- `data/raw/abilene/X01.gz` — measured traffic matrix data.
- `data/raw/abilene/demands` — origin–destination demand metadata.
- `data/raw/abilene/links` — link metadata.
- `data/raw/abilene/A` — routing information.
- `data/processed/abilene_link_loads.npz` — processed OD flows and link loads used by the models.

The preprocessing script checks the expected 144 OD pairs and 54 links, reads the measured value for each OD pair, applies the routing matrix, and selects the 30 internal directed links for the forecasting experiment.

The measurements are sampled every five minutes. The forecasting input is 12 consecutive samples, and the output is the next sample.

**Dataset source:** [Abilene traffic matrix archive](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/)

## Method

### Data split and scaling

The data is divided chronologically into 60% training, 20% validation, and 20% test partitions. Scaling statistics are calculated from training data and applied to the other partitions. The similarity graph is also constructed using training data only.

The experiment uses random seeds 7, 21, and 42. Results are summarized as mean ± sample standard deviation across those three runs.

### Models compared

1. **GRU baseline** — processes the history of link loads without an explicit graph.
2. **Physical Graph-GRU** — aggregates information between links that share a router endpoint, then models the temporal sequence with a GRU.
3. **Similarity Graph-GRU** — uses a graph based on the strongest positive correlations between link-load histories. Each link selects up to three positively correlated links; the graph is derived from training data only.
4. **Fixed mixed Graph-GRU** — combines physical and similarity adjacency with a fixed weight of 0.5:
   
   `A(α) = α A_phys + (1 − α) A_sim`

   For the fixed mixed model, `α = 0.5`.

The Graph-GRU first aggregates link measurements using a normalized adjacency matrix, maps the aggregated values to hidden features, and processes each link’s sequence with a GRU.

### Validation-based alpha selection

A separate experiment evaluates `α` values of 0, 0.25, 0.50, 0.75, and 1.00. The value is selected using validation performance, not test results. In this project, validation selected `α = 0` for every evaluated seed and load regime. Therefore, the selected model is **static Similarity Graph-GRU**, not an adaptive graph policy.

## Main Results

Overall test errors are reported in Mbps. **Lower MAE and RMSE indicate lower prediction error.**

| Model | MAE (Mbps, mean ± SD) | RMSE (Mbps, mean ± SD) |
|---|---:|---:|
| GRU | 22.80 ± 1.35 | 33.25 ± 2.54 |
| Physical Graph-GRU | 25.47 ± 0.14 | 36.66 ± 0.40 |
| Similarity Graph-GRU | **22.28 ± 0.03** | **30.68 ± 0.06** |
| Fixed mixed Graph-GRU | 22.91 ± 0.34 | 32.72 ± 0.64 |

In this experiment, the Similarity Graph-GRU had the lowest overall MAE and RMSE. The fixed mixed and physical graph models did not outperform the similarity-only model overall. These results describe this dataset and test split; they do not establish that the similarity graph will be best on other networks or time periods.

## Load-Regime Evaluation and Important Caveat

The project contains two different load-regime analyses. They should not be confused:

- **Primary, training-defined cutoffs:** Boundaries are calculated from training data only: Low ≤ 7.87 Gbps, Medium > 7.87 and ≤ 9.20 Gbps, and High > 9.20 Gbps. The test-set maximum is 7.26 Gbps, so all 392 test windows fall in the training-defined Low regime. The test set therefore provides no primary prospective evaluation of Medium- or High-load traffic.
- **Descriptive, test-local tertiles:** The ablation script also divides the test windows into three groups using the test distribution itself: Low ≤ 5.66 Gbps, Medium > 5.66 and ≤ 6.07 Gbps, and High > 6.07 Gbps. These groups contain 131, 130, and 131 windows. This is a **post-hoc descriptive breakdown**, not a training-time policy or an adaptive routing mechanism.

For the test-local groups, the Similarity Graph-GRU had the lowest MAE in the Low group. GRU had the lowest MAE in the Medium and High groups. The best-performing model therefore varied by group.

## Reproducibility

The commands below assume you have cloned this repository and are in its top-level directory. The project files are in the `F2_TEAM31/` subfolder.

### Setup on macOS or Linux

```bash
git clone https://github.com/reddysurya996/F2_Team31.git
cd F2_Team31/F2_TEAM31

python3 -m venv .venv
./.venv/bin/python3 -m pip install --upgrade pip
./.venv/bin/python3 -m pip install -r requirements.txt
```

### Prepare the data and run the four-model ablation

```bash
./.venv/bin/python3 src/prepare_abilene.py
./.venv/bin/python3 src/run_ablation.py
```

The ablation script writes per-seed results, summary CSV files, and comparison charts to `outputs/`.

### Other experiment scripts

- `src/train.py` — runs the three-model GRU, physical Graph-GRU, and fixed mixed Graph-GRU comparison.
- `src/summarize_results.py` — summarizes the output from `src/train.py`.
- `src/evaluate_load_scenarios.py` — evaluates models by load scenario.
- `src/tune_alpha.py` — evaluates candidate graph-mixing weights.
- `src/evaluate_load_aware.py` and `src/evaluate_load_aware_v2.py` — run load-aware validation and evaluation experiments.
- `src/run_full_load_aware_v2.py` — wrapper for the full load-aware experiment and its `_full` output files.

The full load-aware run trains multiple models and may take longer than the ablation run, especially on CPU.

### Windows setup

Create the virtual environment with `python -m venv .venv`, then install dependencies with `.venv\Scripts\python.exe -m pip install -r requirements.txt`. Run scripts with `.venv\Scripts\python.exe src\run_ablation.py`.

## Project Structure

```text
F2_Team31/
├── F2_TEAM31/
│   ├── data/
│   │   ├── raw/abilene/       # Abilene source data and metadata
│   │   └── processed/         # Processed link-load dataset
│   ├── outputs/               # CSV results, charts, report sections and logs
│   ├── src/                   # Preprocessing, models, training and evaluation
│   ├── requirements.txt
│   └── README.md
└── README.md                  # Repository landing-page documentation
```

The `.venv` environment is machine-specific and is not required in the repository. Dependencies are listed in `requirements.txt`.

## Limitations

- The evaluation uses one network dataset and one chronological split.
- The test period has lower observed load than the training period; training-defined Medium and High regimes have no test samples.
- The test-local tertile analysis is descriptive and post-hoc.
- The experiment predicts one five-minute step ahead.
- Results use three random seeds, which provides only a limited estimate of run-to-run variation.
- The selected alpha was zero in every validation regime, so the final selected graph was not adaptive.
- The Graph-GRU is a simplified implementation and should not be presented as an exact DCRNN reproduction.
- Results should be treated as findings for this experiment, not as proof of universal model superiority.

## References

1. D. Andreoletti, S. Troia, F. Musumeci, S. Giordano, G. Maier, and M. Tornatore, “Network Traffic Prediction Based on Diffusion Convolutional Recurrent Neural Networks,” *Proc. IEEE INFOCOM Workshops*, 2019, doi: [10.1109/INFCOMW.2019.8845132](https://ieeexplore.ieee.org/document/8845132/).
2. J. Tao, K. Cao, and T. Liu, “Traffic Matrix Prediction Based on Cross Aggregate GNN,” 2023, doi: [10.1109/DASC/PiCom/CBDCom/Cy59711.2023.10361471](https://ieeexplore.ieee.org/document/10361471/).
3. Z. Yao et al., “Internet Traffic Forecasting Using Temporal-Topological Graph Convolutional Networks,” *Proc. IJCNN*, 2021, doi: [10.1109/IJCNN52387.2021.9534096](https://ieeexplore.ieee.org/document/9534096/).
4. G. O. Ferreira et al., “Forecasting Network Traffic: A Survey and Tutorial With Open-Source Comparative Evaluation,” *IEEE Access*, 2023, doi: [10.1109/ACCESS.2023.3236261](https://ieeexplore.ieee.org/document/10015152/).
5. Y. Zhang, M. Roughan, N. Duffield, and A. Greenberg, “Fast Accurate Computation of Large-Scale IP Traffic Matrices from Link Loads,” *Proc. ACM SIGMETRICS*, 2003. [Abilene dataset archive](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/).
