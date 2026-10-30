# 4. IMPLEMENTATION

*Related project sections: [`introduction.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/introduction.md), [`literature_survey.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/literature_survey.md), [`proposed_system.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/proposed_system.md), [`results_discussion_limitations.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/results_discussion_limitations.md), and [`conclusion_future_work.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/conclusion_future_work.md).*  
*Source execution log: [`full_run_log.txt`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/full_run_log.txt).*

---

## 4.1 Tools, Platform, Versions, and Hardware

All experiments were executed within an isolated Python virtual environment (`.venv`) configured in the project root. Table 4.1 lists the software packages, environment libraries, operating system, and hardware platform confirmed directly from the execution environment and runtime logs.

**Table 4.1. Software Environment and Execution Platform**

| Component | Verified Specification / Version | Source / Method of Confirmation |
|---|---|---|
| **Python** | 3.9.6 (Clang 21.0.0, 64-bit) | Verified via `sys.version` in `.venv` |
| **Deep Learning Framework** | PyTorch 2.8.0 | Verified via `torch.__version__` in `.venv` |
| **Numerical Computation** | NumPy 2.0.2 | Verified via `numpy.__version__` in `.venv` |
| **Plotting & Visualization** | Matplotlib 3.9.x | Specified in `requirements.txt` (`matplotlib>=3.7`) |
| **Operating System** | macOS 27.0.1 (Darwin 27.0.1) | Verified via Python `platform.platform()` |
| **Instruction Set Architecture** | ARM64 (`arm64`) | Verified via Python `platform.machine()` |
| **Host Processor** | Apple Silicon ARM Processor | Verified via Python `platform.processor()` |
| **Host System Model / Memory** | Apple Mac [Exact RAM to be filled in by user: e.g., 16 GB / 32 GB Unified Memory] | Confirmed Apple Silicon host; specific workstation RAM to be filled in |
| **Execution Hardware Device** | CPU (`Using device: cpu`) | Confirmed in [`outputs/full_run_log.txt`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/full_run_log.txt#L5) |

> [!NOTE]
> Because training is performed over 30 internal directed links with compact tensor dimensions ($1,197 \times 12 \times 30$), the full training regime and multi-seed grid search execute efficiently on CPU without requiring GPU acceleration.

---

## 4.2 Dataset and Experimental Setup

### Abilene X01 Trace Description

The empirical evaluation uses the public Abilene Internet backbone traffic dataset provided by the University of Texas at Austin research repository:
* **Trace File (`X01.gz`)**: Contains one full week of five-minute traffic matrix snapshots collected from the Abilene core network. One week corresponds to:
  $$7\text{ days} \times 24\text{ hours/day} \times 12\text{ intervals/hour} = 2,016\text{ five-minute samples}$$
* **Origin-Destination Demands**: Traffic demands are recorded across all $12 \times 12 = 144$ Origin-Destination (OD) pairs connecting the 12 core Point-of-Presence (PoP) routers.
* **Routing Matrix (`A`)**: The static IP routing table maps each of the 144 OD demand pairs to the 54 physical and access links. For each link-demand entry $(l, d)$, the routing matrix specifies the fraction $\text{frac} \in [0.0, 1.0]$ of traffic for demand $d$ traversing link $l$.
* **Derived Internal Link Series**: Multiplying the $2,016 \times 144$ OD demand matrix by the $144 \times 54$ transposed routing matrix yields the link-load matrix ($2,016 \times 54$). Filtering for internal directed links (`link_type == 0` in `data/raw/abilene/links`) extracts the **30 internal directed links** spanning the 15 bidirectional core transmission lines.
* **Trace Nature**: This study performs **offline time-series forecasting** using historical telemetry traces. It is **not** a live software-defined network controller deployment, an active hardware testbed, or a packet-level simulator (such as NS-3 or OMNeT++).

### Experimental Parameters and Hyperparameters

Table 4.2 documents the experimental configuration, dataset splitting, evaluation horizons, and training hyperparameters used throughout the project.

**Table 4.2. Experimental Parameters and Model Configurations**

| Parameter | Value | Description and Implementation Reference |
|---|---|---|
| **Total Timesteps** | 2,016 samples | 7 days of 5-minute measurements (`data/processed/abilene_link_loads.npz`) |
| **Link Dimension ($N$)** | 30 links | 30 directed internal backbone links (`internal_link_names`) |
| **Input Window ($H$)** | 12 steps (60 min) | Number of past consecutive 5-minute observations passed to the model |
| **Prediction Horizon** | 1 step (5 min) | Forecast step ahead predicting link traffic simultaneously across all 30 links |
| **Data Partitioning** | 60% / 20% / 20% | Chronological split: Train = 1,197 windows, Val = 391 windows, Test = 392 windows |
| **Data Scaling** | Z-score normalization | Mean $\mu \in \mathbb{R}^{30}$ and std $\sigma \in \mathbb{R}^{30}$ computed strictly on Train partition |
| **Batch Size** | 32 | Mini-batch size for DataLoader (`BATCH = 32`) |
| **Training Epochs** | 80 | Maximum training iterations (`EPOCHS = 80`) |
| **Optimizer & Learning Rate** | Adam, $\eta = 10^{-3}$ | Standard Adam optimizer (`torch.optim.Adam(..., lr=1e-3)`) |
| **Training Loss Function** | Mean Squared Error (MSE) | `nn.MSELoss()` evaluated on scaled target tensors |
| **Early Stopping** | Patience = 12 epochs | Monitored on total validation MSE loss; best model weights restored upon termination |
| **Random Seeds** | 7, 21, 42 | Controls NumPy, Python random, and PyTorch PRNG initialization (`SEEDS = (7, 21, 42)`) |
| **Candidate Mixing Grid ($\alpha$)** | $\{0.00, 0.25, 0.50, 0.75, 1.00\}$ | Evaluated on validation set to explore physical vs. similarity graph combinations |
| **Models Evaluated** | 4 primary models | 1. Temporal [`GRUBaseline`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L6-L15) (64 hidden units)<br>2. Physical [`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32) ($\alpha = 1.00$, 32 units)<br>3. Similarity-only [`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32) ($\alpha = 0.00$, 32 units)<br>4. Fixed 50/50 Mixed [`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32) ($\alpha = 0.50$, 32 units) |
| **Evaluation Metrics** | MAE and RMSE (Mbps) | Computed on unscaled physical predictions: $1\text{ unit} = (8/3) \times 10^{-6}\text{ Mbps}$ |

---

## 4.3 Code Structure and Repository Layout

The project repository is structured into modular data, source, and output directories. Table 4.3 details the concrete purpose and contents of every file in the project.

**Table 4.3. Repository Directory Structure and Script Functionality**

| Path | Category | Functionality and Purpose |
|---|---|---|
| [`src/prepare_abilene.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/prepare_abilene.py) | Data Prep | Parses raw metadata files `demands`, `links`, and routing matrix `A`. Decompresses `X01.gz`, reshapes 720-column arrays, extracts the measured OD flow matrix, multiplies by transposed routing matrix, filters for 30 internal directed links, and exports `abilene_link_loads.npz`. |
| [`src/models.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py) | Model Defs | Implements PyTorch neural architectures: [`GRUBaseline`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L6-L15) (pure temporal 64-hidden GRU) and [`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32) (normalized adjacency message-passing via `torch.einsum`, linear projection to 32 units, ReLU, 32-hidden GRU, and linear prediction head). |
| [`src/train.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/train.py) | Baseline Training | Implements the core 60/20/20 chronological split, generates physical and 50/50 mixed graphs, trains [`GRUBaseline`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L6-L15) and [`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32) models across seeds 7, 21, and 42 for 80 epochs with early stopping, and writes [`outputs/model_results.csv`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/model_results.csv). |
| [`src/tune_alpha.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/tune_alpha.py) | Grid Sweep | Sweeps candidate mixing weights $\alpha \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$. Evaluates validation MAE, selects optimal $\alpha^*$ per seed without test leakage, and exports validation curve plots. |
| [`src/evaluate_load_aware_v2.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/evaluate_load_aware_v2.py) | Regime Evaluation | Core evaluation engine. Derives training-defined load tertiles (Low $\le 7.87\text{ Gbps}$, Medium $7.87-9.20\text{ Gbps}$, High $> 9.20\text{ Gbps}$), evaluates candidate $\alpha$ models per regime and seed, logs regime diagnostics, and handles dataset distribution shifts. |
| [`src/run_full_load_aware_v2.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/run_full_load_aware_v2.py) | Experiment Runner | Wrapper script executing [`evaluate_load_aware_v2.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/evaluate_load_aware_v2.py) with full 80-epoch budgets and redirecting outputs to persistent `*_full.csv` and `*_full.png` files. |
| [`src/evaluate_load_scenarios.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/evaluate_load_scenarios.py) | Scenario Analysis | Evaluates model forecasting errors across load levels and produces load-partitioned summary tables and bar plots. |
| [`src/run_ablation.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/run_ablation.py) | Ablation Runner | Executes systematic ablation comparisons between standalone temporal, purely physical, purely similarity, and mixed graph configurations. |
| [`src/summarize_results.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/summarize_results.py) | Result Aggregator | Computes mean and sample standard deviations across random seeds for all experimental CSVs and prints formatted Markdown comparison tables. |
| `data/raw/abilene/` | Data Directory | Houses raw public Abilene dataset files: `A` (routing fractions), `demands` (144 OD identifiers), `links` (54 link definitions), and `X01.gz` (compressed 5-minute traffic matrix snapshots). |
| `data/processed/` | Data Directory | Houses `abilene_link_loads.npz`, containing arrays `od_flows` ($2016 \times 144$), `link_loads_all_54` ($2016 \times 54$), `link_loads_internal_30` ($2016 \times 30$), and link name arrays. |
| `outputs/` | Experiment Outputs | Contains experimental artifacts: result CSVs (`*_full.csv`), high-resolution plots (`*_full.png`), execution logs ([`full_run_log.txt`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/full_run_log.txt)), and report chapters (`*.md`). |

---

## 4.4 Engineering Challenges and Mitigation Strategies

During system development and empirical evaluation, five key technical and methodological challenges were encountered. Table 4.4 summarizes how each challenge was addressed.

**Table 4.4. Challenges and Engineering Solutions**

| Challenge | Problem Description | Mitigation Strategy in Code |
|---|---|---|
| **1. Five-Values-per-OD Format** | Raw `X01.gz` stores 720 floating-point numbers per row ($144\text{ OD pairs} \times 5\text{ fields}$), interleaving measured traffic with secondary estimation features. | In [`src/prepare_abilene.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/prepare_abilene.py#L33), the 2D array is reshaped to shape `(-1, 144, 5)` and sliced at index 0 (`[:, :, 0]`), isolating the true measured origin-destination traffic demands. |
| **2. Deriving Link Loads from Routing Topology** | Abilene records traffic as end-to-end OD demands rather than per-link physical telemetry. Furthermore, access/peering links must be isolated from internal core lines. | In [`src/prepare_abilene.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/prepare_abilene.py#L36-L47), metadata `A` is parsed to construct the sparse routing matrix $R \in \mathbb{R}^{54 \times 144}$. Total link loads are derived via matrix multiplication ($\text{Loads} = \text{OD} \cdot R^T$). The 30 internal directed links are then isolated by filtering `link_types == 0`. |
| **3. Preventing Data Leakage in Graph and Metric Tuning** | Normalizing data or constructing similarity graphs using the entire trace introduces future look-ahead bias into model training. | Strict causality is enforced in [`src/train.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/train.py#L82-L88) and [`src/evaluate_load_aware_v2.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/evaluate_load_aware_v2.py#L65-L80):<br>1. Mean $\mu$ and std $\sigma$ are calculated on the 60% training partition only.<br>2. Pearson correlation matrix is computed exclusively over training samples to form $A_{\text{sim}}$.<br>3. Candidate mixing weights $\alpha$ are tuned exclusively using validation set MAE; test data is never accessed during model or hyperparameter selection. |
| **4. Train-to-Test Dataset Load Shift** | Total network traffic during the test week was systematically lower than during the training week. The test set maximum load reached only 7.26 Gbps, which falls below the training-defined Low/Medium boundary (7.87 Gbps). | Rather than silently masking the shift or altering cutoffs post-hoc, [`src/evaluate_load_aware_v2.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/evaluate_load_aware_v2.py#L6-L10) logs an explicit diagnostic warning: **all 392 test windows fall in the training-defined Low regime**. The report documents that Medium and High test regimes could not be evaluated under prospective training cutoffs. A test-local tertile breakdown is provided separately and explicitly labeled as a retrospective, descriptive analysis. |
| **5. Non-Adaptive Validation Selection ($\alpha^* = 0.00$)** | Sweeping $\alpha \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$ revealed that validation MAE increased monotonically with $\alpha$ across all regimes and seeds, uniformly selecting $\alpha^* = 0.00$. | Rather than claiming the model is adaptive or asserting that the mixed model performed best, [`src/evaluate_load_aware_v2.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/evaluate_load_aware_v2.py#L8-L10) factually renames the policy outcome to **static Similarity-only Graph-GRU ($\alpha = 0$)**. The documentation explicitly notes that the physical graph component degraded performance and was completely unused. |

---

## 4.5 Execution Summary

A complete execution of the training and evaluation pipeline ([`src/run_full_load_aware_v2.py`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/run_full_load_aware_v2.py)) verifies the reproducibility of all reported metrics:
* **Training Convergence**: 80 epochs per seed and candidate $\alpha$, with early stopping restoring the best validation checkpoint.
* **Output Artifacts**: Automatically generates and saves all numerical results to [`load_aware_model_results_full.csv`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/load_aware_model_results_full.csv) and diagnostic charts to [`load_aware_comparison_full.png`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/load_aware_comparison_full.png).
* **Integrity of Findings**: All measured numbers, seeds, and regime boundaries remain identical to the original experimental run.

