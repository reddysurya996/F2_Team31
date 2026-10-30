# 2. LITERATURE SURVEY

*Related project sections: [`introduction.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/introduction.md), [`results_discussion_limitations.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/results_discussion_limitations.md), and [`conclusion_future_work.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/conclusion_future_work.md).*  
*Source spreadsheet: `IEEE_Network_Traffic_Forecasting_Papers.xlsx` (10 verified IEEE publications).*

---

## 2.1 Overview of the Existing Approach or Scheme

Accurate network traffic forecasting is a cornerstone of autonomous network management, dynamic bandwidth provisioning, traffic engineering, and proactive congestion avoidance in wide-area telecommunication networks. Over the past decade, methodologies for forecasting backbone link loads and traffic matrices (TMs) have evolved across three primary paradigms:

1. **Statistical and Linear Time-Series Models**: Early techniques relied on classical univariate and multivariate statistical predictors, such as Autoregressive Moving Average (ARMA), Autoregressive Integrated Moving Average (ARIMA), Seasonal ARIMA (SARIMA), Principal Component Analysis (PCA), and Gravity models [10]. While these models are computationally lightweight and provide analytical tractability, they assume linear dependencies and wide-sense stationarity. Real-world telecommunication traffic exhibits burstiness, non-linear diurnal fluctuations, routing shifts, and long-range temporal correlations that linear statistical formulations cannot effectively capture.
2. **Deep Recurrent Neural Networks (RNNs)**: The adoption of deep learning introduced architectures such as Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs), exemplified by frameworks like NeuTM [4]. By using gating mechanisms to mitigate vanishing gradients, recurrent networks capture complex temporal sequences and non-linear diurnal patterns over extended observation windows. However, standard RNN architectures operate on flattened 1D vectors or treat individual links in isolation. They treat the network as an arbitrary collection of time series, neglecting the underlying topological structure of communication networks and the physical constraints governing packet routing.
3. **Spatio-Temporal Graph Neural Networks (ST-GNNs)**: To model both spatial topology and temporal dynamics simultaneously, recent research has adapted graph neural networks to telecommunications. Representative models—such as the Diffusion Convolutional Recurrent Neural Network (DCRNN) applied to network traffic by Andreoletti et al. [1] (the base paper for this investigation), Temporal-Topological Graph Convolutional Networks (TTGCN) [3], Cross-Aggregate GNNs (CAGNN) [2], and GCN-GRU models for optical networks [6]—combine spatial graph convolutions with recurrent cells (GRU or LSTM).

### Detailed Analysis of the Existing Base Scheme (Andreoletti et al., 2019 [1])

The primary benchmark and foundational scheme for graph-based backbone link-load prediction is the DCRNN framework formulated by Andreoletti et al. [1]. In their setup, evaluated on the Abilene backbone network (12 router nodes, 15 full-duplex links yielding 30 directed internal links):

* **Topological Formulation**: The network is modeled as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)$, where the vertices $\mathcal{V}$ represent physical network routers and the edges $\mathcal{E}$ represent physical directed links. Spatial dependencies are modeled by computing diffusion convolutions over the physical adjacency matrix $A_{\text{phys}}$ using bidirectional random walks.
* **Temporal Integration**: The spatial diffusion convolution is embedded directly inside the gating equations of a Gated Recurrent Unit (GRU), enabling the model to learn spatial dependencies across neighboring nodes at each recurrent time step.
* **Reported Achievements**: By leveraging the physical network topology, Andreoletti et al. demonstrated marked improvements over non-spatial deep learning baselines (FCNN, CNN, LSTM, and CNN-LSTM), reducing the Mean Absolute Percentage Error (MAPE) from approximately 210% (achieved by standalone LSTMs) to 43%, and lowering Mean Absolute Error (MAE) by roughly 30 Mbps across the 30 directed backbone links of Abilene.

### Limitations of the Existing Base Scheme

Despite the reported improvements of DCRNN [1] and related spatio-temporal architectures [3], [6], three fundamental limitations characterize the existing schemes:

1. **Physical Adjacency as an Inexact Proxy for Traffic Correlation**: Existing schemes assume that spatial traffic dependencies follow physical link adjacency—i.e., that two links are correlated because they share a common router node or physical fiber link. However, in wide-area IP/MPLS networks, traffic is routed according to origin-destination demand pairs and shortest-path routing policies (e.g., OSPF/IS-IS weights). Two physically adjacent links connected to the same core router may carry completely orthogonal traffic streams heading toward disparate destinations. Conversely, two links separated by several hops may carry identical aggregated traffic flows from heavy source-destination pairs. Imposing a purely physical adjacency matrix ($A_{\text{phys}}$) forces the graph convolution to exchange messages between physically connected but statistically unrelated links, introducing topological noise.
2. **Absence of Data-Driven Behavioral Similarity Graphs**: Existing literature typically relies on static, predefined physical graphs. They rarely explore empirical correlation graphs ($A_{\text{sim}}$)—where edges connect links whose historical traffic volumes covary strongly over time—or hybrid combinations that balance physical connectivity with observed statistical co-movement.
3. **Rigid Inductive Bias Across Load Regimes**: Existing works evaluate models across an entire test set uniformly, assuming that a single fixed graph structure performs optimally across all operating conditions. In operational networks, low-load regimes may be dominated by localized background traffic, while high-load regimes may be dominated by heavy, correlated network-wide flows. The literature lacks systematic empirical studies that examine whether graph-weight blending policies can adapt across distinct network load regimes.

---

## 2.2 Related-Work Summary Table

The 10 papers summarized in Table 2.1 represent the verified literature identified from the project's literature survey. Each paper's bibliographic metadata, methodology, metrics, and experimental findings have been verified from original IEEE publications.

**Table 2.1. Systematic Summary of Related Work in Network Traffic Forecasting**

| Authors / Year | Technique | Metrics / Tools | Merits | Gaps |
|---|---|---|---|---|
| **D. Andreoletti, S. Troia, F. Musumeci, S. Giordano, G. Maier, and M. Tornatore (2019)** [1] *(Base Paper)* | **DCRNN**: Diffusion Convolutional Recurrent Neural Network combining bidirectional random walks over physical network topology with GRU recurrent cells. | **MAE, RMSE, MAPE**<br>• Dataset: 6 months of Abilene network traffic matrices (12 nodes, 30 directed links).<br>• Baselines: FCNN, CNN, LSTM, CNN-LSTM. | • First rigorous demonstration of graph diffusion convolutions for backbone link-load prediction.<br>• Reduced MAPE from ~210% (LSTM) to 43% and reduced MAE by ~30 Mbps on Abilene. | • Graph relies exclusively on physical router-level topology ($A_{\text{phys}}$).<br>• Cannot represent long-range correlation between physically distant links carrying shared OD flows.<br>• Static graph structure; does not evaluate traffic similarity or load-regime dependency. |
| **J. Tao, K. Cao, and T. Liu (2023)** [2] | **CAGNN**: Cross-Aggregate Graph Neural Network modeling dual graphs (node graph and edge graph) with interactive cross-aggregation for traffic matrices. | **RMSE, MAE**<br>• Dataset: Abilene network traffic matrices at 15-minute intervals.<br>• Baselines: LSTM-RNN, ARIMA, SVR.<br>• Implemented in PyTorch. | • Overcomes the single-graph bottleneck by cross-aggregating information between device-level nodes and link-level flows.<br>• Achieves superior prediction accuracy (RMSE ~0.02) compared to standalone LSTMs. | • Requires complete, end-to-end Origin-Destination routing matrices.<br>• Uses a fixed physical structural graph; does not evaluate empirical traffic similarity or test performance under load distribution shifts. |
| **Z. Yao, Q. Xu, Y. Chen, Y. Tu, H. Zhang, and Y. Chen (2021)** [3] | **TTGCN**: Temporal-Topological Graph Convolutional Network using link-centric line graphs (links as nodes connected by shared routers) and 1D temporal convolutions. | **MAE, RMSE**<br>• Dataset: UKERNA (UK Education and Research Networking Association) backbone.<br>• Baselines: Historical Average, ARIMA, GRU, STGCN. | • Formulates links directly as graph vertices to predict link throughput directly without OD matrix inversion errors.<br>• Outperforms standard STGCN and classical temporal models across multi-step horizons. | • Adjacency is strictly constrained to physical endpoint sharing.<br>• Fails to connect topologically separated links with high co-movement.<br>• Does not evaluate performance across discrete traffic load regimes. |
| **A. Azzouni and G. Pujolle (2018)** [4] | **NeuTM**: Deep Long Short-Term Memory (LSTM) recurrent neural network framework for traffic matrix prediction in Software-Defined Networks (SDN). | **MSE, RMSE**<br>• Dataset: GÉANT research network traffic matrices.<br>• Implemented in Keras / TensorFlow. | • Validates that deep recurrent architectures effectively capture non-linear, long-term temporal dependencies in SDN traffic matrices.<br>• Demonstrates fast convergence with compact model parameters compared to PCA/gravity models. | • Purely temporal model; flattens traffic matrices into 1D vectors.<br>• Completely disregards physical network topology, link connectivity, and spatial correlations between network paths. |
| **S. Mehrizi and S. Chatzinotas (2022)** [5] | **Graph Gaussian Processes (Graph GP)**: Non-parametric Bayesian spatio-temporal modeling using Variational Bayes (VB) inference for traffic prediction and missing data imputation. | **NMSE, RMSE, MAE, Predictive Variance**<br>• Datasets: Abilene and GÉANT networks.<br>• Implemented in Python / GPyTorch. | • Provides principled Bayesian uncertainty quantification (critical for SLA risk assessment).<br>• Effectively handles missing sensor and link telemetry through joint spatial-temporal inference without heavy hyperparameter tuning. | • Cubic computational complexity $O(N^3)$ inherent to Gaussian processes restricts scalability to large carrier topologies.<br>• Uses fixed spatial kernels without data-driven similarity graph blending or dynamic load adaptation. |
| **Y. Gui, D. Wang, L. Guan, and M. Zhang (2020)** [6] | **GCN-GRU**: Spectral Graph Convolutional Network combined with Gated Recurrent Units for optical backbone network traffic forecasting. | **MAE, RMSE, MAPE**<br>• Dataset: Real-world optical network traffic traces.<br>• Baselines: LSTM, MLP. | • Demonstrates that combining graph spectral convolutions with GRU temporal tracking captures optical transmission topology, achieving ~98% forecasting accuracy.<br>• Outperforms traditional sequential models. | • Topology is strictly fixed to optical fiber physical spans.<br>• Does not evaluate statistical co-movement across optical channels or examine model robustness under network load regime shifts. |
| **Z. Zhou, H. Zhang, W. Zhu, Z. Fu, and R. Gu (2023)** [7] | **ASTFN**: Attention-Based Spatial-Temporal Fusion Neural Network combining 2D CNN spatial extraction, feature concatenation, and self-attention LSTM for optical fronthaul. | **MAE, RMSE, MAPE**<br>• Dataset: Optical fronthaul / Cloud Radio Access Network (C-RAN) traffic traces. | • Concatenates original traffic matrices with extracted spatial features to preserve raw signal fidelity.<br>• Self-attention mechanism dynamically weights salient historical time windows. | • Spatial extraction is restricted to 2D grid convolutions (CNNs), which cannot naturally represent arbitrary non-Euclidean network topologies.<br>• Lacks graph neural network formulation and load-regime analysis. |
| **M. Altieri, R. Corizzo, and M. Ceci (2024)** [8] | **GAP-LSTM**: Graph-Based Autocorrelation Preserving Networks integrating GCNs, attention-based LSTM, 2D convolutions, and latent memory states. | **MAE, RMSE, SMAPE**<br>• Datasets: Multivariate geo-distributed sensor and traffic networks.<br>• Implemented in PyTorch / PyTorch Geometric. | • Explicitly preserves spatio-temporal autocorrelation structures across geo-distributed nodes via latent memory states.<br>• Demonstrates superior stability and accuracy over conventional ST-GNN baselines. | • High architectural complexity and parameter footprint.<br>• Employs static distance-based spatial graphs rather than empirical traffic similarity graphs; lacks load-regime conditioning. |
| **C. Pandey, V. Tiwari, J. J. P. C. Rodrigues, and D. S. Roy (2024)** [9] | **5GT-GAN-NET**: Generative Adversarial Network with supervised loss combined with Temporal Attention Mechanism Network (TAMNet) for 5G traffic prediction. | **MSE, MAE**<br>• Dataset: 5G mobile cellular traffic datasets.<br>• Baselines: ARIMA, NARNN, LSTM.<br>• Achieved MSE of 0.000346 and MAE of 0.00685. | • Solves severe data scarcity and user privacy constraints in cellular systems by generating high-fidelity synthetic traffic time series.<br>• Improves MSE by 95.45% vs. ARIMA and MAE by 87.31% vs. NARNN. | • Purely focused on temporal data synthesis and prediction; omits topological graph modeling.<br>• Does not account for spatial link dependencies or topological routing constraints. |
| **G. O. Ferreira, C. Ravazzi, F. Dabbene, G. C. Calafiore, and M. Fiore (2023)** [10] | **Survey, Tutorial, and Benchmark**: Comprehensive mathematical tutorial and open-source comparative benchmarking across statistical (ARMA, ARIMA, SARIMA) and deep learning models (RNN, LSTM, GRU, CNN). | **MAE, RMSE, MAPE, Computational Runtime**<br>• Datasets: Multiple public real-world traffic traces.<br>• Provides standardized open-source evaluation suite. | • Establishes unified mathematical foundations and standardized evaluation methodologies.<br>• Rigorously quantifies trade-offs between fitting accuracy, computational overhead, and inference latency. | • Primarily surveys temporal and Euclidean deep learning models; does not provide extensive comparative benchmarking for recent Graph Neural Network architectures or graph-topology design choices. |

---

## 2.3 Research Gap and Justification for This Project’s Approach

### Identified Research Gaps

A systematic synthesis of the literature reveals three interrelated gaps in network traffic forecasting:

1. **Topological Structural Bias vs. Empirical Traffic Correlation**:
   State-of-the-art graph architectures ([1], [2], [3], [6]) presuppose that physical topological adjacency ($A_{\text{phys}}$) is the definitive carrier of spatial dependencies. In IP/MPLS and core transport networks, however, traffic demand is dictated by origin-destination (OD) routing matrices. Two links connected to the same router frequently serve unrelated traffic flows, while two non-adjacent links may experience near-identical load profiles because they form consecutive legs of major transit paths. Enforcing $A_{\text{phys}}$ alone imposes an inductive bias that can introduce structural noise. What remains missing is an empirical evaluation comparing physical graphs against data-driven similarity graphs ($A_{\text{sim}}$)—constructed from training-data Pearson correlation—and convex combinations of the two ($A(\alpha) = \alpha A_{\text{phys}} + (1 - \alpha) A_{\text{sim}}$).

2. **Unvalidated Hypotheses Concerning Adaptive and Mixed Graphs**:
   Several theoretical proposals suggest that mixing physical and similarity graphs or dynamically adapting graph weights to network load conditions should improve forecasting accuracy. However, such claims are rarely tested through strict, leakage-free validation procedures. Prior studies often lack controlled evaluations where mixing weights ($\alpha$) are tuned on separate validation splits across multiple random seeds, risking over-optimistic conclusions about adaptive or hybrid models.

3. **Evaluation Under Network Load Regimes and Distribution Shifts**:
   Backbone networks exhibit non-stationary traffic conditions with distinct load regimes (e.g., off-peak low load vs. peak high load). Standard evaluations compute aggregate error metrics across the entire test set, obscuring model behavior across operational regimes. Furthermore, real-world data splits frequently experience distribution shifts between training and test periods, which can leave certain load regimes unrepresented during testing. Controlled studies must explicitly document these limitations rather than assuming universal generalization.

### Justification for This Project’s Approach

To address these gaps, this project formulates a controlled comparative study using one week of five-minute traffic data from the Abilene backbone network (12 nodes, 30 directed internal links, 1,209 timesteps). Using an input history of 12 timesteps ($H = 12$, 60 minutes) to predict traffic 1 step ahead (5 minutes), the project implements and evaluates four distinct model configurations:

1. **Temporal Baseline ([`GRUBaseline`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L6-L15))**: A 64-unit Gated Recurrent Unit that processes past link-load measurements without spatial graph message passing, representing temporal-only deep learning schemes (analogous to [4]).
2. **Physical-Topology Graph-GRU ([`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32), $\alpha = 1.00$)**: Combines a spatial linear projection over the normalized physical adjacency matrix $A_{\text{phys}}$ (derived from shared router endpoints) with a 32-unit GRU cell, representing the conventional topological GNN paradigm ([1], [3], [6]).
3. **Similarity-Graph Graph-GRU ([`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32), $\alpha = 0.00$)**: Replaces physical connectivity with an empirical similarity graph $A_{\text{sim}}$, where each link is connected to its top-3 most correlated links based on Pearson correlation coefficients computed strictly from training-set traffic loads.
4. **Mixed Graph-GRU ([`GraphGRU`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/src/models.py#L17-L32), $\alpha \in \{0.25, 0.50, 0.75\}$)**: Blends the normalized physical and similarity graphs via a convex combination:
   $$A(\alpha) = \alpha A_{\text{phys}} + (1 - \alpha) A_{\text{sim}}$$

### Rigorous Empirical Selection and Outcome

To investigate whether graph structure should adapt to traffic conditions, the training set was partitioned into three load regimes based on total observed network load:
* **Low Regime**: $\le 7.87\text{ Gbps}$ (33.3rd percentile of training windows)
* **Medium Regime**: $(7.87, 9.20]\text{ Gbps}$ (33.3rd to 66.7th percentile)
* **High Regime**: $> 9.20\text{ Gbps}$ (> 66.7th percentile)

The mixing weight $\alpha$ was evaluated across a discrete grid $\alpha \in \{0.00, 0.25, 0.50, 0.75, 1.00\}$ using validation-set MAE alone, completely isolated from test data, across three random seeds (7, 21, and 42).

**Key Empirical Findings**:
* **Validation Outcome**: Across all three load regimes and all three random seeds, validation MAE increased monotonically as $\alpha$ increased. The physical graph component consistently degraded validation performance. Consequently, the selection policy uniformly selected $\alpha^* = 0.00$ for all regimes and seeds (as documented in [`load_aware_selected_alphas_full.csv`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/load_aware_selected_alphas_full.csv)).
* **Model Classification**: Because $\alpha^* = 0.00$ was selected everywhere, the resulting model is a **static Similarity-only Graph-GRU ($\alpha = 0.00$)**. It is **not adaptive**, and the mixed-graph configuration ($\alpha = 0.50$) did **not** perform best.
* **Test Performance (Mean $\pm$ Std across 3 seeds)**:
  * **Similarity-only Graph-GRU ($\alpha = 0.00$)**: MAE = **$22.28 \pm 0.03\text{ Mbps}$**, RMSE = **$30.68 \pm 0.06\text{ Mbps}$**
  * **Temporal GRU Baseline**: MAE = $22.80 \pm 1.35\text{ Mbps}$, RMSE = $33.25 \pm 2.54\text{ Mbps}$
  * **Fixed 50/50 Mixed Graph-GRU ($\alpha = 0.50$)**: MAE = $22.91 \pm 0.34\text{ Mbps}$, RMSE = $32.72 \pm 0.64\text{ Mbps}$
  * **Physical Graph-GRU ($\alpha = 1.00$)**: MAE = $25.47 \pm 0.14\text{ Mbps}$, RMSE = $36.66 \pm 0.40\text{ Mbps}$

### Framing of Contribution and Experimental Scope

In accordance with objective reporting standards:
1. **Nature of Contribution**: The contribution of this project is characterized as an empirical **graph-design and evaluation choice** within the context of the Abilene network, rather than a globally proven novel method. The experiment provides concrete evidence that empirical link similarity ($A_{\text{sim}}$) is a significantly stronger structural inductive bias than physical router adjacency ($A_{\text{phys}}$) for this network.
2. **Evaluation Scope and Dataset Shift**: Due to an observed load shift between the training and test weeks, the test set maximum load (7.26 Gbps) fell below the training-defined Low/Medium threshold (7.87 Gbps). Consequently, all 392 test windows fell entirely within the Low-load regime; Medium and High test regimes could not be evaluated under the prospective policy.
3. **Retrospective Nuance**: A post-hoc analysis on test-local tertiles revealed that while the Similarity-only model achieved superior overall performance, the GRU baseline attained lower MAE in the test-local High band (19.57 Mbps vs. 23.75 Mbps), demonstrating that similarity dominance is not universally uniform across all subsets.

By detailing these empirical realities, this project provides a transparent, scientifically grounded benchmark that directly addresses the architectural assumptions prevalent in existing literature.

---

## References

```text
[1] D. Andreoletti, S. Troia, F. Musumeci, S. Giordano, G. Maier, and M. Tornatore, 
    "Network Traffic Prediction Based on Diffusion Convolutional Recurrent Neural Networks," 
    in Proc. IEEE Conf. Comput. Commun. Workshops (INFOCOM WKSHPS), Paris, France, 2019, 
    pp. 408–413, doi: 10.1109/INFCOMW.2019.8845132.
    IEEE Xplore: https://ieeexplore.ieee.org/document/8845132/
    Full text: https://re.public.polimi.it/bitstream/11311/1079801/1/PID5815733-pdfexpress.pdf

[2] J. Tao, K. Cao, and T. Liu, 
    "Traffic Matrix Prediction Based on Cross Aggregate GNN," 
    in Proc. IEEE Int. Conf. Dependable, Autonomic Secure Comput., Pervasive Intell. Comput., 
    Cloud Big Data Comput., Cyber Sci. Technol. Congr. (DASC/PiCom/CBDCom/CyberSciTech), 
    Abu Dhabi, United Arab Emirates, 2023, pp. 269–276, doi: 10.1109/DASC/PiCom/CBDCom/Cy59711.2023.10361471.
    IEEE Xplore: https://ieeexplore.ieee.org/document/10361471/
    DOI Link: https://doi.org/10.1109/DASC/PiCom/CBDCom/Cy59711.2023.10361471

[3] Z. Yao, Q. Xu, Y. Chen, Y. Tu, H. Zhang, and Y. Chen, 
    "Internet Traffic Forecasting Using Temporal-Topological Graph Convolutional Networks," 
    in Proc. Int. Joint Conf. Neural Netw. (IJCNN), Shenzhen, China, 2021, pp. 1–8, 
    doi: 10.1109/IJCNN52387.2021.9534096.
    IEEE Xplore: https://ieeexplore.ieee.org/document/9534096/
    DOI Link: https://doi.org/10.1109/IJCNN52387.2021.9534096

[4] A. Azzouni and G. Pujolle, 
    "NeuTM: A Neural Network-Based Framework for Traffic Matrix Prediction in SDN," 
    in Proc. IEEE/IFIP Netw. Oper. Manage. Symp. (NOMS), Taipei, Taiwan, 2018, pp. 1–5, 
    doi: 10.1109/NOMS.2018.8406199.
    IEEE Xplore: https://ieeexplore.ieee.org/document/8406199/
    DOI Link: https://doi.org/10.1109/NOMS.2018.8406199

[5] S. Mehrizi and S. Chatzinotas, 
    "Network Traffic Modeling and Prediction Using Graph Gaussian Processes," 
    IEEE Access, vol. 10, pp. 133276–133285, 2022, doi: 10.1109/ACCESS.2022.3230908.
    IEEE Xplore: https://ieeexplore.ieee.org/document/9992240/
    Full text: https://orbilu.uni.lu/bitstream/10993/53306/1/Network_Traffic_Modeling_and_Prediction_Using_Graph_Gaussian_Processes.pdf

[6] Y. Gui, D. Wang, L. Guan, and M. Zhang, 
    "Optical Network Traffic Prediction Based on Graph Convolutional Neural Networks," 
    in Proc. 25th Opto-Electronics Commun. Conf. (OECC), Taipei, Taiwan, 2020, pp. 1–3, 
    doi: 10.1109/OECC48412.2020.9273638.
    IEEE Xplore: https://ieeexplore.ieee.org/document/9273638/
    DOI Link: https://doi.org/10.1109/OECC48412.2020.9273638

[7] Z. Zhou, H. Zhang, W. Zhu, Z. Fu, and R. Gu, 
    "Spatial-Temporal Prediction Model for Optical Fronthaul Network Traffic Based on ASTFN," 
    in Proc. 9th Int. Conf. Optics Commun. Netw. (ICOCN), Xiamen, China, 2023, pp. 1–3, 
    doi: 10.1109/ICOCN59242.2023.10236095.
    IEEE Xplore: https://ieeexplore.ieee.org/document/10236095/
    DOI Link: https://doi.org/10.1109/ICOCN59242.2023.10236095

[8] M. Altieri, R. Corizzo, and M. Ceci, 
    "GAP-LSTM: Graph-Based Autocorrelation Preserving Networks for Geo-Distributed Forecasting," 
    IEEE Trans. Neural Netw. Learn. Syst., early access, 2024, doi: 10.1109/TNNLS.2024.3398441.
    IEEE Xplore: https://ieeexplore.ieee.org/document/10533686/
    Full text: https://ieeexplore.ieee.org/ielx7/5962385/6104215/10533686.pdf

[9] C. Pandey, V. Tiwari, J. J. P. C. Rodrigues, and D. S. Roy, 
    "5GT-GAN-NET: Internet Traffic Data Forecasting With Supervised Loss Based Synthetic Data Over 5G," 
    IEEE Trans. Mob. Comput., vol. 23, no. 11, pp. 10694–10705, Nov. 2024, 
    doi: 10.1109/TMC.2024.3364655.
    IEEE Xplore: https://ieeexplore.ieee.org/document/10449458/
    DOI Link: https://doi.org/10.1109/TMC.2024.3364655

[10] G. O. Ferreira, C. Ravazzi, F. Dabbene, G. C. Calafiore, and M. Fiore, 
     "Forecasting Network Traffic: A Survey and Tutorial With Open-Source Comparative Evaluation," 
     IEEE Access, vol. 11, pp. 6018–6044, 2023, doi: 10.1109/ACCESS.2023.3236261.
     IEEE Xplore: https://ieeexplore.ieee.org/document/10015152/
     Full text: https://iris.cnr.it/bitstream/20.500.14243/447430/1/prod_476881-doc_195049.pdf
```

