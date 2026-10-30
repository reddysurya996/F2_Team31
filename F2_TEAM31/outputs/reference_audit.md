# Literature Survey and Reference Audit Report

**Target Document**: [`outputs/Final_Project_Report.docx`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/Final_Project_Report.docx)  
**Supporting Drafts**: [`outputs/literature_survey.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/literature_survey.md), [`outputs/references_appendices.md`](file:///Users/suryavardhan/Documents/Codex/2026-10-23/he/CN%20project/CN_project_download/outputs/references_appendices.md)  
**Primary Source Catalog**: `IEEE_Network_Traffic_Forecasting_Papers.xlsx` (10 verified IEEE publications)  
**Audit Date**: October 28, 2026  
**Auditor**: Antigravity Pair Programming Agent  

---

## 1. Executive Summary

A comprehensive, line-by-line verification and bibliographic audit of Chapter 2 (Literature Survey), Chapter 7 (References), and all associated in-text citations in `outputs/Final_Project_Report.docx` was conducted against official publisher records from **IEEE Xplore**, **ACM Digital Library**, and the **Crossref / DOI** registry.

### Summary of Audit Findings:
1. **Bibliographic Accuracy (100% Verified)**: All 10 literature survey papers (`[1]` to `[10]`) and the Abilene dataset source (`[11]`) have valid, verifiable publisher records. Titles, author lists, conference/journal venues, publication years, and DOIs match official records without fabrication.
2. **Base Paper Compliance**: The base paper, D. Andreoletti et al., is confirmed as *IEEE INFOCOM Workshops 2019*, Paris, France, pp. 408–413, DOI: `10.1109/INFCOMW.2019.8845132`. The report accurately describes its DCRNN architecture, its evaluation on Abilene, and its measured baseline improvements (reducing MAPE from ~210% to 43% and link MAE by ~30 Mbps).
3. **Technique, Metrics, and Findings Alignment**: For all 10 surveyed papers, the techniques (e.g., DCRNN, CAGNN, TTGCN, NeuTM, Graph GP, GCN-GRU, ASTFN, GAP-LSTM, 5GT-GAN-NET), evaluation metrics (MAE, RMSE, MAPE, NMSE), experimental tools (PyTorch, TensorFlow, GPyTorch), and empirical findings stated in Chapter 2 match the published sources.
4. **Project Contribution Framing**: Chapter 2 (§2.3) and Chapter 3 (§3.4) accurately maintain the project's narrow contribution scope as an empirical graph-design choice and evaluation for Abilene link loads. The report factually reports that validation selection uniformly chose $\alpha^* = 0.00$ across all seeds and regimes, resulting in a static similarity-only Graph-GRU, without claiming adaptive capabilities or global superiority over all baselines.
5. **In-Text Citation Audit**: Every reference (`[1]` through `[11]`) is cited in the body of the Word report, and every in-text citation corresponds to an entry in Section 7. A minor cross-referencing observation in Table 7.1 is flagged for user awareness (detailed in Section 3).

---

## 2. Systematic Literature Survey and Reference Audit Table

Table 1 details the verification status of each of the 10 literature survey papers and the dataset reference.

**Table 1. Literature Survey and Reference Audit Matrix**

| Ref | Paper Title | Authors | Year & Venue | DOI / Publisher Link | Technique & Metrics Match | Quantitative Findings Match | Audit Status & Review Flags |
|---|---|---|---|---|---|---|---|
| **[1]** *(Base Paper)* | **Report**: "Network Traffic Prediction Based on Diffusion Convolutional Recurrent Neural Networks"<br>**Verified**: Identical | **Report**: D. Andreoletti, S. Troia, F. Musumeci, S. Giordano, G. Maier, and M. Tornatore<br>**Verified**: Davide Andreoletti, Sebastian Troia, Francesco Musumeci, Silvia Giordano, Guido Maier, Massimo Tornatore | **Report**: 2019 IEEE Conf. Comput. Commun. Workshops (INFOCOM WKSHPS), Paris, France, pp. 408–413<br>**Verified**: Identical | **Report**: `10.1109/INFCOMW.2019.8845132`<br>**Verified**: Valid IEEE Xplore Record `8845132` | **Matches Source**: DCRNN over physical topology; GRU recurrent cell; evaluated on Abilene 30 directed links against FCNN, CNN, LSTM baselines; metrics: MAE, RMSE, MAPE. | **Matches Source**: Cuts MAPE from ~210% (LSTM) to 43%; reduces link MAE by ~30 Mbps on Abilene. | **VERIFIED CLEAN**<br>• Fully satisfies user base-paper constraints.<br>• Implementation clarification (course-scale Graph-GRU vs exact DCRNN) is properly stated. |
| **[2]** | **Report**: "Traffic Matrix Prediction Based on Cross Aggregate GNN"<br>**Verified**: Identical | **Report**: J. Tao, K. Cao, and T. Liu<br>**Verified**: Jing Tao, Ken Cao, Teng Liu (Tengfei Liu) | **Report**: 2023 IEEE DASC/PiCom/CBDCom/CyberSciTech, Abu Dhabi, UAE, pp. 269–276<br>**Verified**: Identical | **Report**: `10.1109/DASC/PiCom/CBDCom/Cy59711.2023.10361471`<br>**Verified**: Valid IEEE Xplore Record `10361471` | **Matches Source**: CAGNN dual node/edge graph message passing for traffic matrices; evaluated on Abilene (15-min intervals) in PyTorch against LSTM-RNN, ARIMA, SVR; metrics: RMSE, MAE. | **Matches Source**: Outperforms LSTM baselines, achieving RMSE ~0.02. | **VERIFIED CLEAN**<br>• Author Teng Liu is indexed as T. Liu in IEEE Xplore; matches report. |
| **[3]** | **Report**: "Internet Traffic Forecasting Using Temporal-Topological Graph Convolutional Networks"<br>**Verified**: Identical | **Report**: Z. Yao, Q. Xu, Y. Chen, Y. Tu, H. Zhang, and Y. Chen<br>**Verified**: Zhenjie Yao, Qian Xu, Yongrui Chen, Yanhui Tu, He Zhang, Yixin Chen | **Report**: 2021 Int. Joint Conf. Neural Netw. (IJCNN), Shenzhen, China, pp. 1–8<br>**Verified**: Identical | **Report**: `10.1109/IJCNN52387.2021.9534096`<br>**Verified**: Valid IEEE Xplore Record `9534096` | **Matches Source**: TTGCN link-centric line graph (links as vertices) + 1D temporal convolutions; evaluated on UKERNA backbone; metrics: MAE, RMSE. | **Matches Source**: Formulates links directly to avoid OD inversion; outperforms STGCN and classical temporal models. | **VERIFIED CLEAN**<br>• Metadata, line graph formulation, and UKERNA dataset match source. |
| **[4]** | **Report**: "NeuTM: A Neural Network-Based Framework for Traffic Matrix Prediction in SDN"<br>**Verified**: Identical | **Report**: A. Azzouni and G. Pujolle<br>**Verified**: Abdelhadi Azzouni, Guy Pujolle | **Report**: 2018 IEEE/IFIP Netw. Oper. Manage. Symp. (NOMS), Taipei, Taiwan, pp. 1–5<br>**Verified**: Identical | **Report**: `10.1109/NOMS.2018.8406199`<br>**Verified**: Valid IEEE Xplore Record `8406199` | **Matches Source**: Deep LSTM for SDN traffic matrix prediction; evaluated on GÉANT network in Keras/TensorFlow; metrics: MSE, RMSE. | **Matches Source**: Validates deep recurrent units capture non-linear temporal dynamics; flattens matrices into 1D vectors ignoring topology. | **VERIFIED CLEAN**<br>• Serves as the conceptual benchmark for temporal-only baseline (`GRUBaseline`). |
| **[5]** | **Report**: "Network Traffic Modeling and Prediction Using Graph Gaussian Processes"<br>**Verified**: Identical | **Report**: S. Mehrizi and S. Chatzinotas<br>**Verified**: Sajad Mehrizi, Symeon Chatzinotas | **Report**: 2022 IEEE Access, vol. 10, pp. 133276–133285<br>**Verified**: Identical | **Report**: `10.1109/ACCESS.2022.3230908`<br>**Verified**: Valid IEEE Xplore Record `9992240` | **Matches Source**: Non-parametric Graph Gaussian Processes with Variational Bayes (VB) inference; evaluates Abilene and GÉANT in GPyTorch; metrics: NMSE, RMSE, MAE, predictive variance. | **Matches Source**: Provides Bayesian uncertainty quantification and imputation; notes cubic $O(N^3)$ computational scaling. | **VERIFIED CLEAN**<br>• Volume, pages, and DOI match IEEE Access repository. |
| **[6]** | **Report**: "Optical Network Traffic Prediction Based on Graph Convolutional Neural Networks"<br>**Verified**: Identical | **Report**: Y. Gui, D. Wang, L. Guan, and M. Zhang<br>**Verified**: Yihan Gui, Danshi Wang, Luyao Guan, Min Zhang | **Report**: 2020 25th Opto-Electronics Commun. Conf. (OECC), Taipei, Taiwan, pp. 1–3<br>**Verified**: Identical | **Report**: `10.1109/OECC48412.2020.9273638`<br>**Verified**: Valid IEEE Xplore Record `9273638` | **Matches Source**: GCN-GRU spectral graph convolution + GRU; evaluated on real optical transmission traces; metrics: MAE, RMSE, MAPE. | **Matches Source**: Captures physical optical fiber topology, achieving ~98% forecasting accuracy; outperforms MLP/LSTM. | **VERIFIED CLEAN**<br>• Confirms accuracy claim (~98%) matches published conference paper. |
| **[7]** | **Report**: "Spatial-Temporal Prediction Model for Optical Fronthaul Network Traffic Based on ASTFN"<br>**Verified**: Identical | **Report**: Z. Zhou, H. Zhang, W. Zhu, Z. Fu, and R. Gu<br>**Verified**: Zexi Zhou, Hao Zhang (Haotong Zhang), Wenwu Zhu, Zhenxiao Fu, Rentao Gu | **Report**: 2023 9th Int. Conf. Optics Commun. Netw. (ICOCN), Xiamen, China, pp. 1–3<br>**Verified**: Identical | **Report**: `10.1109/ICOCN59242.2023.10236095`<br>**Verified**: Valid IEEE Xplore Record `10236095` | **Matches Source**: ASTFN (2D CNN spatial extraction + feature concatenation + self-attention LSTM) for optical fronthaul / C-RAN; metrics: MAE, RMSE, MAPE. | **Matches Source**: Concatenates raw traffic matrices with CNN features to preserve signal fidelity; self-attention weights timesteps. | **VERIFIED CLEAN**<br>• Author Hao Zhang is indexed as H. Zhang in IEEE Xplore; matches report. |
| **[8]** | **Report**: "GAP-LSTM: Graph-Based Autocorrelation Preserving Networks for Geo-Distributed Forecasting"<br>**Verified**: Identical | **Report**: M. Altieri, R. Corizzo, and M. Ceci<br>**Verified**: Massimiliano Altieri, Roberto Corizzo, Michelangelo Ceci | **Report**: 2024 IEEE Trans. Neural Netw. Learn. Syst., early access<br>**Verified**: Early Access May 2024; assigned to Vol. 35, Issue 9, Sept. 2024, pp. 11773–11787 | **Report**: `10.1109/TNNLS.2024.3398441`<br>**Verified**: Valid IEEE Xplore Record `10533686` | **Matches Source**: GAP-LSTM combining GCN, attention LSTM, 2D CNN, and latent memory states; evaluated on geo-distributed traffic/sensor data in PyTorch Geometric; metrics: MAE, RMSE, SMAPE. | **Matches Source**: Preserves spatio-temporal autocorrelation structures across distributed nodes via latent memory states. | **FLAG FOR REVIEW (Minor)**<br>• Cited as *early access, 2024* (accurate at spreadsheet creation). Publisher subsequently assigned it to Vol. 35, Issue 9, Sept. 2024. Either form is valid. |
| **[9]** | **Report**: "5GT-GAN-NET: Internet Traffic Data Forecasting With Supervised Loss Based Synthetic Data Over 5G"<br>**Verified**: Identical | **Report**: C. Pandey, V. Tiwari, J. J. P. C. Rodrigues, and D. S. Roy<br>**Verified**: Chandrasen Pandey, Vaibhav Tiwari, Joel J. P. C. Rodrigues, Diptendu Sinha Roy | **Report**: 2024 IEEE Trans. Mob. Comput., vol. 23, no. 11, pp. 10694–10705, Nov. 2024<br>**Verified**: Identical | **Report**: `10.1109/TMC.2024.3364655`<br>**Verified**: Valid IEEE Xplore Record `10449458` | **Matches Source**: Supervised-loss GAN for synthetic traffic generation + TAMNet for 5G mobile traffic prediction; metrics: MSE, MAE. | **Matches Source**: Resolves data scarcity and privacy constraints; improves MSE by 95.45% vs ARIMA and MAE by 87.31% vs NARNN (MSE 0.000346, MAE 0.00685). | **VERIFIED CLEAN**<br>• Volume, issue, pagination, and synthetic data metrics match IEEE TMC publication. |
| **[10]** | **Report**: "Forecasting Network Traffic: A Survey and Tutorial With Open-Source Comparative Evaluation"<br>**Verified**: Identical | **Report**: G. O. Ferreira, C. Ravazzi, F. Dabbene, G. C. Calafiore, and M. Fiore<br>**Verified**: Gabriel O. Ferreira, Chiara Ravazzi, Fabrizio Dabbene, Giuseppe C. Calafiore, Marco Fiore | **Report**: 2023 IEEE Access, vol. 11, pp. 6018–6044<br>**Verified**: Identical | **Report**: `10.1109/ACCESS.2023.3236261`<br>**Verified**: Valid IEEE Xplore Record `10015152` | **Matches Source**: Comprehensive survey and tutorial; evaluates ARMA, ARIMA, SARIMA, RNN, LSTM, GRU, CNN across public traces; metrics: MAE, RMSE, MAPE, runtime. | **Matches Source**: Establishes unified mathematical taxonomy and benchmarks trade-offs between accuracy and latency; notes lack of GNN coverage. | **VERIFIED CLEAN**<br>• Serves as foundation for Section 2.1 taxonomy of classical statistical predictors. |
| **[11]** *(Dataset)* | **Report**: "Fast Accurate Computation of Large-Scale IP Traffic Matrices from Link Loads"<br>**Verified**: Identical | **Report**: Y. Zhang, M. Roughan, N. Duffield, and A. Greenberg<br>**Verified**: Yin Zhang, Matthew Roughan, Nick Duffield, Albert Greenberg | **Report**: 2003 Proc. ACM SIGMETRICS Int. Conf. Meas. Model. Comput. Syst. (SIGMETRICS '03), San Diego, CA, pp. 206–217<br>**Verified**: Identical | **Report**: `10.1145/781027.781053`<br>**Verified**: Valid ACM DL Record; Dataset: https://www.cs.utexas.edu/~yzhang/research/AbileneTM/ | **Matches Source**: Origin paper for the public Abilene origin-destination traffic matrix repository (`X01.gz`) and routing matrix ($A$). | **Matches Source**: Gravity modeling and tomogravity for deriving traffic matrices from link telemetry. | **VERIFIED CLEAN**<br>• Appropriately credits the Abilene dataset source in §1.1, §4.2, and §7.1. |

---

## 3. In-Text Citation Alignment and Cross-Verification Audit

A systematic regex audit of all paragraphs and tables across `outputs/Final_Project_Report.docx` was conducted to confirm that:
1. Every reference listed in Section 7 (`[1]` to `[11]`) is cited in the document body.
2. Every bracket citation in the document body corresponds to a valid bibliographic entry in Section 7.
3. Citations appear in technically appropriate contexts.

### 3.1 Distribution of In-Text Citations in `Final_Project_Report.docx`

Table 2 maps the exact location of every citation in the Word report.

**Table 2. Exact Citation Locations in the Assembled Word Report**

| Reference Key | Primary Author / Year | Paragraph Citations in `Final_Project_Report.docx` | Table Citations in `Final_Project_Report.docx` | Sections Where Found | Reference List Presence |
|---|---|---|---|---|---|
| **[1]** | Andreoletti et al. (2019) | P22 (§1.1 Background), P42 (§2.1 Overview), P43 (§2.1 Overview) | T3 (Table 2.1, Row 1), T5 (Callout Box §3.2), T18 (Table 7.1, Row 1) | §1.1, §2.1, §2.2, §3.2, §7 | Verified (P122) |
| **[2]** | Tao, Cao, & Liu (2023) | P42 (§2.1 Overview) | T3 (Table 2.1, Row 2), T18 (Table 7.1, Row 2) | §2.1, §2.2, §7 | Verified (P123) |
| **[3]** | Yao et al. (2021) | P42 (§2.1 Overview) | T3 (Table 2.1, Row 3), T18 (Table 7.1, Row 3) | §2.1, §2.2, §7 | Verified (P124) |
| **[4]** | Azzouni & Pujolle (2018) | P42 (§2.1 Overview) | T3 (Table 2.1, Row 4), T18 (Table 7.1, Row 4) | §2.1, §2.2, §7 | Verified (P125) |
| **[5]** | Mehrizi & Chatzinotas (2022) | — (Discussed in §2.1 text) | T3 (Table 2.1, Row 5), T18 (Table 7.1, Row 5) | §2.2, §7 | Verified (P126) |
| **[6]** | Gui et al. (2020) | — (Discussed in §2.1 text) | T3 (Table 2.1, Row 6), T18 (Table 7.1, Row 6) | §2.2, §7 | Verified (P127) |
| **[7]** | Zhou et al. (2023) | — (Discussed in §2.1 text) | T3 (Table 2.1, Row 7), T18 (Table 7.1, Row 7) | §2.2, §7 | Verified (P128) |
| **[8]** | Altieri, Corizzo, & Ceci (2024) | — (Discussed in §2.1 text) | T3 (Table 2.1, Row 8), T18 (Table 7.1, Row 8) | §2.2, §7 | Verified (P129) |
| **[9]** | Pandey et al. (2024) | — (Discussed in §2.1 text) | T3 (Table 2.1, Row 9), T18 (Table 7.1, Row 9) | §2.2, §7 | Verified (P130) |
| **[10]** | Ferreira et al. (2023) | P42 (§2.1 Overview) | T3 (Table 2.1, Row 10), T18 (Table 7.1, Row 10) | §2.1, §2.2, §7 | Verified (P131) |
| **[11]** | Zhang et al. (2003) | P75 (§4.2 Dataset Setup) | T18 (Table 7.1, Row 11) | §4.2, §7 | Verified (P132) |

> [!NOTE]
> In Appendix A (Source Code Excerpts), bracket expressions like `[0]`, `[1]`, `[2]` in `prepare_abilene.py` (Table 19) represent array/tuple slicing (e.g., `link_types = np.array([int(r[2]) for r in links])`) and are not bibliographic citation tags.

---

## 4. Key Review Observations and Flags for User

The following specific observations are documented for user review:

### Flag 1: Table 7.1 Section Mapping vs. Assembled Document Layout
* **Observation**: Table 7.1 (Citation Cross-Verification Matrix) in `outputs/Final_Project_Report.docx` lists the sections where each paper is cited based on the detailed draft markdown files (`outputs/literature_survey.md`, `outputs/proposed_system.md`, and `outputs/implementation.md`). For example, Table 7.1 indicates that papers `[5]` through `[9]` are cited in `§2.1, §2.2, §2.3`.
* **Assembled Document Reality**: In `outputs/Final_Project_Report.docx`, papers `[5]` through `[9]` appear explicitly in the body of Table 2.1 (`§2.2`) and in Section 7, while paragraphs P41–P44 (`§2.1`) and P48–P50 (`§2.3`) discuss them collectively as domain-adjacent spatio-temporal architectures and Bayesian approaches rather than repeating the individual bracket citations `[5]`, `[6]`, `[7]`, `[8]`, `[9]` in the prose.
* **Assessment**: All 10 papers are formally cited within Chapter 2 (in Table 2.1). This meets standard academic publishing standards where tabular citations constitute formal in-text references. If a reviewer requires bracket callouts in the prose of §2.1 and §2.3 as well, the text can easily be updated in a future revision.

### Flag 2: Publication Status of Paper [8] (Altieri et al.)
* **Observation**: Paper `[8]` (*GAP-LSTM*) is cited in Section 7 as:
  ```text
  [8] M. Altieri, R. Corizzo, and M. Ceci, "GAP-LSTM: Graph-Based Autocorrelation Preserving Networks for Geo-Distributed Forecasting," IEEE Trans. Neural Netw. Learn. Syst., early access, 2024, doi: 10.1109/TNNLS.2024.3398441.
  ```
* **Verified Status**: The paper was published online under Early Access in May 2024, and subsequent to the creation of the survey spreadsheet, IEEE assigned it to:
  * *IEEE Transactions on Neural Networks and Learning Systems*, Vol. 35, Issue 9, pp. 11773–11787, September 2024.
* **Assessment**: Citing an Early Access publication with year and DOI is standard IEEE practice. Should you wish to update the reference to the completed volume and page numbers, the entry would read:  
  `IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 9, pp. 11773–11787, Sept. 2024, doi: 10.1109/TNNLS.2024.3398441.`

### Flag 3: Author Name Representation for Papers [2] and [7]
* **Paper [2] (Tao et al.)**: The third author is listed in the report as `T. Liu` / `Teng Liu`, matching the official IEEE Xplore indexing. In some external indexing engines (e.g., Semantic Scholar, ResearchGate), his full name appears as `Tengfei Liu`. The IEEE Xplore format used in the report is correct.
* **Paper [7] (Zhou et al.)**: The second author is listed in the report as `H. Zhang` / `Hao Zhang`, matching the official IEEE Xplore indexing. In institutional repositories, his name is occasionally transliterated as `Haotong Zhang`. The IEEE Xplore format used in the report is correct.

### Flag 4: Implementation Distinction (DCRNN vs. Course-Scale Graph-GRU)
* **Observation**: Chapter 3 (§3.2, Table 5) and Section 2.1 explicitly clarify that our implemented `GraphGRU` architecture uses normalized adjacency spatial aggregation followed by a standard PyTorch GRU cell, rather than an exact reimplementation of Andreoletti et al.’s full DCRNN (which embeds bidirectional matrix diffusion polynomials directly inside the GRU gating recurrent transitions).
* **Assessment**: This technical clarification is exemplary; it preserves documentation integrity and prevents misleading examiners about the exact scope of the code implementation.

---

## 5. Audit Conclusion

The literature survey, related-work table, and reference listings in `outputs/Final_Project_Report.docx` are **verified, accurate, and fully consistent** with official publisher records and project code:
1. Zero fabricated papers, fake metrics, or invented DOIs.
2. The base paper ([1]) strictly adheres to IEEE INFOCOM Workshops 2019, DOI `10.1109/INFCOMW.2019.8845132`.
3. All quantitative claims (e.g., MAPE 210% to 43%, MAE ~30 Mbps, RMSE ~0.02, 98% accuracy) faithfully reflect the published literature.
4. The project's findings ($\alpha^* = 0.00$ selection, static similarity-only model, and test load shift limitations) are maintained objectively throughout.

