# The Final-Stage Bottleneck: A Systematic Dissection of the R-Learner for Network Causal Inference

[![arXiv](https://img.shields.io/badge/arXiv:2511.13018-b31b1b.svg)](https://arxiv.org/abs/2511.13018)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![W&B Project](https://img.shields.io/badge/W&B-Project%20Dashboard-blue.svg)](https://wandb.ai/pesu-ai-ml/final-stage-bottleneck/)

This repository contains the official implementation and experimental framework for our paper, "The Final-Stage Bottleneck: A Systematic Dissection of the R-Learner for Network Causal Inference."

Our work provides the first systematic empirical study to dissect the components of the R-Learner framework for network causal inference. We find that the primary determinant of success is not the quality of the nuisance models, as the standard DML narrative often emphasizes, but the **inductive bias of the final-stage CATE estimator.**

---

## Core Finding: A "Hierarchy of Needs" for Network Causal Inference

Our comprehensive, multi-seed experiments reveal a clear and actionable hierarchy of modeling priorities. The most critical component, by an order of magnitude, is a graph-aware final-stage model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d72d0b11-1903-4af7-8c2f-9311b26ac759" width="800" />
</p>
<p align="center">
  <em><b>Figure 1:</b> Main results on a Barabási-Albert graph (30 seeds). The Graph R-Learner (GNN+GNN) and its Sanity Check (MLP+GNN) variant, both using a graph-aware final stage, catastrophically outperform all models with a graph-blind final stage. Error bars denote one standard deviation.</em>
</p>

1.  **(Must-Have) A Graph-Aware Final Stage:** We prove with overwhelming statistical significance (p < 0.001) that a misspecified, graph-blind final stage results in **catastrophic failure** (the "Representation Bottleneck"). This is the primary driver of performance.
2.  **(Should-Have) A Robust DML Framework:** We show that the R-Learner's debiased structure provides a significant performance advantage over strong, non-DML baselines (a GNN T-Learner), validating its theoretical promise.
3.  **(Good-to-Have) Graph-Aware Nuisance Models:** We discover and provide a mechanistic explanation (the "Hub-Periphery Trade-off") for a subtle but significant "nuisance bottleneck," proving that an end-to-end graph-aware pipeline is required for optimal performance

### Key Scientific Contributions:

1.  **A Catastrophic Representation Bottleneck:** We prove with overwhelming statistical significance (p < 0.001) that R-Learners with a graph-blind final-stage fail completely, even when paired with powerful GNN nuisance models.
2.  **The R-Learner Framework Advantage:** We show that the debiased R-Learner framework provides a significant performance advantage over a strong, non-DML graph-aware baseline (a GNN T-Learner).
3.  **A Topology-Dependent Nuisance Bottleneck:** We discover and provide a mechanistic explanation for a subtle second-order effect (the "Hub-Periphery Trade-off"), showing that a fully graph-aware pipeline is most critical on graphs with diffuse community structures.

---

## Reproducing Our Results

The entire experimental suite is orchestrated by `run_experiments.py` (for main results) and `run_sensitivity_analysis.py` (for appendix/robustness checks). All results are logged to Weights & Biases.

### 1. Setup

This project uses Python 3.10+ and PyTorch.

```bash
# 1. Clone the repository
git clone https://github.com/S-Sairam/final-stage-bottleneck.git
cd final-stage-bottleneck

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```
### 2. Running Experiments
Ensure you are logged into your Weights & Biases account before running experiments: wandb login
To reproduce a single key result (Main BA Graph, 30 seeds):
```bash
python run_experiments.py --config=configs/main_ba_simple_h.yaml
```
To reproduce all main paper results:
(Warning: This will run all experimental configurations for 30 seeds each.)
```bash
python run_experiments.py --all
```
To run the sensitivity analyses (e.g., for the appendix):
```bash
python run_sensitivity_analysis.py
```
### Citation
Our paper is currently in preparation for submission. If you find this work and benchmark useful in your research, please star the repository and check back for an updated citation.
```bibtex
@misc{s2025finalstagebottlenecksystematicdissection,
      title={The Final-Stage Bottleneck: A Systematic Dissection of the R-Learner for Network Causal Inference}, 
      author={Sairam S, Sara Girdhar, Shivam Soni},
      year={2025},
      eprint={2511.13018},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.13018}, 
}
```
