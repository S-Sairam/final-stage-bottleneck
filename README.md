# The Final-Stage Bottleneck: A Systematic Dissection of the R-Learner for Network Causal Inference

[![License: MIT](https://img.shields.io/badge/License-Apache-2.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0))
[![W&B Project](https://img.shields.io/badge/W&B-Project_Dashboard-blue.svg)](https://wandb.ai/pesu-ai-ml/final-stage-bottleneck/)

This repository contains the official implementation and experimental framework for the paper, "The Final-Stage Bottleneck: A Systematic Dissection of the R-Learner for Network Causal Inference."

Our work provides the first systematic empirical study to dissect the components of the R-Learner framework for network causal inference. We find that the primary determinant of success is not the quality of the nuisance models, as the standard DML narrative often emphasizes, but the **inductive bias of the final-stage CATE estimator.**

---

## Core Finding: A "Hierarchy of Needs" for Network Causal Inference

Our comprehensive, multi-seed experiments reveal a clear and actionable hierarchy of modeling priorities. The most critical component, by an order of magnitude, is a graph-aware final-stage model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d72d0b11-1903-4af7-8c2f-9311b26ac759" width="800" />
</p>
<p align="center">
  <em><b>Figure 1:</b> Main results on a Barabási-Albert graph. The Graph R-Learner (GNN+GNN) and its Sanity Check (MLP+GNN) variant, both using a graph-aware final stage, catastrophically outperform all models with a graph-blind final stage.</em>
</p>

### Key Scientific Contributions:

1.  **A Catastrophic Representation Bottleneck:** We prove with overwhelming statistical significance (p < 0.001) that R-Learners with a graph-blind final-stage (e.g., Linear) fail completely, even when paired with powerful GNN nuisance models.
2.  **The R-Learner Framework Advantage:** Our experiments show that the debiased R-Learner framework provides a significant performance advantage over a strong, non-DML graph-aware baseline (a GNN T-Learner).
3.  **A Topology-Dependent Nuisance Bottleneck:** We discover and provide a mechanistic explanation for a subtle second-order effect, showing that a fully graph-aware pipeline (GNN nuisances + GNN final stage) provides a statistically significant performance gain, particularly on graphs with diffuse community structures (ER, SBM).

---

## Reproducing Our Results

The entire experimental suite is orchestrated by `run_experiments.py`, which reads configurations from the `configs/` directory. All results are logged to Weights & Biases.

### 1. Setup

Create and activate a Python virtual environment, then install the required packages.

```bash
# Clone the repository
git clone https://github.com/S-Sairam/final-stage-bottleneck.git
cd final-stage-bottleneck

# Install dependencies
pip install -r requirements.txt
```

### 2. Running a Single Experiment
To reproduce our main result on a Barabási-Albert graph over 30 seeds, run:
(Ensure you are logged into W&B: wandb login)

```bash
python run_experiments.py --config=configs/main_ba_simple_h.yaml

```
### 3. Running the Full Experimental Suite
To reproduce all tables and figures from the paper, run:
```bash
python run_experiments.py --all
```
