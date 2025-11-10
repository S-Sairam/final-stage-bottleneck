# The Final-Stage Bottleneck: A Systematic Dissection of the Graph R-Learner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![W&B Project](https://img.shields.io/badge/W&B-Project_Dashboard-blue.svg)](https://wandb.ai/pesu-ai-ml/final-stage-bottleneck/)

This repository contains the official implementation and experimental framework for the paper, "The Final-Stage Bottleneck: A Systematic Dissection of the R-Learner for Network Causal Inference."

Our work provides the first systematic empirical study to dissect the components of the R-Learner framework for network causal inference. We find that the primary determinant of success is not the quality of the nuisance models, as the standard DML narrative often emphasizes, but the **inductive bias of the final-stage CATE estimator.**

---

## The Core Finding: A Hierarchy of Needs

Our comprehensive, multi-seed experiments reveal a clear "hierarchy of needs" for accurate estimation. The most critical component, by an order of magnitude, is a graph-aware final-stage model.

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/d72d0b11-1903-4af7-8c2f-9311b26ac759" />
 

### Key Findings:

1.  **A Catastrophic Representation Bottleneck:** R-Learners with a graph-blind final-stage (e.g., Linear) fail completely, even when paired with powerful GNN nuisance models.
2.  **The R-Learner Framework Advantage:** The debiased R-Learner framework provides a significant performance advantage over a strong, non-DML graph-aware baseline (a GNN T-Learner).
3.  **A Topology-Dependent Nuisance Bottleneck:** A fully graph-aware pipeline (GNN nuisances + GNN final stage) provides a final, statistically significant performance gain, particularly on graphs with diffuse community structures (ER, SBM).

---

## Reproducing Our Results

The entire experimental suite is orchestrated by `run_experiments.py`, which reads configurations from the `configs/` directory.

### 1. Setup

Create and activate a Python virtual environment, then install the required packages.

```bash
# Clone the repository
git clone https://github.com/S-Sairam/final-stage-bottleneck.git
cd final-stage-bottleneck
```

# Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Running a Single Experiment
To reproduce our main result (on a Barabási-Albert graph over 10 seeds), run:
```bash
# Make sure you are logged into W&B: `wandb login`
python3 run_experiments.py --config=configs/main_ba_simple_h.yaml
```
_(Note: Change num_seeds in the YAML file to 10 for a quicker run, or 30 for the full paper result.)_
### 3. Running the Full Experimental Suite

To reproduce all tables and figures from the paper, run:

```bash
python3 run_experiments.py --all
```
_(Warning: This will run dozens of multi-seed experiments.)_
