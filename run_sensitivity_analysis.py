# run_sensitivity_analyses.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import wandb
from tqdm import tqdm
from typing import Dict, List
import torch.nn.functional as F

# Import all the necessary components from your src package
from src.data import simulate_data
from src.engine import get_nuisance_predictions, estimate_cate_linear, estimate_cate_gnn

# --- Add the GAT Model for the Architecture Sensitivity Test ---
try:
    from torch_geometric.nn import GATConv
    from src.models import GNN # Use the flexible GNN from your models
    GAT_ENABLED = True
except (ImportError, ModuleNotFoundError):
    print("Warning: GATConv could not be imported. Architecture sensitivity test will be skipped.")
    GAT_GNN = None
    GAT_ENABLED = False

# ============================================================
# Core Experiment Logic (Unchanged)
# ============================================================
def run_single_sensitivity_exp(seed: int, config: Dict) -> Dict:
    data_args = {'seed': seed, **config['data_params']}
    sim_data = simulate_data(**data_args)
    if sim_data is None: return {'ablation': float('nan'), 'graphdml': float('nan')}
    X, T, Y, edge_index, true_tau = sim_data
    
    Y_hat_graph, T_hat_graph = get_nuisance_predictions(
        X, T, Y, edge_index, use_gnn=True, 
        model_kwargs=config['model_params'], 
        training_kwargs=config['training_params']
    )
    Y_res_graph, T_res_graph = Y.squeeze() - Y_hat_graph, T.squeeze() - T_hat_graph
    
    tau_hat_ablation = estimate_cate_linear(Y_res_graph, T_res_graph, X)
    mse_ablation = torch.mean((tau_hat_ablation - true_tau)**2).item()
    
    tau_hat_full = estimate_cate_gnn(
        Y_res_graph, T_res_graph, X, edge_index, 
        model_kwargs=config['model_params'], 
        training_kwargs=config['training_params']
    )
    mse_full = torch.mean((tau_hat_full - true_tau)**2).item()
    
    return {'ablation': mse_ablation, 'graphdml': mse_full}


# ============================================================
# Analysis 1: Architecture Sensitivity (Now with W&B)
# ============================================================
def run_arch_sensitivity(config: Dict, num_seeds: int):
    if not GAT_ENABLED: return

    run = wandb.init(project="final-stage-bottleneck-appendix", name="Architecture_Sensitivity", config=config)
    print("\n--- Running Architecture Sensitivity Analysis (GCN vs. GAT) ---")
    results_gcn, results_gat = {'ablation': [], 'graphdml': []}, {'ablation': [], 'graphdml': []}

    for i in tqdm(range(num_seeds), desc="Arch Sensitivity Seeds"):
        config_gcn = yaml.safe_load(yaml.dump(config))
        config_gcn['model_params']['layer_type'] = 'gcn'
        res_gcn = run_single_sensitivity_exp(seed=i, config=config_gcn)
        for key in res_gcn: results_gcn[key].append(res_gcn[key])
        
        config_gat = yaml.safe_load(yaml.dump(config))
        config_gat['model_params']['layer_type'] = 'gat'
        res_gat = run_single_sensitivity_exp(seed=i, config=config_gat)
        for key in res_gat: results_gat[key].append(res_gat[key])

    labels = ['Ablation (GNN+Lin)', 'Graph R-Learner (GNN+GNN)']
    gcn_means = [np.mean(results_gcn['ablation']), np.mean(results_gcn['graphdml'])]
    gat_means = [np.mean(results_gat['ablation']), np.mean(results_gat['graphdml'])]
    
    x = np.arange(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, gcn_means, width, label='GCN (Baseline Arch)', color='skyblue')
    ax.bar(x + width/2, gat_means, width, label='GAT (Alternative Arch)', color='salmon')
    ax.set_ylabel('Mean Squared CATE Error'); ax.set_title('Architecture Sensitivity: GCN vs. GAT')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); fig.tight_layout()
    
    run.log({"architecture_sensitivity_plot": wandb.Image(fig)})
    plt.savefig("sensitivity_architecture.png"); plt.close()
    print("  -> Architecture sensitivity plot saved and logged.")
    run.finish()

# ============================================================
# Analysis 2 & 3: Noise and Sample Size Sweeps (Now with W&B)
# ============================================================
def run_parameter_sweep(config: Dict, param_name: str, param_values: List, num_seeds: int):
    run = wandb.init(project="final-stage-bottleneck-appendix", name=f"{param_name.title()}_Sweep", config={'param': param_name, 'values': param_values})
    print(f"\n--- Running Sensitivity Analysis for '{param_name}' ---")
    
    all_results = {'ablation': [], 'graphdml': []}

    for value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        temp_config = yaml.safe_load(yaml.dump(config))
        if param_name in temp_config['data_params']: temp_config['data_params'][param_name] = value
        else: temp_config[param_name] = value
        
        seed_results = {'ablation': [], 'graphdml': []}
        for i in range(num_seeds):
            res = run_single_sensitivity_exp(seed=i, config=temp_config)
            for key in res: seed_results[key].append(res[key])
        
        # Log the mean result for this parameter value to W&B
        wandb.log({
            f'mean_mse_ablation': np.mean(seed_results['ablation']),
            f'mean_mse_graphdml': np.mean(seed_results['graphdml']),
            param_name: value
        }, step=int(value) if isinstance(value, (int, float)) else None)

        for key in all_results:
            all_results[key].append(np.mean(seed_results[key]))
            
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_values, all_results['ablation'], marker='o', linestyle='--', label='Ablation (GNN+Lin)')
    ax.plot(param_values, all_results['graphdml'], marker='s', linestyle='-', label='Graph R-Learner (GNN+GNN)')
    ax.set_xlabel(param_name.replace('_', ' ').title()); ax.set_ylabel('Mean Squared CATE Error')
    ax.set_title(f'Model Performance vs. {param_name.replace("_", " ").title()}')
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5); fig.tight_layout()

    run.log({f"{param_name}_sweep_plot": wandb.Image(fig)})
    plt.savefig(f"sensitivity_{param_name}.png"); plt.close()
    print(f"  -> {param_name} sensitivity plot saved and logged.")
    run.finish()

# ============================================================
# Main Orchestrator for Sensitivity Analyses
# ============================================================
if __name__ == "__main__":
    with open('configs/main_ba_simple_h.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    NUM_SEEDS_SENSITIVITY = 5

    # --- Analysis 1: Architecture Sensitivity ---
    run_arch_sensitivity(config=base_config, num_seeds=NUM_SEEDS_SENSITIVITY)

    # --- Analysis 2: Signal-to-Noise Ratio ---
    run_parameter_sweep(config=base_config, 
                        param_name='noise_level', 
                        param_values=[0.1, 0.5, 1.0, 2.0, 5.0], 
                        num_seeds=NUM_SEEDS_SENSITIVITY)

    # --- Analysis 3: Sample Efficiency ---
    run_parameter_sweep(config=base_config, 
                        param_name='n', 
                        param_values=[200, 500, 1000, 2000, 5000], 
                        num_seeds=NUM_SEEDS_SENSITIVITY)
    
    print("\nAll sensitivity analyses complete and logged to W&B project 'final-stage-bottleneck-appendix'.")
