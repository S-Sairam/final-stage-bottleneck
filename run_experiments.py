# run_experiments.py
"""
Main orchestrator for the Graph R-Learner paper experiments.

This script reads experiment configurations from YAML files, runs the full
multi-seed experimental suite for each configuration, generates all result
tables and plots, and logs everything to Weights & Biases for transparent
and reproducible research.

Usage:
    python run_experiments.py --config=configs/main_ba_simple_h.yaml
    python run_experiments.py --all  (to run all configs in the /configs dir)
"""

import yaml
import wandb
import torch
import os
import argparse
from typing import Dict
import matplotlib.pyplot as plt
# Import all the necessary components from your src package
from src.data import simulate_data
from src.engine import (
    get_nuisance_predictions,
    estimate_cate_linear,
    estimate_cate_gnn,
    estimate_cate_tlearner_gnn
)
from src.analysis import (
    generate_report,
    run_two_model_test,
    run_error_analysis,
    run_tsne_visualization,
    run_hub_vs_periphery_analysis
)

def run_single_experiment(seed: int, config: Dict) -> Dict:
    """
    Executes one full experimental run for a single seed and configuration.
    Returns a dictionary of all resulting MSEs and prediction tensors.
    """
    print(f"  Running Seed {seed+1}/{config['num_seeds']}...")
    
    # --- Unpack config parameters with clarity ---
    data_args = config.get('data_params', {})
    model_args = config.get('model_params', {})
    training_args = config.get('training_params', {})
    
    # --- Simulate Data ---
    sim_data = simulate_data(seed=seed, **data_args)
    if sim_data is None: return None
    X, T, Y, edge_index, true_tau = sim_data
    
    # --- R-Learner Nuisance Component ---
    # Pass model and training kwargs separately
    Y_hat_base, T_hat_base = get_nuisance_predictions(
        X, T, Y, use_gnn=False, 
        model_kwargs=model_args, 
        training_kwargs=training_args
    )
    Y_res_base, T_res_base = Y.squeeze() - Y_hat_base, T.squeeze() - T_hat_base
    
    Y_hat_graph, T_hat_graph = get_nuisance_predictions(
        X, T, Y, edge_index, use_gnn=True, 
        model_kwargs=model_args, 
        training_kwargs=training_args
    )
    Y_res_graph, T_res_graph = Y.squeeze() - Y_hat_graph, T.squeeze() - T_hat_graph
    
    # --- CATE Estimation Stage ---
    results = {}
    results['baseline_preds'] = estimate_cate_linear(Y_res_base, T_res_base, X)
    results['ablation_preds'] = estimate_cate_linear(Y_res_graph, T_res_graph, X)
    results['sanity_check_preds'] = estimate_cate_gnn(
        Y_res_base, T_res_base, X, edge_index, 
        model_kwargs=model_args, 
        training_kwargs=training_args
    )
    results['graphdml_preds'] = estimate_cate_gnn(
        Y_res_graph, T_res_graph, X, edge_index, 
        model_kwargs=model_args, 
        training_kwargs=training_args
    )
    
    # --- External T-Learner Baseline ---
    results['tlearner_preds'] = estimate_cate_tlearner_gnn(
        X, T, Y, edge_index, 
        model_kwargs=model_args, 
        training_kwargs=training_args
    )
    
    # --- Calculate MSEs ---
    for key, preds in list(results.items()):
        if '_preds' in key:
            results[key.replace('_preds', '_mse')] = torch.mean((preds - true_tau)**2).item()
            
    return results

def main(args):
    """ Main function to orchestrate the experimental suites. """
    
    if args.all:
        config_files = [os.path.join('configs', f) for f in os.listdir('configs') if f.endswith('.yaml')]
    else:
        config_files = [args.config]

    for config_path in config_files:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # --- Start a new W&B Run for this configuration ---
        run = wandb.init(
            project="final-stage-bottleneck",
            name=config['name'],
            config=config,
            reinit=True
        )

        results_over_seeds = {
            'baseline_mse': [], 'ablation_mse': [], 'sanity_check_mse': [],
            'graphdml_mse': [], 'tlearner_mse': []
        }
        
        config_str = f"Config: {config['name']}"
        print(f"\n{'#'*70}\n# Starting Experiment Suite: {config_str}\n{'#'*70}")
        
        first_seed_preds = None
        for i in range(config['num_seeds']):
            seed_results = run_single_experiment(seed=i, config=config)
            if seed_results is None: continue
            
            if i == 0: first_seed_preds = seed_results
            
            per_seed_metrics = {f'seed_{i+1}_{k}': v for k, v in seed_results.items() if '_mse' in k}
            wandb.log(per_seed_metrics, step=i+1)
            
            for key in results_over_seeds:
                if key in seed_results:
                    results_over_seeds[key].append(seed_results[key])
        
        # --- Generate Final Report & Log to W&B ---
        baseline_mse, graphdml_mse, p_value, final_plot, report_str = generate_report(results_over_seeds, config)
        run.log({"final_results_plot": wandb.Image(final_plot)})
        run.summary['final_report'] = report_str
        plt.close(final_plot)

        # --- Run and Log Diagnostics ---
        diag_result = run_two_model_test(baseline_mse, graphdml_mse, p_value)
        run.summary['diagnostic_result'] = diag_result
        
        # --- Generate and Log Analysis Plots ---
        if first_seed_preds and config['data_params'].get('cate_type') != 'local_x':
            print("\n--- Generating and Logging Analysis Plots ---")
            data_args = {'seed': 0, **config['data_params']}
            sim_data = simulate_data(**data_args)
            if sim_data:
                X, _, _, edge_index, true_tau = sim_data
                
                error_plot = run_error_analysis(first_seed_preds, edge_index, true_tau, config)
                run.log({"error_analysis_plot": wandb.Image(error_plot)})
                plt.close(error_plot)
                
                if 'real_data_name' not in config['data_params'] or not config['data_params']['real_data_name']:
                     tsne_plot = run_tsne_visualization(X, edge_index, true_tau, config)
                     run.log({"tsne_plot": wandb.Image(tsne_plot)})
                     plt.close(tsne_plot)
                if config['name'] == 'Main_Result_BA_Graph':
                    print("\n--- Running Hub vs. Periphery Analysis ---")
                    hub_plot = run_hub_vs_periphery_analysis(first_seed_preds, edge_index, true_tau, config)
                    run.log({"hub_vs_periphery_plot": wandb.Image(hub_plot)})
                    plt.close(hub_plot)                 
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Graph R-Learner paper experiments.")
    parser.add_argument('--config', type=str, help="Path to a single experiment config YAML file.")
    parser.add_argument('--all', action='store_true', help="Flag to run all experiments in the 'configs' directory.")
    
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("Please specify a config file with --config or run all with --all.")
    
    main(args)
