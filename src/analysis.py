# src/analysis.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from scipy.stats import ttest_rel
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from torch_geometric.utils import degree
from .models import GNN, FinalGNN
from .engine import estimate_cate_gnn, train_gnn_model

def run_error_analysis(results: dict, edge_index, true_tau, config: dict):
    plt.figure(figsize=(12, 5))
    node_degrees = degree(edge_index[0], num_nodes=len(true_tau)).numpy()
    
    ax1 = plt.subplot(1, 2, 1)
    errors_ablation = (results['ablation_preds'] - true_tau).pow(2).numpy()
    ax1.scatter(node_degrees, errors_ablation, alpha=0.3, label='Squared Error')
    ax1.set_title("Error Analysis: Ablation (GNN+Linear)")
    ax1.set_xlabel("Node Degree"); ax1.set_ylabel("Squared CATE Error")
    ax1.set_ylim(bottom=0)

    ax2 = plt.subplot(1, 2, 2)
    errors_graphdml = (results['graphdml_preds'] - true_tau).pow(2).numpy()
    ax2.scatter(node_degrees, errors_graphdml, alpha=0.3, c='salmon')
    ax2.set_title("Error Analysis: Graph R-Learner (GNN+GNN)")
    ax2.set_xlabel("Node Degree"); ax2.set_ylabel("Squared CATE Error")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plot_filename = f"analysis_error_vs_degree_{config['name']}.png"
    plt.savefig(plot_filename)
    print(f"  -> Error analysis plot saved to {plot_filename}")
    return plt.gcf()

def run_tsne_visualization(X, edge_index, true_tau, config: dict):
    model = FinalGNN(X.shape[1])
    dummy_y = torch.randn_like(true_tau); dummy_t = torch.randn_like(true_tau)
    estimate_cate_gnn(dummy_y, dummy_t, X, edge_index, model_obj=model)
    
    with torch.no_grad():
        embeddings = F.relu(model.gnn.convs[0](X, edge_index))
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(embeddings.numpy())
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=true_tau.numpy(), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='True CATE Value')
    plt.title(f"t-SNE of FinalGNN Embeddings ({config['name']})")
    plt.xlabel("t-SNE Dimension 1"); plt.ylabel("t-SNE Dimension 2")
    plot_filename = f"analysis_tsne_{config['name']}.png"
    plt.savefig(plot_filename)
    print(f"  -> t-SNE visualization saved to {plot_filename}")
    return plt.gcf()

def run_two_model_test(baseline_mse: float, graphdml_mse: float, p_value: float):
    print("\n--- Practitioner's Diagnostic (Two-Model Test) ---")
    significant = p_value < 0.05
    large_effect = baseline_mse / (graphdml_mse + 1e-9) > 2.0
    if significant and large_effect:
        print("Result: POSITIVE. The Graph R-Learner significantly and substantially outperforms the blind baseline.")
        print("This is strong evidence for the presence of network-driven causal heterogeneity.")
        return "POSITIVE"
    else:
        print("Result: NEGATIVE. No strong evidence that a graph-aware model is necessary.")
        return "NEGATIVE"

def generate_report(results_over_seeds: dict, config: dict):
    headers = ["Method", "Mean MSE", "Std Dev MSE"]
    table, means, stds = [], {}, {}
    # Use a fixed order for the table
    method_order = ['baseline_mse', 'ablation_mse', 'sanity_check_mse', 'tlearner_mse', 'graphdml_mse']
    for key in method_order:
        if key in results_over_seeds:
            mses = results_over_seeds[key]
            means[key] = np.mean(mses)
            stds[key] = np.std(mses)
            table.append([key.replace('_mse','').upper(), f"{means[key]:.4f}", f"{stds[key]:.4f}"])
            
    config_str = f"Config: {config['name']}"
    report_str = f"\n\n{'='*70}\nRESULTS FOR {config_str}\n{'='*70}\n"
    report_str += tabulate(table, headers=headers, tablefmt="grid") + "\n"
    
    report_str += "\n--- Significance Tests (vs. Graph R-Learner) ---\n"
    _, p_abl = ttest_rel(results_over_seeds['graphdml_mse'], results_over_seeds['ablation_mse'])
    report_str += f"vs. Ablation (GNN+Lin): p-value = {p_abl:.2e} {'(Significant)' if p_abl < 0.05 else ''}\n"
    _, p_san = ttest_rel(results_over_seeds['graphdml_mse'], results_over_seeds['sanity_check_mse'])
    report_str += f"vs. Sanity (MLP+GNN):  p-value = {p_san:.2e} {'(Significant)' if p_san < 0.05 else ''}\n"
    if 'tlearner_mse' in results_over_seeds:
        _, p_tlearner = ttest_rel(results_over_seeds['graphdml_mse'], results_over_seeds['tlearner_mse'])
        report_str += f"vs. GNN T-Learner:    p-value = {p_tlearner:.2e} {'(Significant)' if p_tlearner < 0.05 else ''}\n"

    print(report_str) # Print to console
    
    # Plotting
    plot_labels = [
        'Baseline\n(MLP+Lin)', 'Ablation\n(GNN+Lin)', 'Sanity Check\n(MLP+GNN)',
        'GNN T-Learner\n(External)', 'Graph R-Learner\n(GNN+GNN)'
    ]
    mean_mses = [means[k] for k in method_order]
    std_devs = [stds[k] for k in method_order]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_labels, mean_mses, yerr=std_devs, capsize=5, color='skyblue')
    bars[-1].set_color('salmon') # Highlight Graph R-Learner
    plt.ylabel("Mean Squared Error in CATE Estimation (Lower is Better)")
    plt.title(f"End-to-End Graph Awareness is Necessary\n({config['name']})")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plot_filename = f"results_{config['name']}.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")

    return means['baseline_mse'], means['graphdml_mse'], p_abl, plt.gcf(), report_str
