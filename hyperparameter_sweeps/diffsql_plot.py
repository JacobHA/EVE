import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def plot_diff_sql_sweep_results(csv_filepath, max_entropy, avg_last_n=3):
    """
    Plots the hyperparameter sweep results for Model-Based Differential SQL.
    """
    # 1. Load the data with the exact Differential SQL headers
    df = pd.read_csv(
        csv_filepath, 
        names=['alpha', 'beta_rho', 'matrix_iterations', 'reward_iterations', 
               'empirical_entropies', 'cumulative_matrix_iters'],
        header=0 
    )
    
    # 2. Parse the string lists and normalize entropy
    df['empirical_entropies'] = df['empirical_entropies'].apply(
        lambda x: np.array(ast.literal_eval(x)) / max_entropy
    )
    df['cumulative_matrix_iters'] = df['cumulative_matrix_iters'].apply(
        lambda x: np.array(ast.literal_eval(x))
    )
    
    # 3. Calculate the stability metric
    df['final_performance'] = df['empirical_entropies'].apply(lambda x: np.mean(x[-avg_last_n:]))
    
    df['alpha'] = df['alpha'].round(3)
    df['beta_rho'] = df['beta_rho'].round(3)

    # Find the absolute best configuration
    best_row = df.loc[df['final_performance'].idxmax()]
    best_alpha = best_row['alpha']
    best_beta = best_row['beta_rho']
    best_curve_y = best_row['empirical_entropies']
    best_curve_x = best_row['cumulative_matrix_iters']

    # --- Setup the Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # ==========================================
    # PANEL 1: Continuous Heatmap (pcolormesh)
    # ==========================================
    # Pivot the dataframe for the 2D grid
    heatmap_data = df.pivot_table(
        index='beta_rho', 
        columns='alpha', 
        values='final_performance', 
        aggfunc='mean'
    )
    
    alphas = heatmap_data.columns.values
    beta_rhos = heatmap_data.index.values
    Z = heatmap_data.values
    
    # Use pcolormesh for continuous axes
    mesh = axes[0].pcolormesh(alphas, beta_rhos, Z, cmap='viridis', shading='auto')
    
    # Log scales help fan out the lower learning rates visually
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    axes[0].set_xlim(alphas.min(), alphas.max())
    axes[0].set_ylim(beta_rhos.min(), beta_rhos.max())
    
    cbar1 = fig.colorbar(mesh, ax=axes[0])
    cbar1.set_label(f'Normalized State Entropy (Avg Last {avg_last_n})')
    
    axes[0].set_title('Diff-SQL Hyperparameter Interplay', fontsize=14, pad=15)
    axes[0].set_ylabel(r'Average Reward Tracking Rate ($\beta_\rho$)', fontsize=12)
    axes[0].set_xlabel(r'Q-Learning Rate ($\alpha$)', fontsize=12)

    # ==========================================
    # PANEL 2: Ragged Learning Curves
    # ==========================================
    norm = plt.Normalize(df['final_performance'].min(), df['final_performance'].max())
    cmap = plt.cm.viridis
    
    # Plot all background curves
    for _, row in df.iterrows():
        color = cmap(norm(row['final_performance']))
        axes[1].plot(
            row['cumulative_matrix_iters'], 
            row['empirical_entropies'], 
            color=color, 
            alpha=0.25, 
            linewidth=1.5
        )
        
    # Plot the BEST curve prominently
    axes[1].plot(
        best_curve_x, 
        best_curve_y, 
        color='red', 
        alpha=1.0, 
        linewidth=3, 
        label=rf'Best: $\alpha$={best_alpha}, $\beta_\rho$={best_beta}'
    )
    
    axes[1].set_title('Normalized Entropy vs. Computational Cost', fontsize=14, pad=15)
    axes[1].set_xlabel('Cumulative Matrix Iterations (DP sweeps)', fontsize=12)
    axes[1].set_ylabel(r'Normalized State Entropy ($\% \mathcal{H}_{max}$)', fontsize=12)
    
    axes[1].set_ylim([0.0, 1.05])
    
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower right', fontsize=11)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar2 = fig.colorbar(sm, ax=axes[1])
    cbar2.set_label('Final Normalized Performance', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

# --- Execution ---
# Replace with the actual max_entropy calculated for your environment
max_entropy = np.log(39)
plot_diff_sql_sweep_results('diff_sql_model_sweep_results.csv', max_entropy=max_entropy)