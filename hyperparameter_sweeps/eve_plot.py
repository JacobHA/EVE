import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def plot_eve_sweep_results(csv_filepath, max_entropy, avg_last_n=3):
    """
    Plots the hyperparameter sweep results specifically for the EVE algorithm.
    """
    # 1. Load the data with the exact EVE headers
    df = pd.read_csv(
        csv_filepath, 
        names=['inner_lr', 'outer_lr', 'matrix_iterations', 'reward_iterations', 
               'empirical_entropies', 'cumulative_matrix_iters'],
        header=0  # Assumes the sweep script wrote the header row
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
    
    df['inner_lr'] = df['inner_lr'].round(3)
    df['outer_lr'] = df['outer_lr'].round(3)

    # Find the absolute best configuration
    best_row = df.loc[df['final_performance'].idxmax()]
    best_inner = best_row['inner_lr']
    best_outer = best_row['outer_lr']
    best_curve_y = best_row['empirical_entropies']
    best_curve_x = best_row['cumulative_matrix_iters']

    # --- Setup the Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # ==========================================
    # PANEL 1: Continuous Heatmap (pcolormesh)
    # ==========================================
    # Pivot the dataframe for the 2D grid
    heatmap_data = df.pivot_table(
        index='outer_lr', 
        columns='inner_lr', 
        values='final_performance', 
        aggfunc='mean'
    )
    
    inner_lrs = heatmap_data.columns.values
    outer_lrs = heatmap_data.index.values
    Z = heatmap_data.values
    
    # Use pcolormesh for continuous axes
    mesh = axes[0].pcolormesh(inner_lrs, outer_lrs, Z, cmap='viridis', shading='auto')
    
    # Log scales for learning rates to fan out the smaller values
    # axes[0].set_xscale('log')
    # axes[0].set_yscale('log')
    
    axes[0].set_xlim(inner_lrs.min(), inner_lrs.max())
    # axes[0].set_ylim(outer_lrs.min(), outer_lrs.max())
    
    cbar1 = fig.colorbar(mesh, ax=axes[0])
    cbar1.set_label(f'Normalized State Entropy (Avg Last {avg_last_n})')
    
    axes[0].set_title('EVE Hyperparameter Interplay', fontsize=14, pad=15)
    axes[0].set_ylabel('Outer PPI Learning Rate', fontsize=12)
    axes[0].set_xlabel('Inner Sweep Learning Rate', fontsize=12)

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
        label=f'Best: Inner LR={best_inner}, Outer LR={best_outer}'
    )
    
    axes[1].set_title('Normalized Entropy vs. Computational Cost', fontsize=14, pad=15)
    axes[1].set_xlabel('Cumulative Matrix Iterations (Inner sweeps)', fontsize=12)
    axes[1].set_ylabel('Normalized State Entropy ($\% \mathcal{H}_{max}$)', fontsize=12)
    
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
states = 38 + 1
max_entropy = np.log(states)
plot_eve_sweep_results('eve_state_sweep_results.csv', max_entropy=max_entropy, avg_last_n=8)