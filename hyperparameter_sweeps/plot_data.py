import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def plot_sweep_results(csv_filepath, max_entropy, avg_last_n=3):
    """
    Plots the hyperparameter sweep results.
    
    Args:
        csv_filepath: Path to the generated CSV file.
        max_entropy: The theoretical maximum entropy for the environment (to normalize against).
        avg_last_n: How many of the final iterations to average for the stability metric.
    """
    # 1. Load the data
    df = pd.read_csv(
        csv_filepath, 
        names=['discount', 'alpha', 'matrix_iterations', 'reward_iterations', 'empirical_entropies', 'cumulative_matrix_iters'],
        header=0 
    )
    
    # 2. Parse the string representation of the lists and NORMALIZE by max_entropy
    df['empirical_entropies'] = df['empirical_entropies'].apply(
        lambda x: np.array(ast.literal_eval(x)) #/ max_entropy
    )
    
    df['cumulative_matrix_iters'] = df['cumulative_matrix_iters'].apply(
        lambda x: np.array(ast.literal_eval(x))
    )
    
    # 3. Calculate the stability metric using the exposed avg_last_n parameter
    df['final_performance'] = df['empirical_entropies'].apply(lambda x: np.mean(x[-avg_last_n:]))
    
    # Round hyperparameters for clean heatmap axis labels
    df['discount'] = df['discount'].round(3)
    df['alpha'] = df['alpha'].round(3)


    # Find the absolute best configuration
    best_row = df.loc[df['final_performance'].idxmax()]
    best_discount = best_row['discount']
    best_alpha = best_row['alpha']
    best_curve_y = best_row['empirical_entropies']
    best_curve_x = best_row['cumulative_matrix_iters'] # Ragged X for the best curve

    # --- Setup the Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    
   # ==========================================
    # PANEL 1: Continuous Heatmap (pcolormesh)
    # ==========================================
    # Pivot the dataframe to create a 2D grid 
    heatmap_data = df.pivot_table(
        index='discount', 
        columns='alpha', 
        values='final_performance', 
        aggfunc='mean'
    )

    
    # Extract the actual numerical values for the axes
    alphas = heatmap_data.columns.values
    gammas = heatmap_data.index.values
    Z = heatmap_data.values
    
    # Create the continuous mesh plot
    # shading='nearest' or 'auto' makes it look like a standard heatmap grid
    mesh = axes[0].pcolormesh(alphas, gammas, Z, cmap='viridis', shading='auto')
    # axes[0].set_xscale('log')
    # axes[0].set_xlim(alphas.min(), alphas.max())
    
    # Add the colorbar specifically for the mesh
    cbar = fig.colorbar(mesh, ax=axes[0])
    cbar.set_label(f'Normalized Entropy (Avg Last {avg_last_n})')
    
    axes[0].set_title('Hyperparameter Interplay', fontsize=14, pad=15)
    axes[0].set_ylabel('Discount Factor ($\gamma$)', fontsize=12)
    axes[0].set_xlabel('Learning Rate ($\\alpha$)', fontsize=12)

    # ==========================================
    # PANEL 2: Ragged Learning Curves
    # ==========================================
    norm = plt.Normalize(df['final_performance'].min(), df['final_performance'].max())
    cmap = plt.cm.viridis
    
    # Plot all background curves with their specific ragged X-axis
    for _, row in df.iterrows():
        color = cmap(norm(row['final_performance']))
        axes[1].plot(
            row['cumulative_matrix_iters'], # RAGGED X-AXIS
            row['empirical_entropies'], 
            color=color, 
            alpha=0.2, 
            linewidth=1.5
        )
        
    # Plot the BEST curve prominently
    axes[1].plot(
        best_curve_x, 
        best_curve_y, 
        color='red', 
        alpha=1.0, 
        linewidth=3, 
        label=f'Best: gamma={best_discount}, alpha={best_alpha}'
    )
    
    axes[1].set_title('Normalized Entropy vs. Computational Cost', fontsize=14, pad=15)
    axes[1].set_xlabel('Cumulative Matrix Iterations (Soft Q-learning sweeps)', fontsize=12)
    axes[1].set_ylabel('Normalized State-Action Entropy (% H_max)', fontsize=12)
    
    # axes[1].set_ylim([0.0, 1.05])
    
    # Optional: If the x-axis spans massive orders of magnitude, uncomment this:
    # axes[1].set_xscale('log') 
    
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower right', fontsize=11)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar2 = fig.colorbar(sm, ax=axes[1])
    cbar2.set_label('Final Normalized Performance', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

# --- Example Execution ---
# Replace max_entropy with the actual calculated value from your environment
n_visitable_states = 38
theoretical_max_entropy = np.log(n_visitable_states)# + np.log(n_actions)
map_name = '11x11dzigzag'
plot_sweep_results(f'{map_name}_sweep_results.csv', max_entropy=theoretical_max_entropy, avg_last_n=5)