import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def plot_champion_comparison(disc_csv, eve_csv, max_entropy, avg_last_n=3):
    # ==========================================
    # 1. Load and Parse Discounted (SQL) Data
    # ==========================================
    df_disc = pd.read_csv(
        disc_csv, 
        names=['discount', 'alpha', 'matrix_iterations', 'reward_iterations', 
               'empirical_entropies', 'cumulative_matrix_iters'],
        header=0 
    )
    df_disc['empirical_entropies'] = df_disc['empirical_entropies'].apply(
        lambda x: np.array(ast.literal_eval(x)) / max_entropy
    )
    df_disc['cumulative_matrix_iters'] = df_disc['cumulative_matrix_iters'].apply(
        lambda x: np.array(ast.literal_eval(x))
    )
    df_disc['final_perf'] = df_disc['empirical_entropies'].apply(lambda x: np.mean(x[-avg_last_n:]))
    
    # Extract Best Discounted Run
    best_disc = df_disc.loc[df_disc['final_perf'].idxmax()]

    # ==========================================
    # 2. Load and Parse EVE Data
    # ==========================================
    df_eve = pd.read_csv(
        eve_csv, 
        names=['inner_lr', 'outer_lr', 'matrix_iterations', 'reward_iterations', 
               'empirical_entropies', 'cumulative_matrix_iters'],
        header=0 
    )
    df_eve['empirical_entropies'] = df_eve['empirical_entropies'].apply(
        lambda x: np.array(ast.literal_eval(x)) / max_entropy
    )
    df_eve['cumulative_matrix_iters'] = df_eve['cumulative_matrix_iters'].apply(
        lambda x: np.array(ast.literal_eval(x))
    )
    df_eve['final_perf'] = df_eve['empirical_entropies'].apply(lambda x: np.mean(x[-avg_last_n:]))
    
    # Extract Best EVE Run
    best_eve = df_eve.loc[df_eve['final_perf'].idxmax()]

    # ==========================================
    # 3. Plot the Head-to-Head Comparison
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Plot Discounted Champion
    plt.plot(
        best_disc['cumulative_matrix_iters'], 
        best_disc['empirical_entropies'], 
        color='#ff7f0e', # Standard Orange
        linewidth=3,
        marker='s',
        markevery=max(1, len(best_disc['cumulative_matrix_iters']) // 10),
        label=f"Discounted SQL ($\gamma$={best_disc['discount']:.3f}, $\\alpha$={best_disc['alpha']:.3f})"
    )

    # Plot EVE Champion
    plt.plot(
        best_eve['cumulative_matrix_iters'], 
        best_eve['empirical_entropies'], 
        color='#2ca02c', # Standard Green
        linewidth=3,
        marker='o',
        markevery=max(1, len(best_eve['cumulative_matrix_iters']) // 10),
        label=f"EVE (Inner LR={best_eve['inner_lr']:.3f}, Outer LR={best_eve['outer_lr']:.3f})"
    )

    plt.title('Best Run Comparison: EVE vs. Discounted SQL', fontsize=16, pad=15)
    plt.xlabel('Cumulative Matrix Iterations (Computational Cost)', fontsize=13)
    plt.ylabel('Normalized State Entropy (% of Max)', fontsize=13)
    
    plt.ylim([0.0, 1.05])
    
    # If the X-axis spans vastly different orders of magnitude (e.g., EVE takes 1,000 
    # and SQL takes 100,000 iterations), uncomment the log scale below:
    # plt.xscale('log')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()

# --- Example Execution ---
max_entropy = np.log(39)
plot_champion_comparison(
    disc_csv='windycliff_sweep_results.csv', 
    eve_csv='eve_state_sweep_results.csv', 
    max_entropy=max_entropy
)