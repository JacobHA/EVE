import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

def plot_gamma_sweep(csv_filename):
    # 1. Load the data
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}. Make sure the sweep has finished running!")
        return

    # 2. Parse the stringified lists back into Python arrays
    df['empirical_entropies'] = df['empirical_entropies'].apply(ast.literal_eval)
    df['cumulative_matrix_iters'] = df['cumulative_matrix_iters'].apply(ast.literal_eval)

    # 3. Sort by discount factor so the colors and legend are ordered perfectly
    df = df.sort_values(by='discount').reset_index(drop=True)

    # 4. Set up the plot
    plt.figure(figsize=(12, 7))
    
    # Generate a smooth colormap for the lines based on the number of gammas tested
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    # 5. Plot each run
    for idx, row in df.iterrows():
        gamma = row['discount']
        best_alpha = row['alpha']
        x = row['cumulative_matrix_iters']
        y = row['empirical_entropies']
        
        # We include the best_alpha in the label to see what the sweep actually chose!
        label_str = f"γ = {gamma:.2f} (best α = {best_alpha:.1e})"
        
        plt.plot(x, y, marker='.', linewidth=2, color=colors[idx], label=label_str)

    # 6. Formatting
    plt.title('State Entropy Learning Curves by Discount Factor (γ)', fontsize=14, pad=15)
    plt.xlabel('Cumulative Matrix Iterations (Computational Cost)', fontsize=12)
    plt.ylabel('Empirical State Entropy', fontsize=12)
    
    # Move the legend outside the plot so it doesn't cover the data
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Hyperparameters", frameon=False)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Automatically adjust padding so the external legend isn't cut off
    plt.tight_layout()
    
    # 7. Save and display
    output_img = 'gamma_sweep_entropies.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_img}")
    plt.show()

if __name__ == '__main__':
    # Make sure this matches the path you saved the sweep data to
    file_path = 'windycliff_best_alpha_sweep_results.csv'
    plot_gamma_sweep(file_path)