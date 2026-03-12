import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

map_name = 'windycliff'  # Ensure this matches the map used in your experiments

map_to_entropies = {
    'windycliff': {'uniform_entropy': 4.2251,
                   'max_entropy': np.log(39*4)},  # Pre-computed max entropy
    '11x11snake': {'uniform_entropy': 3.6855,
                   'max_entropy': 4.406719247264253}, 
}

def plot_all_algorithms(map_name=map_name):
    softq_file = f'{map_name}_discount_variance_results.csv'
    eve_file = f'{map_name}_eve_beta1_results.csv'
    diff_softq_file = f'{map_name}_differential_variance_results.csv'
    hazan_file = f'{map_name}_hazan_variance_results.csv'

    plt.figure(figsize=(12, 7))

    # --- 1. Plot EVE Baseline (Mean + StdErr) ---
    try:
        df_eve = pd.read_csv(eve_file)
        
        all_entropies = df_eve['empirical_entropies'].apply(ast.literal_eval).tolist()
        all_iters = df_eve['cumulative_matrix_iters'].apply(ast.literal_eval).tolist()
        
        entropies_arr = np.array(all_entropies)
        
        # Calculate Mean and Standard Error across the runs (axis=0)
        mean_entropies = np.mean(entropies_arr, axis=0)[:-1]
        stderr_entropies = np.std(entropies_arr, axis=0)[:-1]
        
        eve_iters = np.array(all_iters[0])[:-1]
        
        plt.plot(eve_iters, mean_entropies, 
                 linewidth=3.5, color='red', linestyle='-', zorder=10,
                 label="EVE")
        
        plt.fill_between(eve_iters, 
                         mean_entropies - stderr_entropies, 
                         mean_entropies + stderr_entropies, 
                         color='red', alpha=0.2, zorder=9)
                         
    except FileNotFoundError:
        print(f"Warning: Could not find {eve_file}.")

    # --- 2. Plot Soft Q-Learning Gamma Sweep (Mean + StdErr) ---
    try:
        df_softq = pd.read_csv(softq_file)
        df_softq['empirical_entropies'] = df_softq['empirical_entropies'].apply(ast.literal_eval)
        df_softq['cumulative_matrix_iters'] = df_softq['cumulative_matrix_iters'].apply(ast.literal_eval)
        
        unique_gammas = sorted(df_softq['discount'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_gammas)))

        for idx, gamma in enumerate(unique_gammas):
            df_gamma = df_softq[df_softq['discount'] == gamma]
            
            all_entropies = df_gamma['empirical_entropies'].tolist()
            entropies_arr = np.array(all_entropies)
            
            mean_entropies = np.mean(entropies_arr, axis=0)
            stderr_entropies = np.std(entropies_arr, axis=0)
            
            x_iters = np.array(df_gamma['cumulative_matrix_iters'].iloc[0])
            
            plt.plot(x_iters, mean_entropies, 
                     linewidth=2.0, alpha=0.9, color=colors[idx], 
                     label=f"Soft-Q (γ = {gamma:.2f})")
            
            plt.fill_between(x_iters, 
                             mean_entropies - stderr_entropies, 
                             mean_entropies + stderr_entropies, 
                             color=colors[idx], alpha=0.2, zorder=8)
                             
    except FileNotFoundError:
        print(f"Warning: Could not find {softq_file}.")

    # --- 3. Plot Differential Soft Q-Learning (Mean + StdErr) ---
    try:
        df_diff = pd.read_csv(diff_softq_file)
        
        all_diff_entropies = df_diff['empirical_entropies'].apply(ast.literal_eval).tolist()
        diff_entropies_arr = np.array(all_diff_entropies)
        
        mean_diff = np.mean(diff_entropies_arr, axis=0)
        stderr_diff = np.std(diff_entropies_arr, axis=0)
        
        diff_iters = np.array(ast.literal_eval(df_diff['cumulative_matrix_iters'].iloc[0]))
        
        plt.plot(diff_iters, mean_diff, 
                 linewidth=3.0, color='dodgerblue', linestyle='-', zorder=9,
                 label=f"Differential Soft-Q")
                 
        plt.fill_between(diff_iters, 
                         mean_diff - stderr_diff, 
                         mean_diff + stderr_diff, 
                         color='dodgerblue', alpha=0.2, zorder=8)
                         
    except FileNotFoundError:
        print(f"Warning: Could not find {diff_softq_file}.")

    # --- 4. Plot Hazan MaxEnt (Mean + StdErr) ---
    try:
        df_hazan = pd.read_csv(hazan_file)
        
        all_hazan_entropies = df_hazan['empirical_entropies'].apply(ast.literal_eval).tolist()
        hazan_entropies_arr = np.array(all_hazan_entropies)
        
        mean_hazan = np.mean(hazan_entropies_arr, axis=0)
        stderr_hazan = np.std(hazan_entropies_arr, axis=0) 
        
        hazan_iters = np.array(ast.literal_eval(df_hazan['cumulative_matrix_iters'].iloc[0]))
        
        plt.plot(hazan_iters, mean_hazan,
                 linewidth=3.0, color='darkorange', linestyle='-', zorder=9,
                 label=f"MaxEnt")
                 
        plt.fill_between(hazan_iters, 
                         mean_hazan - stderr_hazan, 
                         mean_hazan + stderr_hazan, 
                         color='darkorange', alpha=0.2, zorder=8)
                         
    except FileNotFoundError:
        print(f"Warning: Could not find {hazan_file}.")

    # --- 5. Plot Baselines & Annotations ---
    ax = plt.gca() 
    
    uniform_entropy = map_to_entropies[map_name]['uniform_entropy']
    max_entropy = map_to_entropies[map_name]['max_entropy']
    
    plt.axhline(max_entropy, color='gray', linestyle=':', linewidth=1.5)
    plt.text(0.97, max_entropy + 0.02, 'Theoretical Max. Entropy', 
             transform=ax.get_yaxis_transform(), 
             horizontalalignment='right', 
             verticalalignment='bottom', 
             color='black', fontsize=12, fontweight='bold')

    plt.axhline(uniform_entropy, color='black', linestyle='-.', linewidth=1.5)
    plt.text(0.97, uniform_entropy + 0.02, "Uniform Policy's Entropy", 
             transform=ax.get_yaxis_transform(), 
             horizontalalignment='right', 
             verticalalignment='bottom', 
             color='black', fontsize=12, fontweight='bold')

    # --- 6. Add the Environment Inset ---
    try:
        img = mpimg.imread('cliff_desc.png')
        inset_ax = ax.inset_axes([0.32, 0.08, 0.4, 0.3]) 
        inset_ax.imshow(img)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2.0) 
    except FileNotFoundError:
        print("Warning: Could not find 'cliff_desc.png' for the inset.")

    # --- 7. Formatting ---
    current_ymin, current_ymax = plt.ylim()
    plt.ylim(current_ymin, max(current_ymax, max_entropy + 0.1))
    
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Empirical State-Action Entropy', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.legend(loc='lower right', frameon=True, fontsize=13, borderpad=1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('algorithm_comparison_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to algorithm_comparison_plot.png")
    plt.show()

if __name__ == '__main__':
    plot_all_algorithms()