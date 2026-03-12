import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'eve_state_sweep_results.csv'
df = pd.read_csv(file_path)

# Convert string representation of lists to actual Python lists
df['empirical_entropies'] = df['empirical_entropies'].apply(ast.literal_eval)
df['cumulative_matrix_iters'] = df['cumulative_matrix_iters'].apply(ast.literal_eval)

# Flatten the paired lists into separate rows for plotting
records = []
for _, row in df.iterrows():
    beta = row['beta']
    entropies = row['empirical_entropies']
    iters = row['cumulative_matrix_iters']
    for i, ent in zip(iters, entropies):
        records.append({'beta': beta, 'matrix_iteration': i, 'entropy': ent})

df_exploded = pd.DataFrame(records)
# Max entropy maps to exactly 1. Lower entropies map to values > 1.
max_entropy = df_exploded['entropy'].max()
df_exploded['transformed_entropy'] = df_exploded['entropy']#np.log((df_exploded['entropy'] / max_entropy) + 1e-12)

# Create a pivot table for the heatmap
heatmap_data = df_exploded.pivot(index='beta', columns='matrix_iteration', values='transformed_entropy')
heatmap_data = heatmap_data.sort_index(ascending=False)
# go up to 40 matrix iterations for better visualization
# heatmap_data = heatmap_data.loc[:, heatmap_data.columns <= 40]

# Plot the heatmap 
# set colormap to jet with a custom range to enhance contrast
plt.figure(figsize=(12, 8))
plt.figure(figsize=(10, 5))
sns.heatmap(heatmap_data, cmap='jet', cbar_kws={'label': 'Empirical Entropy'})
plt.title('Empirical Entropy vs Beta and Matrix Iterations')
plt.xlabel('Cumulative Matrix Iterations')
plt.ylabel('Beta')


# Adjust x-ticks to prevent overlap if there are many columns
if len(heatmap_data.columns) > 20:
    plt.xticks(np.arange(0, len(heatmap_data.columns), step=max(1, len(heatmap_data.columns)//10)), 
               heatmap_data.columns[::max(1, len(heatmap_data.columns)//10)], rotation=45)

plt.tight_layout()
plt.savefig('entropy_heatmap.png')