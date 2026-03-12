import os
import csv
import numpy as np

# User-specific imports
from frozen_lake_env import ModifiedFrozenLake, MAPS
from utils import compute_policy_induced_distribution
from utils import get_dynamics_and_rewards, get_mdp_transition_matrix
from uv_agent_utils import get_empirical_entropy
from visualization import plot_dist

# map_name = '11x11snake' 
map_name = 'windycliff'

def evaluate_eve_beta1(run_id, name=map_name):
    """
    Executes a single run of the EVE State-Entropy algorithm at beta=1.0 
    with a random positive initialization.
    """
    beta = 1.0
    
    # Set seed for reproducibility across runs
    np.random.seed(run_id)
    
    # 1. Environment Setup
    desc = np.array(MAPS[name], dtype='c')
    desc[desc == b'W'] = b'H'
    desc[desc == b'G'] = b'F'
    
    env = ModifiedFrozenLake(
        n_action=4, desc=desc, step_penalization=0, max_reward=0, 
        min_reward=-1, never_done=True, cyclic_mode=True, 
        goal_attractor=0., slippery=0
    )

    dynamics, _ = get_dynamics_and_rewards(env)
    n_states, SA = dynamics.shape
    n_actions = SA // n_states
    
    prior_policy = np.ones((n_states, n_actions)) / n_actions
    
    # --- FIX: Random positive initialization for u ---
    # Using [0.1, 1.0] ensures it is strictly positive and avoids instability
    u = np.random.uniform(low=0.1, high=1.0, size=(n_states * n_actions, 1))
    u /= np.sum(u) # Normalize to start as a valid distribution
    
    power = beta / (1 + beta)
    
    total_matrix_iterations = 0
    empirical_entropies = []
    cumulative_matrix_iters = []
    
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    P_dense = P.toarray()  # Convert sparse matrix to dense array
                
    # --- Inner Fixed-Point Sweep ---
    eval_freq = 1000
    for ustep in range(10000):
        # EVE Update
        x_ = ((u.T @ P_dense).T)**((1-beta)/beta) * u**(-1/beta)
        u_target = np.power(( (u.T @ P_dense).T)**(1/beta) / (P_dense @ x_), power)
        u = u_target
                    
        total_matrix_iterations += 1
            
        # --- Vectorized Policy Extraction ---       
        optimal_policy = u.reshape(n_states, n_actions) * prior_policy
        optimal_policy /= np.sum(optimal_policy, axis=1, keepdims=True)
        
        # --- Evaluation ---
        if ustep % eval_freq == 0: 
            s_entropy, _, _ = get_empirical_entropy(env, optimal_policy, n_episodes=10, max_steps=500, state_action=True)
            
            empirical_entropies.append(round(s_entropy, 4))
            cumulative_matrix_iters.append(total_matrix_iterations)
            print(f"Run {run_id} | Iter {total_matrix_iterations}: Entropy = {s_entropy:.4f}")
            # update the prior policy:
            prior_policy = optimal_policy.copy()
            P = get_mdp_transition_matrix(dynamics, prior_policy)
            P_dense = P.toarray()  # Update dense version for next iteration
        
    s_dist = compute_policy_induced_distribution(dynamics, optimal_policy, steps=100)
    plot_dist(desc, optimal_policy, s_dist, filename=f'./results/{name}/eve_policy_distribution_{run_id}.png', normalize=True, figsize=(8, 4))

    return {
        'run_id': run_id,
        'beta': beta,
        'matrix_iterations': total_matrix_iterations,
        'reward_iterations': total_matrix_iterations, 
        'empirical_entropies': empirical_entropies,
        'cumulative_matrix_iters': cumulative_matrix_iters
    }

if __name__ == '__main__':
    output_dir = './results/hyperparameter_sweeps'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f'{map_name}_eve_beta1_results.csv')
    
    num_runs = 5
    all_results = []
    
    print(f"Starting {num_runs} EVE evaluations for beta=1.0 with random initializations...")
    
    for run_idx in range(num_runs):
        print(f"\n--- Starting Run {run_idx} ---")
        result = evaluate_eve_beta1(run_id=run_idx)
        all_results.append(result)
    
    print(f"\nSweep complete. Writing data to {csv_filename}...")
    
    # Added 'run_id' to the headers
    headers = ['run_id', 'beta', 'matrix_iterations', 'reward_iterations', 
               'empirical_entropies', 'cumulative_matrix_iters']
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results) # Write all 5 runs
        
    print("Data successfully saved.")