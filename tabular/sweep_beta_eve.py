import os
import csv
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# User-specific imports
from frozen_lake_env import ModifiedFrozenLake, MAPS
from utils import get_dynamics_and_rewards, get_mdp_transition_matrix
from uv_agent_utils import get_empirical_entropy

map_name = '11x11dzigzag'
def evaluate_eve_hyperparameters(beta, name=map_name):
    """
    Worker function: Executes the EVE State-Entropy algorithm exactly 
    matching the working sequential math.
    """
    # 1. Thread-safe Environment Setup
    desc = np.array(MAPS[name], dtype='c')
    desc[desc == b'W'] = b'H'
    desc[desc == b'G'] = b'F'
    mask = desc.flatten() != b'F'
    
    env = ModifiedFrozenLake(
        n_action=4, desc=desc, step_penalization=0, max_reward=0, 
        min_reward=-1, never_done=True, cyclic_mode=True, 
        goal_attractor=0., slippery=0
    )

    dynamics, _ = get_dynamics_and_rewards(env)
    n_states, SA = dynamics.shape
    n_actions = SA // n_states
    
    prior_policy = np.ones((n_states, n_actions)) / n_actions
    u = np.ones((n_states * n_actions,1 )) / n_states
    
    # User specified beta=1.0 fixed
    
    power = beta / (1 + beta)
    
    total_matrix_iterations = 0
    total_reward_iterations = 0
    empirical_entropies = []
    cumulative_matrix_iters = []
    power = beta/(1+beta)

    
    # --- BULLETPROOF PS CONSTRUCTION ---
    # Matches the working code exactly using the imported utility
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Ps = P.toarray().reshape(n_states, n_actions, n_states, n_actions).sum(axis=1).sum(axis=2)
                
    # --- Inner Fixed-Point Sweep (State Entropy) ---
    for ustep in range(1000):
        x_ = ((u.T @ P.A).T)**((1-beta)/beta) * u**(-1/beta)
        u_target = np.power(( (u.T @ P.A).T)**(1/beta) / (P.A @ x_), power)
        # Apply Inner Learning Rate (Damping)
        u = u_target
                    
        total_matrix_iterations += 1
        
        total_reward_iterations += 1
        cumulative_matrix_iters.append(total_matrix_iterations)
            
        # --- Vectorized Policy Extraction ---       
        optimal_policy = u.reshape(n_states, n_actions) * prior_policy
        optimal_policy /= np.sum(optimal_policy, axis=1, keepdims=True)
        if ustep % 10 == 0: # Check for convergence every 10 iterations
            # --- Evaluation ---
            s_entropy, _, _ = get_empirical_entropy(env, optimal_policy, n_episodes=10, max_steps=1000, state_action=True)
            empirical_entropies.append(round(s_entropy, 4))
        
    return {
        'beta': beta,
        'matrix_iterations': total_matrix_iterations,
        'reward_iterations': total_reward_iterations,
        'empirical_entropies': empirical_entropies,
        'cumulative_matrix_iters': cumulative_matrix_iters
    }

if __name__ == '__main__':
    # === Sweep Configuration ===
    output_dir = './results/hyperparameter_sweeps'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, 'eve_state_sweep_results.csv')
    
    # Define the 2D search grid
    betas = np.linspace(0.99, 15.0, 30)
    
    hyperparameters = betas
    print(f"Starting EVE parallel sweep over {len(hyperparameters)} combinations using 8 cores...")
    
    # === Execute Parallel Sweep ===
    results = Parallel(n_jobs=15)(
        delayed(evaluate_eve_hyperparameters)(i) for i in tqdm(hyperparameters)
    )
    
    # === Save to CSV ===
    print(f"Sweep complete. Writing data to {csv_filename}...")
    headers = ['beta', 'matrix_iterations', 'reward_iterations', 
               'empirical_entropies', 'cumulative_matrix_iters']
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print("Data successfully saved.")