"""
Parallel model-based entropy-maximization sweep for Hazan's MaxEnt Algorithm.
Uses an exact standard MDP planner and a convex policy mixture update (Frank-Wolfe).
"""

import os
import csv
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from visualization import plot_dist

# User-specific imports
from frozen_lake_env import ModifiedFrozenLake, MAPS
from utils import get_mdp_transition_matrix, get_dynamics_and_rewards, compute_policy_induced_distribution
from uv_agent_utils import get_empirical_entropy

map_name = 'windycliff'
GAMMA=0.99
def exact_hard_average_reward_vi(dynamics, rewards, max_iter=20):
    """
    APPROXPLAN: Standard Value Iteration to find the optimal deterministic policy
    for the current reward function.
    """
    nS, nA = rewards.shape
    V = np.zeros(nS)
    
    # Run exact hard value iteration
    for _ in range(max_iter):
        expected_V = (dynamics @ V).reshape(nS, nA)
        Q = rewards + GAMMA*expected_V
        # V_new = np.max(Q, axis=1)
        # V = V_new - np.mean(V_new) # Center to prevent divergence
        
    expected_V = (dynamics @ V).reshape(nS, nA)
    Q = rewards + expected_V
    
    # Extract deterministic policy (handling ties uniformly)
    Qmax = np.max(Q, axis=1, keepdims=True)
    is_max = np.isclose(Q, Qmax, atol=1e-8)
    optimal_policy = is_max / is_max.sum(axis=1, keepdims=True)
    
    return optimal_policy, max_iter

def evaluate_hazan_hyperparameters(run_id, eta_lr, name=map_name, state_action=True):
    """
    Worker function: Executes a 500-iteration Frank-Wolfe policy mixture sweep.
    """
    np.random.seed(run_id)
    
    # 1. Thread-safe Environment Setup
    desc = np.array(MAPS[name], dtype='c')
    desc[desc == b'W'] = b'H'
    desc[desc == b'G'] = b'F'
    
    env = ModifiedFrozenLake(
        n_action=4, desc=desc, step_penalization=0, max_reward=0, 
        min_reward=-1, never_done=True, cyclic_mode=True, 
        goal_attractor=0., slippery=1
    )

    dynamics, _ = get_dynamics_and_rewards(env)
    n_states, SA = dynamics.shape
    n_actions = SA // n_states
    
    # Initialize pi_mix (C_0 in the paper) as uniform
    pi_mix = np.ones((n_states, n_actions)) / n_actions
    
    total_matrix_iterations = 0
    total_reward_iterations = 0
    empirical_entropies = []
    cumulative_matrix_iters = [] 
    
    # 2. Main Optimization Loop
    for iteration in range(50):
        # A. Oracle: Get state distribution of current mixture policy
        s_dist = compute_policy_induced_distribution(dynamics, pi_mix, steps=100)
        
        sa_dist = np.tile(s_dist, (n_actions, 1)).T
        sa_dist *= pi_mix
        sa_dist /= (sa_dist.sum() + 1e-12)
        
        # B. Define Reward: gradient of entropy is -log(d)
        rt = -np.log(sa_dist + 1e-12)
        rt -= rt.mean() # Centering for numerical stability
        
        # C. APPROXPLAN: Get optimal deterministic policy for rt
        pi_star, iters = exact_hard_average_reward_vi(dynamics.T, rt, max_iter=20)
        
        total_matrix_iterations += iters
        total_reward_iterations += 1
        
        # D. Mixture Update (Frank-Wolfe step)
        pi_mix = (1.0 - eta_lr) * pi_mix + eta_lr * pi_star

        # Evaluation
        if iteration % 10 == 0:
            sa_s, _, _ = get_empirical_entropy(env, pi_mix, n_episodes=10, max_steps=1000, state_action=state_action)
            empirical_entropies.append(round(sa_s, 4))
            cumulative_matrix_iters.append(total_matrix_iterations)
        
    # plot_dist and save the fig:
    plot_dist(desc, pi_mix, s_dist, filename=f'./results/{name}/hazan_policy_distribution_{run_id}.png', normalize=True, figsize=(8, 4))
    return {
        'run_id': run_id,
        'algorithm': 'Hazan MaxEnt',
        'eta': eta_lr,
        'matrix_iterations': total_matrix_iterations,
        'reward_iterations': total_reward_iterations,
        'empirical_entropies': empirical_entropies,
        'cumulative_matrix_iters': cumulative_matrix_iters,
        'final_entropy': np.mean(empirical_entropies[-10:]) if empirical_entropies else 0.0
    }

if __name__ == '__main__':
    # === Sweep Configuration ===
    output_dir = './results/hyperparameter_sweeps'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f'{map_name}_hazan_variance_results.csv')
    
    # Define grid: Sweep over step sizes (eta) and pick the best one to write
    etas = np.logspace(-3, -1, 5) 
    num_runs = 5
    
    print(f"Starting Hazan MaxEnt sweep over {len(etas)} step sizes ({num_runs} seeds each)...")
    
    all_results = Parallel(n_jobs=10)(
        delayed(evaluate_hazan_hyperparameters)(r, e) for e in etas for r in range(num_runs)
    )
    
    # Filter for the best eta based on average final entropy across seeds
    best_eta = max(etas, key=lambda e: np.mean([res['final_entropy'] for res in all_results if res['eta'] == e]))
    best_runs = [res for res in all_results if res['eta'] == best_eta]
    
    for res in best_runs:
        del res['final_entropy'] # Clean up before writing
    
    # === Save to CSV ===
    print(f"Execution complete. Best eta: {best_eta:.2e}. Writing data to {csv_filename}...")
    headers = ['run_id', 'algorithm', 'eta', 'matrix_iterations', 'reward_iterations', 'empirical_entropies', 'cumulative_matrix_iters']
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(best_runs)
        
    print("Data successfully saved.")