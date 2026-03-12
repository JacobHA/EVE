"""
Parallel model-based entropy-maximization sweep.
Distributes evaluations across 8 CPU cores and logs to a CSV.
"""

import os
import csv
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# User-specific imports
from frozen_lake_env import ModifiedFrozenLake, MAPS
from utils import get_mdp_transition_matrix, get_dynamics_and_rewards, compute_policy_induced_distribution
from uv_agent_utils import get_empirical_entropy

map_name = 'windycliff'
def exact_softqlearning(dynamics, rewards, beta, gamma, prior_policy, max_iter=1000, tol=1e-6, q0=None):
    """Solves for the optimal policy using exact soft Q-learning with vectorized Bellman updates."""
    nS, nA = rewards.shape
    
    Q = q0.copy() if q0 is not None else np.random.uniform(size=(nS, nA))
    iters = 0
    
    for iters in range(max_iter):
        Q_old = Q.copy()
        
        Qmax = np.max(Q, axis=1, keepdims=True)
        Q_shifted = Q - Qmax  
        
        exp_terms = np.sum(prior_policy * np.exp(beta * Q_shifted), axis=1)
        exp_terms = np.maximum(exp_terms, 1e-12) 
        
        V = (1 / beta) * np.log(exp_terms)
        V += Qmax.flatten()  
        
        expected_future_V = (dynamics @ V).reshape(nS, nA)
        Q = rewards + gamma * expected_future_V
        
        if np.max(np.abs(Q - Q_old)) < tol:
            break
            
    Qmax_final = np.max(Q, axis=1, keepdims=True)
    optimal_policy = prior_policy * np.exp(beta * (Q - Qmax_final))
    optimal_policy /= optimal_policy.sum(axis=1, keepdims=True) 
    
    return optimal_policy, Q, iters

def evaluate_hyperparameters(gamma, alpha, name=map_name, state_action=False):
    """
    Worker function: Executes a full beta-sweep for a specific (gamma, alpha) pair.
    Instantiates its own environment to guarantee thread-safety.
    """
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
    
    prior_policy = np.ones((n_states, n_actions)) / n_actions
    running_rewards = np.zeros((n_states, n_actions))
    Q = None
    
    total_matrix_iterations = 0
    total_reward_iterations = 0
    empirical_entropies = []
    cumulative_matrix_iters = [] # NEW: Track the ragged x-axis
    
    # 2. Main Optimization Loop
    for iteration, beta in enumerate([1.0]*20):#np.linspace(1.0, 1.0, 15)):
        policy, Q, iters = exact_softqlearning(
            dynamics.T, running_rewards, beta, gamma=gamma, 
            prior_policy=prior_policy, q0=Q
        )
        
        total_matrix_iterations += iters
        total_reward_iterations += 1
        cumulative_matrix_iters.append(total_matrix_iterations)
        
        s_dist = compute_policy_induced_distribution(dynamics, policy, steps=100)
        
        # Reward Update
        sa_dist = np.tile(s_dist, (n_actions, 1)).T
        sa_dist *= policy
        sa_dist /= (sa_dist.sum() + 1e-12)
        
        raw_new_rewards = -np.log(sa_dist + 1e-8)  
        raw_new_rewards -= raw_new_rewards.mean()
        
        if iteration == 0:
            running_rewards = raw_new_rewards.copy()
        else:
            running_rewards = (alpha * raw_new_rewards) + ((1 - alpha) * running_rewards)

        # Track Empirical Entropy
        sa_s, _, _ = get_empirical_entropy(env, policy, n_episodes=10, max_steps=500, state_action=state_action)
        empirical_entropies.append(round(sa_s, 4))
        
    return {
        'discount': gamma,
        'alpha': alpha,
        'matrix_iterations': total_matrix_iterations,
        'reward_iterations': total_reward_iterations,
        'empirical_entropies': empirical_entropies,
        'cumulative_matrix_iters': cumulative_matrix_iters
    }

if __name__ == '__main__':
    # === Sweep Configuration ===
    output_dir = './results/hyperparameter_sweeps'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f'{map_name}_sweep_results.csv')
    
    # Define the search grid
    gammas = np.linspace(0.7, 1.0, 11)   # 11 values: 0.5, 0.55, ... 1.0
    alphas = np.logspace(-4, -1, 10)  # 10 values: 0.01, 0.12, ... 1.0
    
    # Generate all pairs
    hyperparameter_pairs = [(g, a) for g in gammas for a in alphas]
    print(f"Starting parallel sweep over {len(hyperparameter_pairs)} combinations using 8 cores...")
    
    # === Execute Parallel Sweep ===
    results = Parallel(n_jobs=8)(
        delayed(evaluate_hyperparameters)(g, a) for g, a in tqdm(hyperparameter_pairs)
    )
    
    # === Save to CSV ===
    print(f"Sweep complete. Writing data to {csv_filename}...")
    headers = ['discount', 'alpha', 'matrix_iterations', 'reward_iterations', 'empirical_entropies', 'cumulative_matrix_iters']
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print("Data successfully saved.")