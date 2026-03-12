# """
# Parallel model-based entropy-maximization sweep.
# Distributes evaluations across 8 CPU cores and logs to a CSV.
# Sweeps over discount factors and alphas, saving only the best alpha per discount.
# """

# import os
# import csv
# import numpy as np
# from joblib import Parallel, delayed
# from tqdm import tqdm

# # User-specific imports
# from frozen_lake_env import ModifiedFrozenLake, MAPS
# from utils import get_mdp_transition_matrix, get_dynamics_and_rewards, compute_policy_induced_distribution
# from uv_agent_utils import get_empirical_entropy

# map_name = 'windycliff'

# def exact_softqlearning(dynamics, rewards, beta, gamma, prior_policy, max_iter=100, q0=None):
#     """Solves for the optimal policy using exact soft Q-learning with a strict, uniform number of Bellman updates."""
#     nS, nA = rewards.shape
    
#     Q = q0.copy() if q0 is not None else np.random.uniform(size=(nS, nA))
    
#     # Strictly run for max_iter without breaking early for tolerance
#     for _ in range(max_iter):
#         Qmax = np.max(Q, axis=1, keepdims=True)
#         Q_shifted = Q - Qmax  
        
#         exp_terms = np.sum(prior_policy * np.exp(beta * Q_shifted), axis=1)
#         exp_terms = np.maximum(exp_terms, 1e-12) 
        
#         V = (1 / beta) * np.log(exp_terms)
#         V += Qmax.flatten()  
        
#         expected_future_V = (dynamics @ V).reshape(nS, nA)
#         Q = rewards + gamma * expected_future_V
            
#     Qmax_final = np.max(Q, axis=1, keepdims=True)
#     optimal_policy = prior_policy * np.exp(beta * (Q - Qmax_final))
#     optimal_policy /= optimal_policy.sum(axis=1, keepdims=True) 
    
#     return optimal_policy, Q, max_iter

# def evaluate_hyperparameters(gamma, alpha, name=map_name, state_action=True):
#     """
#     Worker function: Executes a 20-iteration reward sweep for a specific (gamma, alpha) pair.
#     Instantiates its own environment to guarantee thread-safety.
#     """
#     # 1. Thread-safe Environment Setup
#     desc = np.array(MAPS[name], dtype='c')
#     desc[desc == b'W'] = b'H'
#     desc[desc == b'G'] = b'F'
    
#     env = ModifiedFrozenLake(
#         n_action=4, desc=desc, step_penalization=0, max_reward=0, 
#         min_reward=-1, never_done=True, cyclic_mode=True, 
#         goal_attractor=0., slippery=1
#     )

#     dynamics, _ = get_dynamics_and_rewards(env)
#     n_states, SA = dynamics.shape
#     n_actions = SA // n_states
    
#     prior_policy = np.ones((n_states, n_actions)) / n_actions
#     running_rewards = np.zeros((n_states, n_actions))
#     Q = None
    
#     total_matrix_iterations = 0
#     total_reward_iterations = 0
#     empirical_entropies = []
#     cumulative_matrix_iters = [] 
    
#     beta = 1.0  # Fixed beta
    
#     # 2. Main Optimization Loop
#     for iteration in range(500):
#         policy, Q, iters = exact_softqlearning(
#             dynamics.T, running_rewards, beta, gamma=gamma, 
#             prior_policy=prior_policy, q0=Q,
#             max_iter=20
#         )
        
#         total_matrix_iterations += iters
#         total_reward_iterations += 1
#         cumulative_matrix_iters.append(total_matrix_iterations)
        
#         s_dist = compute_policy_induced_distribution(dynamics, policy, steps=100)
        
#         # Reward Update
#         sa_dist = np.tile(s_dist, (n_actions, 1)).T
#         sa_dist *= policy
#         sa_dist /= (sa_dist.sum() + 1e-12)
        
#         raw_new_rewards = -np.log(sa_dist + 1e-8)  
#         raw_new_rewards -= raw_new_rewards.mean()
        
#         if iteration == 0:
#             running_rewards = raw_new_rewards.copy()
#         else:
#             running_rewards = (alpha * raw_new_rewards) + ((1 - alpha) * running_rewards)

#         if iteration % 10 == 0:
#             # Track Empirical Entropy
#             sa_s, _, _ = get_empirical_entropy(env, policy, n_episodes=10, max_steps=1000, state_action=state_action)
#             empirical_entropies.append(round(sa_s, 4))
        
#     return {
#         'discount': gamma,
#         'alpha': alpha,
#         'matrix_iterations': total_matrix_iterations,
#         'reward_iterations': total_reward_iterations,
#         'empirical_entropies': empirical_entropies,
#         'cumulative_matrix_iters': cumulative_matrix_iters,
#         'final_entropy': np.mean(empirical_entropies[-10:]) if empirical_entropies else 0.0
#     }

# if __name__ == '__main__':
#     # === Sweep Configuration ===
#     output_dir = './results/hyperparameter_sweeps'
#     os.makedirs(output_dir, exist_ok=True)
#     csv_filename = os.path.join(output_dir, f'{map_name}_best_alpha_sweep_results.csv')
    
#     # Define the search grid
#     gammas = [0.8, 0.9, 0.95, 0.99] #np.linspace(0.7, 1.0, 11)   
#     alphas = np.logspace(-4, -2, 6)  
    
#     # Generate all pairs
#     hyperparameter_pairs = [(g, a) for g in gammas for a in alphas]
#     print(f"Starting parallel sweep over {len(hyperparameter_pairs)} combinations using 8 cores...")
    
#     # === Execute Parallel Sweep ===
#     results = Parallel(n_jobs=18)(
#         delayed(evaluate_hyperparameters)(g, a) for g, a in tqdm(hyperparameter_pairs)
#     )
    
#     # === Filter for the Best Alpha per Gamma ===
#     best_results = []
#     for g in gammas:
#         # Group all runs that share the current discount factor
#         runs_for_gamma = [r for r in results if r['discount'] == g]
        
#         if runs_for_gamma:
#             # Find the run with the highest final empirical entropy
#             best_run = max(runs_for_gamma, key=lambda x: x['final_entropy'])
            
#             # Remove the temporary sort key before writing to CSV
#             del best_run['final_entropy']
#             best_results.append(best_run)
    
#     # === Save to CSV ===
#     print(f"Sweep complete. Filtered down to the best {len(best_results)} runs.")
#     print(f"Writing data to {csv_filename}...")
#     headers = ['discount', 'alpha', 'matrix_iterations', 'reward_iterations', 'empirical_entropies', 'cumulative_matrix_iters']
    
#     with open(csv_filename, mode='w', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=headers)
#         writer.writeheader()
#         writer.writerows(best_results)
        
#     print("Data successfully saved.")
"""
Parallel model-based entropy-maximization sweep.
Distributes evaluations across CPU cores and logs to a CSV.
Runs multiple seeded initializations for a fixed alpha to allow plotting variance.
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

map_name = '11x11snake'
map_name_to_maxiters = {
    
    '11x11snake': 50,
    'windycliff': 20
}


def exact_softqlearning(dynamics, rewards, beta, gamma, prior_policy, max_iter=100, q0=None):
    """Solves for the optimal policy using exact soft Q-learning with a strict, uniform number of Bellman updates."""
    nS, nA = rewards.shape
    
    Q = q0.copy() if q0 is not None else np.random.uniform(size=(nS, nA))
    
    # Strictly run for max_iter without breaking early for tolerance
    for _ in range(max_iter):
        Qmax = np.max(Q, axis=1, keepdims=True)
        Q_shifted = Q - Qmax  
        
        exp_terms = np.sum(prior_policy * np.exp(beta * Q_shifted), axis=1)
        exp_terms = np.maximum(exp_terms, 1e-12) 
        
        V = (1 / beta) * np.log(exp_terms)
        V += Qmax.flatten()  
        
        expected_future_V = (dynamics @ V).reshape(nS, nA)
        Q = rewards + gamma * expected_future_V
            
    Qmax_final = np.max(Q, axis=1, keepdims=True)
    optimal_policy = prior_policy * np.exp(beta * (Q - Qmax_final))
    optimal_policy /= optimal_policy.sum(axis=1, keepdims=True) 
    
    return optimal_policy, Q, max_iter

def evaluate_hyperparameters(run_id, gamma, alpha, name=map_name, state_action=False):
    """
    Worker function: Executes a 500-iteration reward sweep.
    Uses run_id as a random seed to ensure different Q-table initializations.
    """
    # 0. Set seed for run variance
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
    
    prior_policy = np.ones((n_states, n_actions)) / n_actions
    running_rewards = np.zeros((n_states, n_actions))
    Q = None
    
    total_matrix_iterations = 0
    total_reward_iterations = 0
    empirical_entropies = []
    cumulative_matrix_iters = [] 
    
    beta = 10.0  # Fixed beta
    
    # 2. Main Optimization Loop
    for iteration in range(5000):
        policy, Q, iters = exact_softqlearning(
            dynamics.T, running_rewards, beta, gamma=gamma, 
            prior_policy=prior_policy, q0=Q,
            max_iter=50
        )
        
        total_matrix_iterations += iters
        total_reward_iterations += 1
        
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

        # Evaluation
        if iteration % 10 == 0:
            sa_s, _, _ = get_empirical_entropy(env, policy, n_episodes=10, max_steps=1000, state_action=state_action)
            
            # FIX: Append to both lists at the same time so they are equal length!
            empirical_entropies.append(round(sa_s, 4))
            cumulative_matrix_iters.append(total_matrix_iterations)
        
    return {
        'run_id': run_id,
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
    csv_filename = os.path.join(output_dir, f'{map_name}_discount_variance_results.csv')
    
    # Define the execution grid
    gammas = [0.8, 0.9, 0.95, 0.99]
    fixed_alpha = 1e-3
    num_runs = 5
    
    # Generate all run combinations (5 runs per gamma)
    tasks = [(r, g, fixed_alpha) for r in range(num_runs) for g in gammas]
    print(f"Starting parallel execution of {len(tasks)} runs ({num_runs} seeds per gamma) using up to 18 cores...")
    
    # === Execute Parallel Runs ===
    results = Parallel(n_jobs=18)(
        delayed(evaluate_hyperparameters)(r, g, a) for r, g, a in tqdm(tasks)
    )
    
    # === Save to CSV ===
    print(f"Execution complete. Writing data to {csv_filename}...")
    headers = ['run_id', 'discount', 'alpha', 'matrix_iterations', 'reward_iterations', 'empirical_entropies', 'cumulative_matrix_iters']
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)  # Write all results, no filtering
        
    print("Data successfully saved.")