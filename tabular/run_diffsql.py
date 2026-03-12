# """
# Parallel model-based entropy-maximization sweep for Differential Soft Q-Learning.
# Distributes evaluations across CPU cores, sweeps over the reward update rate (alpha),
# and saves only the single best result to CSV.
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

# def exact_differential_softqlearning(dynamics, rewards, beta, prior_policy, max_iter=1000, q0=None, rho0=0.0):
#     """
#     Solves for the optimal policy using average-reward (differential) soft Q-learning 
#     with a strict, uniform number of Bellman updates.
#     """
#     nS, nA = rewards.shape
#     rho = rho0  # Initialize average reward estimate
#     Q = q0.copy() if q0 is not None else np.random.uniform(size=(nS, nA))
    
#     # Strictly run for max_iter without breaking early
#     for _ in range(max_iter):
#         Qmax = np.max(Q, axis=1, keepdims=True)
#         Q_shifted = Q - Qmax  
        
#         exp_terms = np.sum(prior_policy * np.exp(beta * Q_shifted), axis=1)
#         exp_terms = np.maximum(exp_terms, 1e-12) 
        
#         V = (1 / beta) * np.log(exp_terms)
#         V += Qmax.flatten()  
        
#         expected_future_V = (dynamics @ V).reshape(nS, nA)
        
#         # Differential Update: Gamma is implicitly 1.0
#         Q = rewards - rho + expected_future_V
#         rho = Q[0,0]
#         # The Secret Sauce: Center Q-values to model the average reward (rho)
#         # This prevents the Q-values from drifting to infinity
#         # Q -= np.mean(Q)
            
#     Qmax_final = np.max(Q, axis=1, keepdims=True)
#     optimal_policy = prior_policy * np.exp(beta * (Q - Qmax_final))
#     optimal_policy /= optimal_policy.sum(axis=1, keepdims=True) 
    
#     return optimal_policy, Q, rho, max_iter

# def evaluate_differential_hyperparameters(alpha, name=map_name, state_action=True):
#     """
#     Worker function: Executes a 20-iteration reward sweep for a specific alpha.
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
    
#     beta = 10.0  # Fixed beta
#     rho = 0
#     # 2. Main Optimization Loop
#     for iteration in range(500):
#         policy, Q, rho, iters = exact_differential_softqlearning(
#             dynamics.T, running_rewards, beta, 
#             prior_policy=prior_policy, q0=Q, rho0=rho,
#             max_iter=100
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

#         # Track Empirical Entropy
#         sa_s, _, _ = get_empirical_entropy(env, policy, n_episodes=10, max_steps=500, state_action=state_action)
#         empirical_entropies.append(round(sa_s, 4))
        
#     return {
#         'algorithm': 'Differential Soft-Q',
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
#     csv_filename = os.path.join(output_dir, f'{map_name}_differential_best_alpha_sweep_results.csv')
    
#     # Define the search grid
#     alphas = np.logspace(-4, -2, 3)  # Sweeping 15 alpha values
    
#     print(f"Starting parallel Differential Soft-Q sweep over {len(alphas)} alpha combinations using 8 cores...")
    
#     # === Execute Parallel Sweep ===
#     results = Parallel(n_jobs=12)(
#         delayed(evaluate_differential_hyperparameters)(a) for a in tqdm(alphas)
#     )
    
#     # === Filter for the Best Alpha ===
#     # Find the run with the highest final empirical entropy
#     best_run = max(results, key=lambda x: x['final_entropy'])
    
#     # Remove the temporary sort key before writing to CSV
#     del best_run['final_entropy']
    
#     # === Save to CSV ===
#     print(f"\nSweep complete. Best alpha found: {best_run['alpha']:.2e}")
#     print(f"Writing data to {csv_filename}...")
    
#     headers = ['algorithm', 'alpha', 'matrix_iterations', 'reward_iterations', 'empirical_entropies', 'cumulative_matrix_iters']
    
#     with open(csv_filename, mode='w', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=headers)
#         writer.writeheader()
#         writer.writerow(best_run) # Writing just the single best run
        
#     print("Data successfully saved.")

"""
Parallel model-based entropy-maximization sweep for Differential Soft Q-Learning.
Distributes evaluations across CPU cores.
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

def exact_differential_softqlearning(dynamics, rewards, beta, prior_policy, max_iter=1000, q0=None, rho0=0.0):
    """
    Solves for the optimal policy using average-reward (differential) soft Q-learning 
    with a strict, uniform number of Bellman updates.
    """
    nS, nA = rewards.shape
    rho = rho0  # Initialize average reward estimate
    Q = q0.copy() if q0 is not None else np.random.uniform(size=(nS, nA))
    
    # Strictly run for max_iter without breaking early
    for _ in range(max_iter):
        Qmax = np.max(Q, axis=1, keepdims=True)
        Q_shifted = Q - Qmax  
        
        exp_terms = np.sum(prior_policy * np.exp(beta * Q_shifted), axis=1)
        exp_terms = np.maximum(exp_terms, 1e-12) 
        
        V = (1 / beta) * np.log(exp_terms)
        V += Qmax.flatten()  
        
        expected_future_V = (dynamics @ V).reshape(nS, nA)
        
        # Differential Update: Gamma is implicitly 1.0
        Q = rewards - rho + expected_future_V
        rho = Q[0,0]
            
    Qmax_final = np.max(Q, axis=1, keepdims=True)
    optimal_policy = prior_policy * np.exp(beta * (Q - Qmax_final))
    optimal_policy /= optimal_policy.sum(axis=1, keepdims=True) 
    
    return optimal_policy, Q, rho, max_iter

def evaluate_differential_hyperparameters(run_id, alpha, name=map_name, state_action=False):
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
    
    beta = 10.0  # Reset to 1.0 for fair comparison with EVE
    rho = 0
    
    # 2. Main Optimization Loop
    for iteration in range(5000):
        policy, Q, rho, iters = exact_differential_softqlearning(
            dynamics.T, running_rewards, beta, 
            prior_policy=prior_policy, q0=Q, rho0=rho,
            max_iter=50 # Matched the max_iter to the discounted script
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

        # Evaluation (Every 10 steps to sync lists)
        if iteration % 50 == 0:
            sa_s, _, _ = get_empirical_entropy(env, policy, n_episodes=10, max_steps=1000, state_action=state_action)
            
            empirical_entropies.append(round(sa_s, 4))
            cumulative_matrix_iters.append(total_matrix_iterations)
        
    return {
        'run_id': run_id,
        'algorithm': 'Differential Soft-Q',
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
    csv_filename = os.path.join(output_dir, f'{map_name}_differential_variance_results.csv')
    
    # Define the execution grid
    fixed_alpha = 1e-3
    num_runs = 5
    
    tasks = [r for r in range(num_runs)]
    print(f"Starting parallel execution of Differential Soft-Q for {num_runs} seeds using 5 cores...")
    
    # === Execute Parallel Runs ===
    results = Parallel(n_jobs=5)(
        delayed(evaluate_differential_hyperparameters)(r, fixed_alpha) for r in tqdm(tasks)
    )
    
    # === Save to CSV ===
    print(f"Execution complete. Writing data to {csv_filename}...")
    headers = ['run_id', 'algorithm', 'alpha', 'matrix_iterations', 'reward_iterations', 'empirical_entropies', 'cumulative_matrix_iters']
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)  # Write all results, no filtering
        
    print("Data successfully saved.")