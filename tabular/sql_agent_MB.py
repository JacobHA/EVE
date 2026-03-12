"""
Model-based entropy-maximization algorithm in a modified FrozenLake environment.
Iteratively updates rewards to encourage uniform visitation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_env import ModifiedFrozenLake, MAPS
# from tabular.utils import compute_policy_induced_distribution
from utils import get_mdp_transition_matrix
from uv_agent_utils import get_empirical_entropy, log_metrics, visualize_reward
from utils import get_dynamics_and_rewards, solve_unconstrained, compute_policy_induced_distribution
from visualization import plot_dist
import seaborn as sns
import numpy as np

def exact_softqlearning(dynamics, rewards, beta, gamma, prior_policy, max_iter=1000, tol=1e-6, q0=None):
    """
    Solves for the optimal policy using exact soft Q-learning with vectorized Bellman updates.
    """
    nS, nA = rewards.shape
    
    # Initialize Q-values
    if q0 is not None:
        Q = q0.copy()
    else:
        Q = np.random.uniform(size=(nS, nA))
    
    for iters in range(max_iter):
        Q_old = Q.copy()
        
        # Compute the soft value function
        Qmax = np.max(Q, axis=1, keepdims=True)
        Q_shifted = Q - Qmax  
        
        # Calculate expected values 
        exp_terms = np.sum(prior_policy * np.exp(beta * Q_shifted), axis=1)
        exp_terms = np.maximum(exp_terms, 1e-12) 
        
        V = (1 / beta) * np.log(exp_terms)
        V += Qmax.flatten()  
        
        # --- THE SPEEDUP: Vectorized Bellman Update ---
        # dynamics @ V performs the sum(P * V) across all states and actions simultaneously.
        # The result is shape (nS * nA,), which we reshape back to the 2D grid.
        expected_future_V = (dynamics @ V).reshape(nS, nA)
        Q = rewards + gamma * expected_future_V
        
        # Check for convergence
        if np.max(np.abs(Q - Q_old)) < tol:
            break
            
    # Compute the optimal policy with the LogSumExp stability trick
    Qmax_final = np.max(Q, axis=1, keepdims=True)
    optimal_policy = prior_policy * np.exp(beta * (Q - Qmax_final))
    optimal_policy /= optimal_policy.sum(axis=1, keepdims=True)  # Normalize
    
    return optimal_policy, Q, iters

def entropy(dist):
    """Compute the entropy of a distribution."""
    dist = dist[dist > 0]  # filter out zero entries to avoid log(0)
    return -np.sum(dist * np.log(dist))

# === Configuration ===
name = 'windycliff'

state_action = True  # use state-action distribution (True) or just state (False)
# === Environment setup ===
if name not in MAPS:
    raise ValueError(f"Map '{name}' not found. Available maps: {list(MAPS.keys())}")

os.makedirs(f'./results/{name}', exist_ok=True)
desc = np.array(MAPS[name], dtype='c')
desc[desc == b'W'] = b'H'  # treat holes as free space
desc[desc == b'G'] = b'F'  # treat holes as free space
# desc[desc == b'H'] = b'F`'  # treat holes as free space

env = ModifiedFrozenLake(
    n_action=4,
    desc=desc,
    step_penalization=0,
    max_reward=0,
    min_reward=-1,
    never_done=True,
    cyclic_mode=True,
    goal_attractor=0.,
    slippery=1  # deterministic dynamics
)

# === Get dynamics and rewards ===
dynamics, _ = get_dynamics_and_rewards(env)
n_states, SA = dynamics.shape
n_actions = SA // n_states

# === Initialize policies and rewards ===
prior_policy = np.ones((n_states, n_actions)) / n_actions

# === Entropy tracking ===
n_visitable_states = np.sum(desc.flatten() == b'F') + 1 # for start state
max_entropy = np.log(n_visitable_states) + (np.log(n_actions) if state_action else 0)
entropies, empirical_entropies = [], []

empirical_uniform_entropy, state_wise_entropy, _ = get_empirical_entropy(env, prior_policy, state_action=state_action)
mask = desc.flatten() != b'F'
P = get_mdp_transition_matrix(dynamics, prior_policy)
# marginalize over actions:
Ps = P.toarray().reshape(env.nS, env.nA, env.nS, env.nA).sum(axis=1).sum(axis=2)  # shape (nS, nS)
u = np.ones((env.nS*env.nA,1))
w = np.ones((env.nS,1)) / env.nS
dummy_distribution = np.ones((env.nS,env.nA)) / env.nS / env.nA

update_prior_policy = False #ppi_iter % 2 == 0
PLOT_EVERY = 1
errors=[]

# === Main optimization loop ===
# for beta in [0.5, 1.0, 2.0, 4.0]:
# Note: np.zeros is cleaner than np.random.rand(...) * 0.0
running_rewards = np.zeros((n_states, n_actions)) 

# Mixing weight (alpha). 
# 0.5 means a 50/50 split. Lowering it (e.g., 0.1) makes it update slower/more stably.
alpha = 0.05
Q = None
total_matrix_iterations = 0
total_reward_iterations = 0
for iteration, beta in enumerate(np.linspace(1.01, 11, 15)):

    # 1. Train using the stable, accumulated running average
    policy, Q, iters = exact_softqlearning(
        dynamics.T,
        running_rewards, 
        beta,
        gamma=0.75,
        prior_policy=prior_policy,
        q0=Q
    )
    total_matrix_iterations += iters
    total_reward_iterations += 1
    env_dynamics, _ = get_dynamics_and_rewards(env)
    s_dist = compute_policy_induced_distribution(
        env_dynamics,
        policy,
        steps=100
    )
    
    # 2. Calculate the brand new raw rewards
    sa_dist = np.tile(s_dist, (n_actions, 1)).T
    sa_dist *= policy
    sa_dist /= sa_dist.sum()  
    
    raw_new_rewards = -np.log(sa_dist + 1e-8)  
    raw_new_rewards -= raw_new_rewards.mean()
    
    # 3. Mix the raw rewards into the running average (EMA update)
    # This ensures history from ALL previous iterations is preserved and smoothly decays.
    if iteration == 0:
        # On the first step, just adopt the new rewards entirely to avoid 
        # artificially pulling the average down to zero.
        running_rewards = raw_new_rewards.copy()
    else:
        running_rewards = (alpha * raw_new_rewards) + ((1 - alpha) * running_rewards)

    # rewards = -np.log(sa_dist + 1e-8)  # add small constant for numerical stability
    if iteration % PLOT_EVERY == 0:
        visualize_reward(
            desc,
            policy,
            sa_dist,
            i=iteration,
            folder=f'./results/{name}'
        )
        plt.close()
        plt.plot(sa_dist.flatten(), label=f'Iter {iteration}')
        plt.savefig(f'./results/{name}/reward_iteration_{iteration}.png')
        # plt.close()
    # plot the entropies:

    # === Final entropy plot ===
    plt.figure(figsize=(10, 5))
    sa_s, s_s, sa_counts = get_empirical_entropy(
        env,
        policy,
        n_episodes=10,
        max_steps=1000,
        state_action=state_action
    )
    print(
        f"Iteration {iteration}: Empirical Entropy: "
        f"{sa_s:.4f} / {max_entropy:.4f} = {sa_s / max_entropy:.4f}"
    )
    # plt.plot([s / max_entropy for s in entropies], label=r'Closed-Form Entropy ($-\sum uv \log uv$)', color='blue', marker='o')
    plt.plot(
        [s / max_entropy for s in empirical_entropies],
        label='Empirical Entropy (From Rollouts)',
        color='orange',
        marker='x'
    )
    plt.axhline(1.0, color='k', linestyle='--', label='Max Entropy')
    plt.axhline(
        empirical_uniform_entropy / max_entropy,
        color='r',
        linestyle='--',
        label='Uniform Policy Entropy'
    )
    # plt.title('Exploration Policy Entropy During Reward Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('State-Space Entropy (Normalized)')
    # plt.legend(loc=
    # put the legend outside the plot, above everything:
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=2
    )
    # plt.tight_layout()
    # add a horizontal line indicating the beta value when its changed:
    # if reward_iteration % MAX_REWARD_ITER == 0:
    #     plt.axvline(reward_iteration, color='gray', linestyle=':', label=f'Beta={beta}')
    plt.savefig(
        f'./results/{name}/entropy_comparison.png',
        bbox_inches='tight'
    )
    plt.close()

    # # == Error plots ==
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # # First subplot: Max Log-Diff
    # errors_array = np.array(errors)
    # ax1.plot(errors_array[:, 0], 'o-', label='Max Log-Diff in u')
    # ax1.set_yscale('log')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Max Log-Diff')
    # ax1.set_title('Convergence of u')
    # ax1.legend()
    
    # # Second subplot: Oscillation
    # ax2.plot(errors_array[:, 1], 's-', label='Oscillation (u · u_prev)', color='orange')
    # ax2.set_xlabel('Iteration')
    # ax2.set_ylabel('Oscillation')
    # ax2.set_title('u Oscillation')
    # ax2.legend()
    
    # plt.tight_layout()
    # plt.savefig(f'./results/{name}/u_convergence.png')
    # plt.close()