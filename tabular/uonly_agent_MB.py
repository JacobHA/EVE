"""
Model-based entropy-maximization algorithm in a modified FrozenLake environment.
Iteratively updates rewards to encourage uniform visitation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_env import ModifiedFrozenLake, MAPS
from utils import get_mdp_transition_matrix
from uv_agent_utils import get_empirical_entropy, log_metrics, visualize_reward
from utils import get_dynamics_and_rewards, solve_unconstrained
from visualization import plot_dist
import seaborn as sns

def analyze_convergence_rate(P):
    """
    Analyzes the convergence rate for a PRIMITIVE, non-negative matrix P.
    Returns both the Birkhoff bound (based on P^m) and the Eigenvalue rate.
    """
    n = P.shape[0]
    
    # --- Method 1: Eigenvalue Gap (SLEM) ---
    # This is the most accurate measure for "local" asymptotic convergence speed
    # We find the magnitude of the second largest eigenvalue.
    vals = np.linalg.eigvals(P)
    # Sort by magnitude descending
    vals_sorted = np.sort(np.abs(vals))[::-1]
    
    lambda_1 = vals_sorted[0] # Should be approx 1.0 for stochastic P
    lambda_2 = vals_sorted[1]
    
    slem_rate = lambda_2 / lambda_1
    
    # --- Method 2: Birkhoff Contraction on P^m ---
    # We power P until it is strictly positive
    P_power = P.copy()
    m = 1
    max_pow = n * n  # Wielandt's bound (theoretical max steps needed)
    
    is_positive = False
    
    while m <= max_pow:
        if np.all(P_power > 1e-12): # Check for positivity (with tolerance)
            is_positive = True
            break
        P_power = P_power @ P
        m += 1
        
    if not is_positive:
        print("Warning: Matrix might not be primitive (still has zeros after n^2 steps).")
        return slem_rate, 1.0
    
    # Now compute Kappa on P^m
    # Calculate Cross-Ratio Phi
    phi = 0.0
    # Optimization: Only compute if matrix isn't too huge, else stick to eigenvalues
    if n < 1000: 
        for i in range(n):
            for j in range(n):
                if i == j: continue
                # We know P_power is positive, so no division by zero here
                ratios = P_power[i, :] / P_power[j, :]
                current_phi = np.max(ratios) / np.min(ratios)
                if current_phi > phi:
                    phi = current_phi
                    
        kappa_m = (np.sqrt(phi) - 1) / (np.sqrt(phi) + 1)
        
        # Effective per-step rate
        birkhoff_rate_per_step = kappa_m# ** (1.0 / m)
    else:
        birkhoff_rate_per_step = None

    print(f"Matrix became positive at power m={m}")
    return slem_rate, birkhoff_rate_per_step

# Example Usage:
# P = ... (your matrix)
# slem, birkhoff = analyze_convergence_rate(P)
# print(f"Asymptotic rate (Eigenvalue based): {slem:.4f}")
# print(f"Global contraction bound (Birkhoff): {birkhoff}")
sns.set_context("talk")
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
desc[desc == b'W'] = b'H'
desc[desc == b'G'] = b'F'

env = ModifiedFrozenLake(
    n_action=4,
    desc=desc,
    step_penalization=0,
    max_reward=0,
    min_reward=-1,
    never_done=True,
    cyclic_mode=True,
    goal_attractor=0.,
    slippery=0  # deterministic dynamics
)

# === Get dynamics and rewards ===
dynamics, _ = get_dynamics_and_rewards(env)
n_states, SA = dynamics.shape
n_actions = SA // n_states

# === Initialize policies and rewards ===
prior_policy = np.ones((n_states, n_actions)) / n_actions

# === Entropy tracking ===
n_visitable_states = np.sum(desc.flatten() == b'F')
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

update_prior_policy = True #ppi_iter % 2 == 0
PLOT_EVERY = 1
ppi_learning_rate = 0.0#50

errors=[]
# convert scipy.sparse._coo.coo_matrix to numpy array for processing
# p = P.toarray()
m=1
P = np.matrix(np.linalg.matrix_power(P.toarray(),m))
# === Main optimization loop ===
for beta in [1.0]*15:
    for ppi_iter in range(10):
        previous_u = u.copy()
        for ustep in range(16):
            if state_action: 
                power = beta/(1+beta)
                x_ = ((u.T @ P.A).T)**((1-beta)/beta) * u**(-1/beta)
                u = np.power(( (u.T @ P.A).T)**(1/beta) / (P.A @ x_), power)

            else:
                # a state-based update
                x_ = ((w.T @ Ps).T)**((1-beta)/beta) * w**(-1/beta)
                w=np.power(( (w.T @ Ps).T)**(1/beta) / (Ps @ x_), beta/(1+beta))
                # w /= sum(w)
                w_tiled = np.tile(w, (1, n_actions))  # expand to state-action space
                # expand back to state-action space via w(s) = sum_a u(s,a) and u(s,a) = sum_s' P(s'|s,a) w(s')
                u = np.zeros((env.nS, env.nA))

                for s in range(env.nS):
                    for a in range(env.nA):
                        u[s,a] = np.sum(P.A[:, s*env.nA + a] * w_tiled.flatten()) / prior_policy[s,a] # sum over s' of P(s'|s,a) w(s')
                        # PPI works fine after dividing by the prior here, but haven't fully verified (requires action marginalization in tilted matrix)
                        # May hinge on right evec independence of actions

                u = u.flatten()
        du = np.log(u + 1e-12) - np.log(previous_u + 1e-12)  # log-space difference to avoid scale issues
        osc = u.flatten().dot(previous_u.flatten()) / (np.linalg.norm(u) * np.linalg.norm(previous_u))
        errors.append([np.max(du) - np.min(du), osc.item()])
        
        print(u.reshape(env.nS,env.nA)[mask].min(), u.reshape(env.nS,env.nA)[mask].max())
            
        # refill masked u values with average of unmasked to allow proper normalization
        optimal_policy = u.reshape(env.nS,env.nA) * prior_policy
        optimal_policy[mask, :] = 1 # unif
        optimal_policy /= np.sum(optimal_policy, axis=1, keepdims=True)


        empirical_entropy, emp_dist = log_metrics(
            mask, max_entropy, env, optimal_policy, state_action
        )

        empirical_entropies.append(empirical_entropy)
        
        if update_prior_policy:
            print("Updating prior policy.")
            prior_policy = optimal_policy.copy()*ppi_learning_rate + prior_policy * (1-ppi_learning_rate)  # mix with prior policy
            # add a small uniform component to ensure all actions have non-zero probability (for stability)
            # prior_policy = prior_policy * (1 - 1e-3) + (1e-3 / n_actions)
            P = get_mdp_transition_matrix(dynamics, prior_policy)
            Ps = P.toarray().reshape(env.nS, env.nA, env.nS, env.nA).sum(axis=1).sum(axis=2)  # shape (nS, nS)

            print(analyze_convergence_rate(P.toarray()))
            # print(u)

        # Save plots
        if ppi_iter % PLOT_EVERY == 0:
            visualize_reward(desc, optimal_policy, emp_dist, i=ppi_iter, folder=f'./results/{name}')
            plt.close()
            plt.plot(emp_dist.flatten(), label=f'Iter {ppi_iter}')
            plt.savefig(f'./results/{name}/reward_iteration_{ppi_iter}.png')

        # === Final entropy plot ===
        plt.figure(figsize=(10, 5))
        # plt.plot([s / max_entropy for s in entropies], label=r'Closed-Form Entropy ($-\sum uv \log uv$)', color='blue', marker='o')
        plt.plot([s / max_entropy for s in empirical_entropies], label='Empirical Entropy (From Rollouts)', color='orange', marker='x')
        plt.axhline(1.0, color='k', linestyle='--', label='Max Entropy')
        plt.axhline(empirical_uniform_entropy / max_entropy, color='r', linestyle='--', label='Uniform Policy Entropy')
        # plt.title('Exploration Policy Entropy During Reward Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('State-Space Entropy (Normalized)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2)
        plt.savefig(f'./results/{name}/entropy_comparison.png', bbox_inches='tight')
        plt.close()

        # == Error plots ==
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
        
        # First subplot: Max Log-Diff
        errors_array = np.array(errors)
        ax1.plot(errors_array[:, 0], 'o-', label='Max Log-Diff in u')
        ax1.set_yscale('log')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Max Log-Diff')
        # draw a sloped line showing a reference exponential decay for convergence:
        # tau = 0.99988**15
        tau = 0.99948**16
        value = (2*tau + tau**2 * (beta - 1))/(1+beta)
        every_N = 10
        reference_line = errors_array[0, 0] * (value ** np.arange(every_N*len(errors_array)))
        ax1.plot(np.arange(len(errors_array)), reference_line[::every_N], 'k--')
        ax1.set_title('Convergence of u')
        ax1.legend()

        
        plt.tight_layout()
        plt.savefig(f'./results/{name}/u_convergence.png')
        plt.close()