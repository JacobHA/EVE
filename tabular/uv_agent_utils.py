import matplotlib.pyplot as plt
import numpy as np
from visualization import plot_dist


def get_empirical_entropy(env, policy, n_episodes=10, max_steps=10000, state_action=True):
    # return 0,0,np.ones((env.nS, env.nA))  # placeholder for the function, to be implemented
    """ runs trajectories to calculate the empirical entropy of the policy in the environment """
    policy = policy.reshape(env.nS, env.nA)  # ensure policy is in the right shape
    policy = policy / np.sum(policy, axis=1, keepdims=True)  # normalize policy to ensure it sums to 1
    # Run the policy in the environment to get empirical visitations
    state_action_counts = np.ones((env.nS, env.nA))
    for episode in range(n_episodes):
        state, info = env.reset()
        returns = 0
        for _ in range(max_steps):

            action = np.random.choice(env.nA, p=policy[state].flatten())
            next_state, rwd, term, trunc, info = env.step(action)
            
            returns += rwd
            state_action_counts[state,action] += 1
            state = next_state
            if (term):
                state, _ = env.reset()
    state_action_counts += 1e-10  # add small value to avoid division by zero
    state_action_counts /= np.sum(state_action_counts)  # normalize visit counts
    if state_action:
        empirical_entropy = -np.sum(state_action_counts * np.log(state_action_counts))  # add small value to avoid log(0)
        state_wise_entropy = -np.log(state_action_counts)
    else:
        # first marginalize over actions
        state_distribution = state_action_counts.sum(axis=1)
        state_distribution /= np.sum(state_distribution)
        empirical_entropy = -np.sum(state_distribution * np.log(state_distribution ))
        state_wise_entropy = -np.log(state_distribution)
    return empirical_entropy, state_wise_entropy, state_action_counts

def visualize_reward(desc, policy, distribution, i=0, folder=''):
    # visualize the steady state distribution, u*v:
    try:
        steady_state_dist = distribution.sum(axis=1).flatten()
    except np.exceptions.AxisError:
        steady_state_dist = distribution.flatten()
    
    # viz_rwd /= np.max(np.abs(viz_rwd))  # normalize to [0, 1]
    # scale up the policy to more easily see the differences
    scaled_policy = policy #** 8.0  # scale the policy for better visualization
    scaled_policy /= np.sum(scaled_policy, axis=1, keepdims=True)  # normalize policy to ensure it sums to 1
    plot_dist(desc, scaled_policy, steady_state_dist, filename=f'{folder}/reward_distribution_{i}.png', normalize=True, figsize=(8, 4))

def log_metrics(mask, max_entropy, env, policy, state_action=True):
    """ helper function for logging during training """
    # print the entropy;
    # assumes the distribution is over SA
    # mask out the states that are not valid (e.g. walls)
    # distribution = distribution[~mask] if mask is not None else distribution  # apply mask to distribution
    # distribution /= np.sum(distribution)  # normalize distribution
    # entropy = -np.sum(  distribution * np.log(distribution ))
    # print(f"Step {iteration}: Entropy: {entropy:.4f} / {max_entropy:.4f} = {entropy / (max_entropy):.4f}")

    s, _, empirical_distribution = get_empirical_entropy(env, policy, state_action=state_action, max_steps=1000, n_episodes=100) # note, the output here could be used as new reward for comparison
    # mask the empirical distribution as well
    if not state_action:
            
        empirical_dist = empirical_distribution.sum(axis=1)
        if mask is not None:
            empirical_dist = empirical_dist[~mask]  # apply mask to empirical distribution
        empirical_dist /= np.sum(empirical_dist)  # normalize empirical distribution
        empirical_entropy = -np.sum(empirical_dist * np.log(empirical_dist))  # add small value to avoid log(0)
    else:
        # reshape the mask to cast along action axis:
        if mask is not None:
            mask = np.tile(mask[:, np.newaxis], (1, env.nA))
            empirical_dist = empirical_distribution[~mask]
        empirical_dist /= np.sum(empirical_dist)
        empirical_entropy = -np.sum(empirical_dist * np.log(empirical_dist))
    # Log results
    print(f"Empirical Entropy: {empirical_entropy:.4f} / {max_entropy:.4f} = {empirical_entropy / (max_entropy):.4f}")
    return empirical_entropy, empirical_distribution

def greedy_policy(pi):
    # make a greedy policy from the given policy
    greedy_pi = np.zeros_like(pi)
    greedy_pi[np.arange(pi.shape[0]), np.argmax(pi, axis=1)] = 1.0
    return greedy_pi