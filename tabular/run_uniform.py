import os
import csv
import numpy as np

# User-specific imports
from frozen_lake_env import ModifiedFrozenLake, MAPS
from utils import get_dynamics_and_rewards, get_mdp_transition_matrix
from uv_agent_utils import get_empirical_entropy

# Note: Changed to match the previous script so the plots align, 
# but feel free to switch back to '11x11dzigzag' if needed!
map_name = '11x11snake' 


beta = 1.0

# 1. Environment Setup
desc = np.array(MAPS[map_name], dtype='c')
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
state_action = False
s_entropy, _, _ = get_empirical_entropy(env, prior_policy, n_episodes=10, max_steps=500, state_action=state_action)
            

# get the number of 'F' states in the map:
f_states = np.sum(env.desc == b'F') + 2 # add 1 for the start state, one for absorbing states
max_entropy_sa = np.log(f_states * n_actions) # Max entropy for uniform distribution over visitable states and actions
max_entropy_s = np.log(f_states) # Max entropy for uniform distribution over visitable states only
max_entropy = max_entropy_sa if state_action else max_entropy_s
print("max entropy", max_entropy)
print(f"Final Empirical State-Entropy at Beta=1.0: {s_entropy:.4f}")