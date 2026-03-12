import itertools
import gymnasium
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

def chi(u, n_states, n_actions, prior_policy=None):
    if prior_policy is None:
        prior_policy = np.ones((n_states, n_actions)) / n_actions
    u = u.reshape(n_states, n_actions)
    return (prior_policy * u).sum(axis=1)


def get_dynamics_and_rewards(env):

    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j * prb)

    dynamics = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    colsums = dynamics.sum(axis=0)
    assert (colsums.round(12) == 1.).all(), f"{colsums.min()}, {colsums.max()}"

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)),
                         shape=shape).sum(axis=0)

    return dynamics, rewards


def find_exploration_policy(dynamics, rewards, n_states, n_actions, beta=1, alpha=0.01, prior_policy=None, debug=False, max_it=20):

    rewards[:] = 0
    prior_policy = np.matrix(np.ones((n_states, n_actions))) / \
        n_actions if prior_policy is None else prior_policy
    if debug:
        entropy_list = []

    for i in range(1, 1 + max_it):
        u, v, optimal_policy, _, estimated_distribution, _ = solve_biased_unconstrained(
            beta, dynamics, rewards, prior_policy, bias_max_it=20)

        sa_dist = np.multiply(u, v.T)
        mask = sa_dist > 0
        r = rewards.copy()
        r[:] = 0.
        r[mask] = - np.log(sa_dist[mask].tolist()[0]) / beta
        r = r - r.max()
        rewards = (1 - alpha) * rewards + alpha * r

        if debug:
            x = sa_dist[sa_dist > 0]
            entropy = - np.multiply(x, np.log(x)).sum()
            entropy_list.append(entropy)

            # print(f"{i=}\t{alpha=:.3f}\t{entropy=: 10.4f}\t", end='')

    return optimal_policy


def solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000, tolerance=1e-8):
    # Do not scale the convergence tolerance by beta; keep an absolute tolerance
    # Scaling tolerance by beta makes convergence criteria laxer for large beta
    # and can hide numerical errors. Keep the tolerance as given.

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    # Shift rewards by their max before exponentiating to avoid overflow
    # or extreme underflow when beta is large. We will correct the Perron
    # eigenvalue at the end to account for this shift.
    r = np.array(rewards).flatten()
    r_max = float(r.max()) if r.size > 0 else 0.0
    r_shift = r - r_max
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * r_shift))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()
    M_scale = 1.

    # left eigenvector
    u = np.matrix(np.ones((nSnA, 1)))
    u_scale = np.sum(u)

    # right eigenvector
    v = np.matrix(np.ones((nSnA, 1))) * nSnA ** 2

    lol = float('inf')
    hil = 0.

    for i in range(1, eig_max_it+1):

        uk = (Mt).dot(u)
        lu = np.sum(uk) / u_scale
        mask = np.logical_and(uk > 0., uk < np.inf)
        rescale = 1. / np.sqrt(uk[mask].max()*uk[mask].min())
        uk = uk / lu * rescale
        u_scale *= rescale

        v_scale = np.sum(v)
        vk = M.dot(v)
        lv = np.sum(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))
                       ).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))
                       ).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100 == 0:
            rescale = 1 / np.sqrt(lu)
            Mt = Mt * rescale
            M_scale *= rescale

        if u_err <= tolerance and v_err <= tolerance:
            # if u_err <= tolerance:
            l = lu / M_scale
            # print(f"{i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {M_scale=:.4e}, {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu / M_scale
        # : {i: 8d}, {u.min() = : .4e}, {u.max() = : .4e}. {M_scale = : .4e}, {lu = : .4e}, {l = : .4e}, {u_err = : .4e}, {v_err = : .4e}")
        print(f"Did not converge")

    # Adjust eigenvalue to undo the reward shift: since we used
    # T = exp(beta*(r - r_max)) = exp(beta*r) / exp(beta*r_max), the
    # computed eigenvalue is scaled by 1/exp(beta*r_max). Multiply back.
    l = (lu / M_scale) * np.exp(beta * r_max)

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(
        u, v.T).reshape((nS, nA)))#.sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def solve_unconstrained_v1(beta, dynamics, rewards, prior_policy, eig_max_it=10000, tolerance=1e-8):

    scale = 1 / np.exp(beta * rewards.min())

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()

    # left eigenvector
    u = np.matrix(np.ones((nSnA, 1))) * scale
    u_scale = np.linalg.norm(u)

    # right eigenvector
    v = np.matrix(np.ones((nSnA, 1))) * scale
    v_scale = np.linalg.norm(v)

    lol = float('inf')
    hil = 0.

    metrics_list = []

    for i in range(1, eig_max_it+1):

        uk = (Mt).dot(u)
        lu = np.linalg.norm(uk) / u_scale
        uk = uk / lu

        vk = M.dot(v)
        lv = np.linalg.norm(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))
                       ).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))
                       ).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100_000 == 0:
            metrics_list.append(dict(
                lu=lu,
                lv=lv,
                u_err=u_err,
                v_err=v_err,
            ))

        if u_err <= tolerance and v_err <= tolerance:
            l = lu
            # print(f"{i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu
        # : {i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
        print(f"Did not converge")

    l = lu

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(
        u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def solve_biased_unconstrained(beta, prior_dynamics, rewards, prior_policy=None, target_dynamics=None, eig_max_it=10000, alpha=0.9999, bias_max_it=200, ground_truth_policy=None, tolerance=1e-6):

    nS, nSnA = prior_dynamics.shape
    nA = nSnA // nS

    if prior_policy is None:
        prior_policy = np.matrix(np.ones((nS, nA))) / nA

    if target_dynamics is None:
        target_dynamics = prior_dynamics

    ### initialization ###
    td_bias = prior_dynamics.copy()
    td_bias.data[:] = 1.
    rw_bias = np.zeros_like(rewards)
    biased_dynamics = prior_dynamics.copy()
    biased_rewards = rewards

    error_policy_list = []
    error_dynamics_list = []
    policy_list = []
    for i in range(1, bias_max_it+1):

        l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solve_unconstrained(
            beta, biased_dynamics, biased_rewards, prior_policy, eig_max_it=eig_max_it)
        policy_list.append(optimal_policy)
        if ground_truth_policy is not None:
            error_policy = compute_max_kl_divergence(
                optimal_policy, ground_truth_policy, axis=1)
            error_policy_list.append(error_policy)

        x = target_dynamics.tocoo()
        optimal = np.array(optimal_dynamics[x.row, x.col]).flatten()
        mask = optimal > 0.
        # x.data[mask] = np.log(x.data[mask] / optimal[mask]) * x.data[mask]
        # x.data[~mask] = 0
        x.data[mask] = np.log(optimal[mask] / x.data[mask]) * optimal[mask]
        x.data[~mask] = 0

        kl_err = np.abs(x.sum(axis=0)).max()
        error_dynamics_list.append(kl_err)
        if kl_err < tolerance:
            print(f'Solved in {i} iterations')
            break

        ratio = prior_dynamics.tocoo()
        mask = ratio.data > 0
        ratio.data[mask] = np.array(target_dynamics[ratio.row, ratio.col]).flatten()[
            mask] / ratio.data[mask]
        ratio.data[~mask] = 0.

        chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
        chi_inv = np.array(chi).flatten()
        mask = chi_inv > 0
        chi_inv[mask] = 1 / chi_inv[mask]
        chi_inv[chi_inv == np.inf] = 0.
        chi_inv = np.matrix(chi_inv).T

        next_td_bias = ratio.multiply(chi_inv)
        scale = prior_dynamics.multiply(next_td_bias).sum(axis=0)
        scale_inv = np.array(scale).flatten()
        mask = scale_inv > 0
        scale_inv[mask] = 1 / scale_inv[mask]
        scale_inv = np.matrix(scale_inv)

        next_td_bias = next_td_bias.multiply(scale_inv).tocsr()
        td_bias = td_bias + alpha * (next_td_bias - td_bias)

        biased_dynamics = prior_dynamics.multiply(td_bias)

        elem = target_dynamics.tocoo()
        biased = np.array(biased_dynamics[elem.row, elem.col]).flatten()
        biased_inv = 1 / biased
        biased_inv[biased_inv == np.inf] = 1.
        mask = (biased > 0) & (elem.data > 0)
        elem.data[mask] = np.log(
            elem.data[mask] * biased_inv[mask]) * elem.data[mask]
        elem.data[~mask] = 0.
        rw_bias = elem.sum(axis=0) / beta

        biased_rewards = rewards + rw_bias
        reward_offset = - biased_rewards.max()
        biased_rewards += reward_offset

    if i == bias_max_it:
        print(f'Did not finish after {i} iterations')

    info = dict(
        error_dynamics_list=error_dynamics_list,
        error_policy_list=error_policy_list,
        policy_list=policy_list,
        iterations_completed=i,
    )
    return u, v, optimal_policy, optimal_dynamics, estimated_distribution, info


def compute_max_kl_divergence(dist_a, dist_b, axis=0):
    numer = csr_matrix(dist_a)
    denom = coo_matrix(dist_b)
    kldiv = denom.copy()
    numer = np.array(numer[denom.row, denom.col]).flatten()
    kldiv.data = np.log(numer / denom.data) * numer
    kldiv = kldiv.sum(axis=axis)

    return kldiv.max()


def compute_policy_induced_distribution(dynamics, policy, steps, isd=None):
    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    mdp_generator = get_mdp_transition_matrix(dynamics, policy)

    if isd is not None:
        x = np.multiply(np.matrix(isd).T, policy).flatten().T
    else:
        x = np.matrix(np.ones((nS * nA, 1))) / nS / nA

    x_previous = None
    for _ in range(steps):
        x = mdp_generator.dot(x)
        # check if x has converged to a stationary distribution (optional)
        if x_previous is not None and np.linalg.norm(x - x_previous) < 1e-6:
            break
        x_previous = x

    return np.array(x).reshape((nS, nA)).sum(axis=1)


def get_mdp_transition_matrix(transition_dynamics, policy):

    nS, nSnA = transition_dynamics.shape
    nA = nSnA // nS

    td_coo = transition_dynamics.tocoo()

    rows = (td_coo.row.reshape((-1, 1)) * nA +
            np.array(list(range(nA)))).flatten()
    cols = np.broadcast_to(td_coo.col.reshape((-1, 1)),
                           (len(td_coo.col), nA)).flatten()
    data = np.broadcast_to(td_coo.data, (nA, len(td_coo.data))).T.flatten()

    mdp_transition_matrix = csr_matrix((data, (rows, cols)), shape=(
        nSnA, nSnA)).multiply(policy.reshape((-1, 1)))

    return mdp_transition_matrix


def largest_eigs_dense(A, n_eigs=1):

    if 'toarray' in dir(A):
        # need to be a dense matrix
        A = A.toarray()

    eigvals, eigvecs = np.linalg.eig(A)
    try:
        eigvals, eigvecs = process_complex_eigs(eigvals, eigvecs)
    except ValueError:
        raise

    return eigvals[:n_eigs], eigvecs[:, :n_eigs]


def printf(label, i, out_of=0, add_1=True):
    if add_1:
        index = i + 1
    else:
        index = i
    if out_of != 0:
        print(f'\r{label}: {index}/{out_of}', flush=True, end='')
    else:
        print(f'\r{label}: {index}', flush=True, end='')
    if out_of != 0:
        if i == out_of-1:
            print('\nFinished.')
    return


def training_episode(env, training_policy):
    sarsa_experience = []

    state = env.reset()
    action = np.random.choice(env.nA, p=training_policy[state])
    done = False
    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = np.random.choice(env.nA, p=training_policy[next_state])
        sarsa_experience.append(
            ((state, action, reward, next_state, next_action), done))
        state, action = next_state, next_action

    return sarsa_experience


def gather_experience(env, training_policy, batch_size, n_jobs=1):
    if n_jobs > 1:
        split_experience = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(training_episode)(env, training_policy) for _ in range(batch_size))
    elif n_jobs == 1:
        split_experience = [training_episode(
            env, training_policy) for _ in range(batch_size)]

    return list(itertools.chain.from_iterable(split_experience))

# From old gym code for DiscreteEnv:
def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()

class DiscreteEnv(gymnasium.Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, False, {"prob": p})
