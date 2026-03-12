"""
Deep EVE  vs  Soft-Q  vs  Hazan MaxEnt
========================================
Environment : 11×11 PointMaze, + obstacle (arm=3).
              Absorbing states at (0,5),(10,5),(5,0),(5,10).

EVE (FA)    : beta=1, no PPI. One-hot features, 128-dim MLP.
              Frozen-target sweeps. 500 sweeps.

Soft-Q      : gamma=0.99, beta=1 (MaxEnt soft Bellman).
              Intrinsic reward r(s) = -log d(s), mixed smoothly each iter
              with reward_lr. MLP learns Q via soft Bellman regression
              (frozen targets, batched Adam).

Hazan       : Frank-Wolfe on policy mixture (Hazan et al.).
              Reward r(s,a) = -log d_sa(s,a) on joint state-action dist.
              Hard VI: Q(s,a) = r(s,a) + gamma*V(s'), V(s) = max_a Q(s,a).
              MLP learns Q for current reward (frozen-target batched sweeps).
              Policy update: pi_mix = (1-eta)*pi_mix + eta*pi*  (Frank-Wolfe).
"""

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  MLP + Adam  (supports both per-sample and batched training)
# ═══════════════════════════════════════════════════════════════════════════════
class MLP:
    def __init__(self, sizes, lr=3e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.layers = [[rng.standard_normal((fo, fi)) * 0.01, np.zeros(fo)]
                       for fi, fo in zip(sizes, sizes[1:])]
        self.lr = lr; self.b1 = 0.9; self.b2 = 0.999; self.ep = 1e-8; self.t = 0
        self.m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.layers]
        self.v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.layers]

    # ── batched forward: X (N,d) -> out (N,), caches activations ─────────────
    def _fwd(self, X):
        acts, zs = [X.T], []
        for i, (W, b) in enumerate(self.layers):
            z = W @ acts[-1] + b[:, None]; zs.append(z)
            acts.append(np.tanh(z) if i < len(self.layers)-1 else z)
        return acts, zs, acts[-1][0]   # acts[-1][0] shape (N,)

    # ── batched Adam step given output error (N,) ─────────────────────────────
    def _bwd(self, acts, zs, err):
        """err = 2*(out - target) / N  (already scaled)."""
        delta = err[None, :]
        self.t += 1
        for i in reversed(range(len(self.layers))):
            W, b = self.layers[i]
            dW = delta @ acts[i].T
            db = delta.sum(1)
            if i > 0:
                delta = (W.T @ delta) * (1 - np.tanh(zs[i-1])**2)
            for j, g in enumerate([dW, db]):
                self.m[i][j] = self.b1*self.m[i][j] + (1-self.b1)*g
                self.v[i][j] = self.b2*self.v[i][j] + (1-self.b2)*g**2
                mh = self.m[i][j] / (1 - self.b1**self.t)
                vh = self.v[i][j] / (1 - self.b2**self.t)
                self.layers[i][j] -= self.lr * mh / (np.sqrt(vh) + self.ep)

    def train_batch(self, X, targets, n_passes=1):
        """Full-batch MSE regression for n_passes steps. Returns final MSE."""
        loss = 0.
        for _ in range(n_passes):
            acts, zs, out = self._fwd(X)
            err = out - targets
            loss = float((err**2).mean())
            self._bwd(acts, zs, 2 * err / len(targets))
        return loss

    def predict_batch(self, X, w=None):
        h = X.T; wl = w if w is not None else self.layers
        for i, (W, b) in enumerate(wl):
            z = W @ h + b[:, None]; h = np.tanh(z) if i < len(wl)-1 else z
        return h[0]

    def snapshot(self): return [[W.copy(), b.copy()] for W, b in self.layers]

    def reset(self, seed=0):
        """Re-initialise weights and Adam state."""
        rng = np.random.default_rng(seed)
        self.layers = [[rng.standard_normal((fo, fi)) * 0.01, np.zeros(fo)]
                       for fi, fo in zip(self.sizes, self.sizes[1:])]
        self.t = 0
        self.m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.layers]
        self.v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.layers]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Environment
# ═══════════════════════════════════════════════════════════════════════════════
class PointMaze:
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]; N_ACTIONS = 4

    def __init__(self, size=11):
        self.size = size; cx = cy = size // 2; arm = 3
        self.obstacle_cells = set()
        for d in range(-arm, arm+1):
            self.obstacle_cells.add((cx+d, cy)); self.obstacle_cells.add((cx, cy+d))
        self.absorbing_cells = {(0,cy),(size-1,cy),(cx,0),(cx,size-1)}
        self.free_cells = [(r,c) for r in range(size) for c in range(size)
                           if (r,c) not in self.obstacle_cells]
        self.cell_to_idx = {c:i for i,c in enumerate(self.free_cells)}
        self.n_free = len(self.free_cells)
        self.start_cell = (0,0); self.s0 = self.cell_to_idx[self.start_cell]
        self.absorbing_idx = {self.cell_to_idx[c] for c in self.absorbing_cells}
        self._build(); self.state = self.s0

    def _is_free(self, r, c):
        return 0<=r<self.size and 0<=c<self.size and (r,c) not in self.obstacle_cells

    def _build(self):
        n, A = self.n_free, self.N_ACTIONS; pi0 = 1./A
        self.det_fwd = np.zeros((n, A), dtype=np.int32)
        for s, (r,c) in enumerate(self.free_cells):
            for a, (dr,dc) in enumerate(self.ACTIONS):
                nr, nc = r+dr, c+dc
                self.det_fwd[s,a] = self.cell_to_idx[(nr,nc)] if self._is_free(nr,nc) else s

        n_sa = n*A
        self.P_dense = np.zeros((n_sa, n_sa))
        fwd = defaultdict(lambda: defaultdict(float))
        bwd = defaultdict(lambda: defaultdict(float))
        for s in range(n):
            for a in range(A):
                sp = self.s0 if s in self.absorbing_idx else self.det_fwd[s,a]
                for ap in range(A):
                    self.P_dense[sp*A+ap, s*A+a] += pi0
                    fwd[(s,a)][(sp,ap)] += pi0
                    bwd[(sp,ap)][(s,a)] += pi0
        assert np.allclose(self.P_dense.sum(0), 1.)
        self.fwd_trans = {k: list(v.items()) for k,v in fwd.items()}
        self.bwd_trans = {k: list(v.items()) for k,v in bwd.items()}

        self.p_sa = np.zeros((n, A, n), dtype=np.float32)
        for s in range(n):
            for a in range(A):
                sp = self.s0 if s in self.absorbing_idx else self.det_fwd[s,a]
                self.p_sa[s, a, sp] = 1.0

    def reset(self): self.state = self.s0; return self.state
    def step(self, a):
        sp = self.s0 if self.state in self.absorbing_idx else self.det_fwd[self.state, a]
        self.state = sp; return sp, 0., False, {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════
def make_X(env):
    n, A = env.n_free, env.N_ACTIONS
    X = np.zeros((n*A, n+A), dtype=np.float32)
    for s in range(n):
        for a in range(A):
            X[s*A+a, s] = 1.; X[s*A+a, n+a] = 1.
    return X

def policy_from_Q(Q):
    Q = Q - Q.max(axis=1, keepdims=True)
    p = np.exp(Q); p /= p.sum(axis=1, keepdims=True); return p

def greedy_policy_from_Q(Q):
    Qmax = Q.max(axis=1, keepdims=True)
    is_max = np.isclose(Q, Qmax, atol=1e-8).astype(float)
    return is_max / is_max.sum(axis=1, keepdims=True)

def stationary_dist(env, pol, d_init=None, tol=1e-7, max_iter=2000):
    """Exact stationary distribution via power iteration.
    d_init: warm-start from previous d (dramatically reduces iterations when
    policy changes slowly, as in Hazan/Soft-Q outer loops)."""
    T = np.einsum('sa,san->ns', pol, env.p_sa)
    d = d_init.copy() if d_init is not None else np.ones(env.n_free) / env.n_free
    d /= d.sum()
    for _ in range(max_iter):
        d_new = T @ d; d_new /= d_new.sum()
        if np.abs(d_new - d).max() < tol: break
        d = d_new
    return d

def H_empirical(env, pol, n_ep=40, max_st=1000):
    counts = np.zeros(env.n_free)
    for _ in range(n_ep):
        s = env.reset()
        for _ in range(max_st):
            a = np.random.choice(env.N_ACTIONS, p=pol[s]); s,_,_,_ = env.step(a)
            counts[s] += 1
    p = counts / (counts.sum()+1e-9); p = p[p>0]
    return float(-np.sum(p * np.log(p)))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Tabular EVE ground truth
# ═══════════════════════════════════════════════════════════════════════════════
def tabular_eve(env, n_iter=3000):
    P = env.P_dense; u = np.ones(env.n_free * env.N_ACTIONS)
    for _ in range(n_iter):
        x_ = u**(-1.)
        u = np.power((u.T @ P).T / (P @ x_), .5); u /= u.mean()
    return u


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Deep EVE
# ═══════════════════════════════════════════════════════════════════════════════
def eve_targets(env, mlp, frozen, X):
    n, A = env.n_free, env.N_ACTIONS
    q = mlp.predict_batch(X, frozen)
    targets = np.zeros(n*A)
    for s in range(n):
        for a in range(A):
            fv = [np.log(p) + q[sp*A+ap] for (sp,ap),p in env.fwd_trans[(s,a)] if p>1e-12]
            M = max(fv); fwd = M + np.log(sum(np.exp(v-M) for v in fv))
            bv = [np.log(p) - q[sb*A+ab] for (sb,ab),p in env.bwd_trans[(s,a)] if p>1e-12]
            M2 = max(bv); bwd = M2 + np.log(sum(np.exp(v-M2) for v in bv))
            targets[s*A+a] = 0.5*fwd - 0.5*bwd
    return targets

def get_policy_eve(env, mlp, X):
    return policy_from_Q(mlp.predict_batch(X).reshape(env.n_free, env.N_ACTIONS))

def train_eve(env, X, n_sweeps=500, lr=3e-3, eval_freq=50, seed=42):
    np.random.seed(seed)
    mlp = MLP([env.n_free+env.N_ACTIONS, 128, 128, 1], lr=lr, seed=seed)
    entropies, losses, steps_log = [], [], []
    for sweep in range(1, n_sweeps+1):
        frozen  = mlp.snapshot()
        targets = eve_targets(env, mlp, frozen, X)
        loss    = mlp.train_batch(X, targets, n_passes=1)
        losses.append(loss)
        if sweep % eval_freq == 0:
            H = H_empirical(env, get_policy_eve(env, mlp, X))
            entropies.append(H); steps_log.append(sweep)
            print(f"  [EVE]    sweep {sweep:>4}  H={H:.4f}  MSE={loss:.5f}")
    return mlp, entropies, steps_log, losses


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Soft-Q baseline
#     r(s) = -log d(s), smoothly mixed with reward_lr.
#     MLP learns Q via frozen-target soft Bellman regression.
# ═══════════════════════════════════════════════════════════════════════════════
def softq_bellman_targets(env, reward_s, Q_frozen, gamma=0.99):
    """Soft Bellman targets using frozen Q for bootstrap. Returns (n*A,) array."""
    n, A = env.n_free, env.N_ACTIONS
    V = Q_frozen.max(1) + np.log(np.exp(Q_frozen - Q_frozen.max(1,keepdims=True)).sum(1))
    Q_new = reward_s[:, None] + gamma * (env.p_sa * V[None, None, :]).sum(2)
    return Q_new.flatten()

def train_softq(env, X, n_iters=500, gamma=0.99, reward_lr=0.1,
                mlp_lr=3e-3, mlp_passes=10, eval_freq=50, seed=42):
    """
    Soft-Q with smoothly mixed intrinsic reward r(s) = -log d(s).
    reward_lr controls how quickly the reward adapts each iteration.
    mlp_passes: batched gradient passes per outer iteration.
    """
    np.random.seed(seed)
    n, A = env.n_free, env.N_ACTIONS
    mlp = MLP([n+A, 128, 128, 1], lr=mlp_lr, seed=seed)

    pol      = np.ones((n,A)) / A
    d        = stationary_dist(env, pol)
    reward_s = -np.log(d + 1e-9); reward_s -= reward_s.mean()

    entropies, steps_log = [], []
    for it in range(1, n_iters+1):
        # 1. Exact stationary distribution (warm-started from previous d)
        d = stationary_dist(env, pol, d_init=d)
        # 2. Smooth reward mixing
        r_new    = -np.log(d + 1e-9); r_new -= r_new.mean()
        reward_s = (1 - reward_lr) * reward_s + reward_lr * r_new
        # 3. Frozen soft-Bellman targets
        Q_frozen = mlp.predict_batch(X).reshape(n, A)
        targets  = softq_bellman_targets(env, reward_s, Q_frozen, gamma)
        # 4. Fit MLP (batched)
        mlp.train_batch(X, targets, n_passes=mlp_passes)
        # 5. Extract softmax policy
        pol = policy_from_Q(mlp.predict_batch(X).reshape(n, A))
        if it % eval_freq == 0:
            H = H_empirical(env, pol)
            entropies.append(H); steps_log.append(it)
            print(f"  [Soft-Q] iter  {it:>4}  H={H:.4f}")
    return pol, mlp, entropies, steps_log


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Hazan MaxEnt  (Frank-Wolfe on policy mixture)
#     r(s,a) = -log d_sa(s,a),  hard VI,  pi_mix update.
# ═══════════════════════════════════════════════════════════════════════════════
def hard_vi_targets(env, reward_sa, Q_frozen, gamma=0.99):
    """Hard Bellman targets: V(s)=max_a Q(s,a), target(s,a)=r(s,a)+gamma*E[V(s')]."""
    V = Q_frozen.max(axis=1)
    Q_new = reward_sa + gamma * (env.p_sa * V[None, None, :]).sum(2)
    return Q_new.flatten()

def train_hazan(env, X, n_iters=500, gamma=0.99, eta=0.2,
                mlp_lr=3e-3, mlp_passes=10, eval_freq=50, seed=42):
    """
    Hazan MaxEnt (Frank-Wolfe on policy mixture) with MLP Q-function.

    Each outer iteration:
      1. Compute exact d_sa(s,a) = d(s) * pi_mix(a|s)
      2. Set r(s,a) = -log d_sa(s,a)  (gradient of entropy w.r.t. d_sa)
      3. Hard VI: fit MLP Q to frozen-target Bellman equations for r
      4. Extract greedy pi* = argmax_a Q(s,a)
      5. Frank-Wolfe: pi_mix = (1-eta)*pi_mix + eta*pi*
    """
    np.random.seed(seed)
    n, A = env.n_free, env.N_ACTIONS
    mlp = MLP([n+A, 128, 128, 1], lr=mlp_lr, seed=seed)

    pi_mix = np.ones((n, A)) / A
    d_s    = np.ones(n) / n   # warm-start init

    entropies, steps_log = [], []
    for it in range(1, n_iters+1):
        # 1. Joint state-action distribution (warm-started)
        d_s  = stationary_dist(env, pi_mix, d_init=d_s)
        d_sa = d_s[:, None] * pi_mix          # (n, A)
        # 2. Reward = gradient of entropy
        reward_sa = -np.log(d_sa + 1e-9)
        reward_sa -= reward_sa.mean()
        # 3. Hard VI: frozen-target sweeps
        for _ in range(mlp_passes):
            Q_frozen = mlp.predict_batch(X).reshape(n, A)
            targets  = hard_vi_targets(env, reward_sa, Q_frozen, gamma)
            mlp.train_batch(X, targets, n_passes=1)
        # 4. Greedy policy from MLP Q
        Q_cur   = mlp.predict_batch(X).reshape(n, A)
        pi_star = greedy_policy_from_Q(Q_cur)
        # 5. Frank-Wolfe mixture update
        pi_mix = (1 - eta) * pi_mix + eta * pi_star

        if it % eval_freq == 0:
            H = H_empirical(env, pi_mix)
            entropies.append(H); steps_log.append(it)
            print(f"  [Hazan]  iter  {it:>4}  H={H:.4f}")
    return pi_mix, mlp, entropies, steps_log


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(0)
    env = PointMaze(); X = make_X(env)
    n_iters = 500; eval_freq = 50

    print("Computing tabular EVE ground truth...")
    u_tab  = tabular_eve(env)
    q_tab  = np.log(u_tab).reshape(env.n_free, env.N_ACTIONS)
    H_tab  = H_empirical(env, policy_from_Q(q_tab), n_ep=60, max_st=2000)
    H_unif = H_empirical(env, np.ones((env.n_free,env.N_ACTIONS))/env.N_ACTIONS)
    print(f"Tabular EVE : H={H_tab:.4f}")
    print(f"Uniform     : H={H_unif:.4f}")
    print(f"Max possible: {np.log(env.n_free):.4f}\n")

    print("=== Deep EVE ===")
    eve_mlp, eve_H, eve_steps, eve_loss = train_eve(
        env, X, n_sweeps=n_iters, lr=3e-3, eval_freq=eval_freq, seed=42)

    print("\n=== Soft-Q (reward_lr=0.1, gamma=0.99) ===")
    sq_pol, sq_mlp, sq_H, sq_steps = train_softq(
        env, X, n_iters=n_iters, gamma=0.99, reward_lr=0.1,
        mlp_lr=3e-3, mlp_passes=10, eval_freq=eval_freq, seed=42)

    print("\n=== Hazan MaxEnt (eta=0.2, gamma=0.99) ===")
    hz_pol, hz_mlp, hz_H, hz_steps = train_hazan(
        env, X, n_iters=n_iters, gamma=0.99, eta=0.2,
        mlp_lr=3e-3, mlp_passes=10, eval_freq=eval_freq, seed=42)

    print(f"\n── Final results ───────────────────────────")
    print(f"  Tabular EVE  : {H_tab:.4f}")
    print(f"  Deep EVE     : {eve_H[-1]:.4f}")
    print(f"  Soft-Q       : {sq_H[-1]:.4f}")
    print(f"  Hazan MaxEnt : {hz_H[-1]:.4f}")
    print(f"  Uniform      : {H_unif:.4f}")
    print(f"  Max H        : {np.log(env.n_free):.4f}")

    plot_all(env, eve_mlp, sq_pol, hz_pol, X,
             eve_H, eve_steps, eve_loss,
             sq_H, sq_steps, hz_H, hz_steps, H_tab, H_unif)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Plots
# ═══════════════════════════════════════════════════════════════════════════════
def draw_maze(ax, env, sz):
    for r in range(sz):
        for c in range(sz):
            col = ("#c0392b" if (r,c) in env.obstacle_cells
                   else "#f39c12" if (r,c) in env.absorbing_cells else "#ecf0f1")
            ax.add_patch(patches.Rectangle((c, sz-1-r), 1, 1,
                                           facecolor=col, edgecolor="#bdc3c7", lw=.5))

def draw_arrows(ax, env, pol, sz):
    dmap = {0:(0.,1.), 1:(0.,-1.), 2:(-1.,0.), 3:(1.,0.)}
    for s, (r,c) in enumerate(env.free_cells):
        ux, uy = dmap[np.argmax(pol[s])]
        ax.annotate("", xy=(c+.5+ux*.35, sz-1-r+.5+uy*.35),
                    xytext=(c+.5, sz-1-r+.5),
                    arrowprops=dict(arrowstyle="-|>", color="navy", lw=1.2, mutation_scale=8))

def plot_all(env, eve_mlp, sq_pol, hz_pol, X,
             eve_H, eve_steps, eve_loss,
             sq_H, sq_steps, hz_H, hz_steps,
             H_tab, H_unif,
             prefix="/mnt/user-data/outputs/eve"):

    max_ent = np.log(env.n_free); sr, sc = env.start_cell; sz = env.size
    eve_pol = get_policy_eve(env, eve_mlp, X)

    # ── Entropy comparison ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(eve_steps, eve_H, "o-", color="#2980b9", lw=2, ms=5, label="Deep EVE (FA)")
    ax.plot(sq_steps,  sq_H,  "s-", color="#e74c3c", lw=2, ms=5, label="Soft-Q  r=-log d(s), γ=0.99")
    ax.plot(hz_steps,  hz_H,  "^-", color="#8e44ad", lw=2, ms=5, label="Hazan MaxEnt (Frank-Wolfe)")
    ax.axhline(max_ent, color="#2c3e50", ls="--", lw=1.5, label=f"Max H = {max_ent:.3f}")
    ax.axhline(H_tab,   color="#27ae60", ls="-.", lw=1.5, label=f"Tabular EVE = {H_tab:.3f}")
    ax.axhline(H_unif,  color="#95a5a6", ls=":",  lw=1.5, label=f"Uniform = {H_unif:.3f}")
    ax.set_xlabel("Iteration / Sweep"); ax.set_ylabel("Empirical H(s)")
    ax.set_title("Deep EVE vs Soft-Q vs Hazan MaxEnt\n"
                 "11×11 PointMaze  (+obstacle, absorbing edges)")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3); plt.tight_layout()
    fig.savefig(f"{prefix}_entropy.png", dpi=130); plt.close(fig)

    # ── EVE loss ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(eve_loss, color="#2980b9", lw=1.3)
    ax.set_xlabel("Sweep"); ax.set_ylabel("MSE (log)"); ax.grid(True, alpha=0.3)
    ax.set_title("Deep EVE — Fixed-point regression loss"); plt.tight_layout()
    fig.savefig(f"{prefix}_loss.png", dpi=130); plt.close(fig)

    # ── Visitation heatmaps (4-way) ───────────────────────────────────────────
    def visits(pol, n_ep=60, max_st=1000):
        counts = np.zeros(env.n_free)
        for _ in range(n_ep):
            s = env.reset()
            for _ in range(max_st):
                a = np.random.choice(env.N_ACTIONS, p=pol[s]); s,_,_,_ = env.step(a)
                counts[s] += 1
        return counts / (counts.sum()+1e-9)

    def to_grid(v):
        g = np.zeros((env.size, env.size))
        for s, (r,c) in enumerate(env.free_cells): g[r,c] = v[s]
        return g

    obs_mask = np.array([[(r,c) in env.obstacle_cells for c in range(env.size)]
                          for r in range(env.size)])
    unif_pol = np.ones((env.n_free, env.N_ACTIONS)) / env.N_ACTIONS

    pols_vis   = [unif_pol, sq_pol, hz_pol, eve_pol]
    titles_vis = ["Uniform random", "Soft-Q  r=-log d(s)", "Hazan MaxEnt", "Deep EVE"]
    grids = [to_grid(visits(p)) for p in pols_vis]
    vmax  = max(g.max() for g in grids)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, g, title in zip(axes, grids, titles_vis):
        im = ax.imshow(np.ma.masked_where(obs_mask, g), origin="upper", cmap="hot",
                       interpolation="nearest", vmin=0, vmax=vmax)
        for r in range(env.size):
            for c in range(env.size):
                if (r,c) in env.obstacle_cells:
                    ax.add_patch(patches.Rectangle((c-.5,r-.5),1,1,facecolor="#2980b9",zorder=2))
                if (r,c) in env.absorbing_cells:
                    ax.add_patch(patches.Rectangle((c-.5,r-.5),1,1,
                                 facecolor="none",edgecolor="#f39c12",lw=2.5,zorder=3))
        ax.plot(sc, sr, "g^", ms=9, zorder=4, label="start")
        ax.set_title(title, fontsize=10); ax.axis("off")
        ax.legend(fontsize=7, loc="lower right")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="visit freq")
    fig.suptitle("State visitation  —  11×11 PointMaze  (orange=absorbing)", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{prefix}_visitation.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ── Policy arrows (3-way) ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (pol, title) in zip(axes, [(sq_pol,"Soft-Q"), (hz_pol,"Hazan MaxEnt"), (eve_pol,"Deep EVE")]):
        draw_maze(ax, env, sz); draw_arrows(ax, env, pol, sz)
        ax.plot(sc+.5, sz-1-sr+.5, "g^", ms=12, zorder=5, label="start")
        ax.set_xlim(0,sz); ax.set_ylim(0,sz); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=13); ax.legend()
    fig.suptitle("Greedy policies  —  11×11 PointMaze  (orange=absorbing)", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{prefix}_policy.png", dpi=130); plt.close(fig)

    print(f"Saved: {prefix}_entropy / _loss / _visitation / _policy .png")


if __name__ == "__main__":
    main()