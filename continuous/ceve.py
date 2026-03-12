"""
Deep EVE  —  11×11 PointMaze, + obstacle, absorbing edge states
===============================================================
Key fixes vs previous version:
  - One-hot state + one-hot action features  (not just r,c coords)
    -> gives MLP direct lookup-table capacity; verified to fit q_tab to MAE<0.001
  - 128-dim hidden layers
  - Frozen target network per sweep (exact FA analogue of tabular FP iteration)
  - Absorbing states at (0,5),(10,5),(5,0),(5,10) break symmetry -> non-trivial u
  - Agent has full access to dynamics (pre-computed P_dense, fwd/bwd transition lists)
"""

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict


# ── MLP + Adam ────────────────────────────────────────────────────────────────
class MLP:
    def __init__(self, sizes, lr=3e-3, seed=0):
        rng=np.random.default_rng(seed)
        self.layers=[[rng.standard_normal((fo,fi))*0.01, np.zeros(fo)]
                     for fi,fo in zip(sizes,sizes[1:])]
        self.lr=lr; self.b1=0.9; self.b2=0.999; self.ep=1e-8; self.t=0
        self.m=[[np.zeros_like(W),np.zeros_like(b)] for W,b in self.layers]
        self.v=[[np.zeros_like(W),np.zeros_like(b)] for W,b in self.layers]
        self._cache=None

    def forward(self, x):
        acts,zs=[x],[]
        for i,(W,b) in enumerate(self.layers):
            z=W@acts[-1]+b; zs.append(z)
            acts.append(np.tanh(z) if i<len(self.layers)-1 else z)
        self._cache=(acts,zs); return float(acts[-1][0])

    def backward(self, d):
        acts,zs=self._cache; delta=np.array([d]); self.t+=1
        for i in reversed(range(len(self.layers))):
            W,b=self.layers[i]; dW=np.outer(delta,acts[i]); db=delta.copy()
            if i>0: delta=(W.T@delta)*(1.-np.tanh(zs[i-1])**2)
            for j,g in enumerate([dW,db]):
                self.m[i][j]=self.b1*self.m[i][j]+(1-self.b1)*g
                self.v[i][j]=self.b2*self.v[i][j]+(1-self.b2)*g**2
                mh=self.m[i][j]/(1-self.b1**self.t)
                vh=self.v[i][j]/(1-self.b2**self.t)
                self.layers[i][j]-=self.lr*mh/(np.sqrt(vh)+self.ep)

    def predict_batch(self, X, w=None):
        """X:(N,d)->(N,).  w=None uses self.layers (live); else uses frozen weights."""
        h=X.T; wl=w if w is not None else self.layers
        for i,(W,b) in enumerate(wl):
            z=W@h+b[:,None]; h=np.tanh(z) if i<len(wl)-1 else z
        return h[0]

    def snapshot(self): return [[W.copy(),b.copy()] for W,b in self.layers]


# ── Environment ───────────────────────────────────────────────────────────────
class PointMaze:
    ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]; N_ACTIONS=4

    def __init__(self, size=11):
        self.size=size; cx=cy=size//2; arm=3
        self.obstacle_cells=set()
        for d in range(-arm,arm+1):
            self.obstacle_cells.add((cx+d,cy))
            self.obstacle_cells.add((cx,cy+d))
        # Absorbing: one step outside each arm tip (one free gap cell in between)
        self.absorbing_cells={(0,cy),(size-1,cy),(cx,0),(cx,size-1)}

        self.free_cells=[(r,c) for r in range(size) for c in range(size)
                         if (r,c) not in self.obstacle_cells]
        self.cell_to_idx={c:i for i,c in enumerate(self.free_cells)}
        self.n_free=len(self.free_cells)
        self.start_cell=(0,0); self.s0=self.cell_to_idx[self.start_cell]
        self.absorbing_idx={self.cell_to_idx[c] for c in self.absorbing_cells}
        self._build(); self.state=self.s0

    def _is_free(self,r,c):
        return 0<=r<self.size and 0<=c<self.size and (r,c) not in self.obstacle_cells

    def _build(self):
        n,A=self.n_free,self.N_ACTIONS; pi0=1./A
        self.det_fwd=np.zeros((n,A),dtype=np.int32)
        for s,(r,c) in enumerate(self.free_cells):
            for a,(dr,dc) in enumerate(self.ACTIONS):
                nr,nc=r+dr,c+dc
                self.det_fwd[s,a]=self.cell_to_idx[(nr,nc)] if self._is_free(nr,nc) else s
        # Dense joint transition matrix P[sp*A+ap, s*A+a] = P(sp,ap|s,a)
        n_sa=n*A
        self.P_dense=np.zeros((n_sa,n_sa))
        fwd=defaultdict(lambda:defaultdict(float))
        bwd=defaultdict(lambda:defaultdict(float))
        for s in range(n):
            for a in range(A):
                sp=self.s0 if s in self.absorbing_idx else self.det_fwd[s,a]
                for ap in range(A):
                    self.P_dense[sp*A+ap,s*A+a]+=pi0
                    fwd[(s,a)][(sp,ap)]+=pi0
                    bwd[(sp,ap)][(s,a)]+=pi0
        assert np.allclose(self.P_dense.sum(0),1.)
        self.fwd_trans={k:list(v.items()) for k,v in fwd.items()}
        self.bwd_trans={k:list(v.items()) for k,v in bwd.items()}

    def reset(self): self.state=self.s0; return self.state
    def step(self,a):
        sp=self.s0 if self.state in self.absorbing_idx else self.det_fwd[self.state,a]
        self.state=sp; return sp,0.,False,{}


# ── Features: one-hot(state) + one-hot(action) ───────────────────────────────
def make_X(env):
    """Feature matrix X[s*A+a] = one-hot(s) ++ one-hot(a).  Shape (n*A, n+A)."""
    n,A=env.n_free,env.N_ACTIONS
    X=np.zeros((n*A,n+A),dtype=np.float32)
    for s in range(n):
        for a in range(A):
            X[s*A+a,s]=1.; X[s*A+a,n+a]=1.
    return X


# ── EVE targets (frozen network) ─────────────────────────────────────────────
def compute_targets(env, mlp, frozen, X):
    n,A=env.n_free,env.N_ACTIONS
    q=mlp.predict_batch(X,frozen)   # (n*A,) frozen q values
    targets=np.zeros(n*A)
    for s in range(n):
        for a in range(A):
            fv=[np.log(p)+q[sp*A+ap] for (sp,ap),p in env.fwd_trans[(s,a)] if p>1e-12]
            M=max(fv); fwd=M+np.log(sum(np.exp(v-M) for v in fv))
            bv=[np.log(p)-q[sb*A+ab] for (sb,ab),p in env.bwd_trans[(s,a)] if p>1e-12]
            M2=max(bv); bwd=M2+np.log(sum(np.exp(v-M2) for v in bv))
            targets[s*A+a]=0.5*fwd-0.5*bwd
    return targets


# ── Policy  π*(a|s) ∝ exp(q(s,a)) ───────────────────────────────────────────
def get_policy(env,mlp,X):
    q=mlp.predict_batch(X).reshape(env.n_free,env.N_ACTIONS)
    q-=q.max(axis=1,keepdims=True)
    p=np.exp(q); p/=p.sum(axis=1,keepdims=True); return p

# ── Tabular EVE (ground truth) ────────────────────────────────────────────────
def tabular_eve(env,n_iter=3000):
    P=env.P_dense; u=np.ones(env.n_free*env.N_ACTIONS)
    for _ in range(n_iter):
        x_=u**(-1.); u=np.power((u.T@P).T/(P@x_),.5); u/=u.mean()
    return u

# ── Empirical entropy ─────────────────────────────────────────────────────────
def H_empirical(env,pol,n_ep=40,max_st=1000):
    counts=np.zeros(env.n_free)
    for _ in range(n_ep):
        s=env.reset()
        for _ in range(max_st):
            a=np.random.choice(env.N_ACTIONS,p=pol[s]); s,_,_,_=env.step(a); counts[s]+=1
    p=counts/(counts.sum()+1e-9); p=p[p>0]; return float(-np.sum(p*np.log(p)))


# ── Training ──────────────────────────────────────────────────────────────────
def train(n_sweeps=800, lr=3e-3, eval_freq=50, seed=42):
    np.random.seed(seed)
    env=PointMaze(); X=make_X(env)
    fdim=env.n_free+env.N_ACTIONS
    mlp=MLP([fdim,128,128,1],lr=lr,seed=seed)
    order=np.arange(env.n_free*env.N_ACTIONS)

    print("Computing tabular ground truth...")
    u_tab=tabular_eve(env)
    q_tab=np.log(u_tab).reshape(env.n_free,env.N_ACTIONS)
    pol_tab=np.exp(q_tab-q_tab.max(1,keepdims=True))
    pol_tab/=pol_tab.sum(1,keepdims=True)
    H_tab=H_empirical(env,pol_tab,n_ep=60,max_st=2000)
    print(f"Tabular EVE: H={H_tab:.4f}  max=log({env.n_free})={np.log(env.n_free):.4f}")
    print(f"q_tab: min={q_tab.min():.2f} max={q_tab.max():.2f} std={q_tab.std():.3f}\n")
    print(f"PointMaze {env.size}×{env.size}  free={env.n_free}  "
          f"absorbing={len(env.absorbing_idx)}  features={fdim}\n")

    entropies,losses,q_errs,sweeps_log=[],[],[],[]
    for sweep in range(1,n_sweeps+1):
        frozen=mlp.snapshot()
        targets=compute_targets(env,mlp,frozen,X)
        np.random.shuffle(order)
        sl=0.
        for idx in order:
            qv=mlp.forward(X[idx]); err=qv-targets[idx]
            mlp.backward(2.*err); sl+=err**2
        losses.append(sl/len(order))
        if sweep%eval_freq==0:
            pol=get_policy(env,mlp,X); H=H_empirical(env,pol)
            q_fa=mlp.predict_batch(X).reshape(env.n_free,env.N_ACTIONS)
            # centre both (q defined up to additive constant)
            qe=np.abs((q_fa-q_fa.mean())-(q_tab-q_tab.mean())).mean()
            entropies.append(H); q_errs.append(qe); sweeps_log.append(sweep)
            print(f"  sweep {sweep:>4}  H={H:.4f} (tab={H_tab:.4f})"
                  f"  MSE={losses[-1]:.5f}  |Δq|={qe:.4f}")
    print("\nDone.")
    return env,mlp,X,q_tab,entropies,q_errs,sweeps_log,losses,H_tab


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot(env,mlp,X,q_tab,entropies,q_errs,sweeps_log,losses,H_tab,
         prefix="eve"):
    max_ent=np.log(env.n_free)
    unif_pol=np.ones((env.n_free,env.N_ACTIONS))/env.N_ACTIONS
    H_unif=H_empirical(env,unif_pol,n_ep=50,max_st=1000)

    # entropy + q-error curve
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,6),sharex=True)
    ax1.plot(sweeps_log,entropies,"o-",color="#2980b9",lw=2,ms=4,label="Deep EVE (FA)")
    ax1.axhline(max_ent,color="#c0392b",ls="--",lw=1.5,label=f"Max H={max_ent:.3f}")
    ax1.axhline(H_tab,color="#27ae60",ls="-.",lw=1.5,label=f"Tabular EVE={H_tab:.3f}")
    ax1.axhline(H_unif,color="#7f8c8d",ls=":",lw=1.5,label=f"Uniform={H_unif:.3f}")
    ax1.set_ylabel("H(s)"); ax1.legend(fontsize=8); ax1.grid(True,alpha=0.3)
    ax1.set_title("Deep EVE — 11×11 PointMaze (+obstacle, absorbing edges)")
    ax2.semilogy(sweeps_log,q_errs,"s-",color="#e67e22",lw=2,ms=4,label="|Δq| FA vs tabular")
    ax2.set_ylabel("|q_FA - q_tab| (centred)"); ax2.set_xlabel("Sweep")
    ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)
    plt.tight_layout(); fig.savefig(f"{prefix}_entropy.png",dpi=130); plt.close(fig)

    # loss
    fig,ax=plt.subplots(figsize=(8,3))
    ax.semilogy(losses,color="#8e44ad",lw=1.3)
    ax.set_xlabel("Sweep"); ax.set_ylabel("MSE (log)"); ax.grid(True,alpha=0.3)
    ax.set_title("Fixed-point regression loss"); plt.tight_layout()
    fig.savefig(f"{prefix}_loss.png",dpi=130); plt.close(fig)

    # visitation heatmaps
    def visits(pol,n_ep=60,max_st=1000):
        counts=np.zeros(env.n_free)
        for _ in range(n_ep):
            s=env.reset()
            for _ in range(max_st):
                a=np.random.choice(env.N_ACTIONS,p=pol[s]); s,_,_,_=env.step(a); counts[s]+=1
        return counts/(counts.sum()+1e-9)
    def to_grid(v):
        g=np.zeros((env.size,env.size))
        for s,(r,c) in enumerate(env.free_cells): g[r,c]=v[s]
        return g

    eve_g=to_grid(visits(get_policy(env,mlp,X)))
    uni_g=to_grid(visits(unif_pol))
    obs_mask=np.array([[(r,c) in env.obstacle_cells for c in range(env.size)]
                        for r in range(env.size)])
    vmax=max(eve_g.max(),uni_g.max()); sr,sc=env.start_cell

    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,g,title in zip(axes,[uni_g,eve_g],["Uniform random","Deep EVE"]):
        im=ax.imshow(np.ma.masked_where(obs_mask,g),origin="upper",cmap="hot",
                     interpolation="nearest",vmin=0,vmax=vmax)
        for r in range(env.size):
            for c in range(env.size):
                if (r,c) in env.obstacle_cells:
                    ax.add_patch(patches.Rectangle((c-.5,r-.5),1,1,facecolor="#2980b9",zorder=2))
                if (r,c) in env.absorbing_cells:
                    ax.add_patch(patches.Rectangle((c-.5,r-.5),1,1,
                                 facecolor="none",edgecolor="#f39c12",lw=2.5,zorder=3))
        ax.plot(sc,sr,"g^",ms=10,zorder=4,label="start"); ax.set_title(title,fontsize=12)
        ax.axis("off"); ax.legend(fontsize=8,loc="lower right")
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label="visit freq")
    fig.suptitle("State visitation — 11×11 PointMaze (orange=absorbing)",fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{prefix}_visitation.png",dpi=130,bbox_inches="tight"); plt.close(fig)

    # policy arrows
    pol=get_policy(env,mlp,X); sz=env.size
    dmap={0:(0.,1.),1:(0.,-1.),2:(-1.,0.),3:(1.,0.)}
    fig,ax=plt.subplots(figsize=(7,7))
    for r in range(sz):
        for c in range(sz):
            col=("#c0392b" if (r,c) in env.obstacle_cells
                 else "#f39c12" if (r,c) in env.absorbing_cells else "#ecf0f1")
            ax.add_patch(patches.Rectangle((c,sz-1-r),1,1,facecolor=col,
                                           edgecolor="#bdc3c7",lw=.5))
    for s,(r,c) in enumerate(env.free_cells):
        ux,uy=dmap[np.argmax(pol[s])]
        ax.annotate("",xy=(c+.5+ux*.35,sz-1-r+.5+uy*.35),xytext=(c+.5,sz-1-r+.5),
                    arrowprops=dict(arrowstyle="-|>",color="navy",lw=1.2,mutation_scale=8))
    ax.plot(sc+.5,sz-1-sr+.5,"g^",ms=12,zorder=5,label="start")
    ax.set_xlim(0,sz);ax.set_ylim(0,sz);ax.set_aspect("equal");ax.axis("off")
    ax.set_title("Deep EVE — Greedy policy (orange=absorbing)"); ax.legend(); plt.tight_layout()
    fig.savefig(f"{prefix}_policy.png",dpi=130); plt.close(fig)

    # FA vs tabular q scatter
    q_fa=mlp.predict_batch(X).reshape(env.n_free,env.N_ACTIONS)
    qf=(q_fa-q_fa.mean()).flatten(); qt=(q_tab-q_tab.mean()).flatten()
    fig,ax=plt.subplots(figsize=(5,5))
    ax.scatter(qt,qf,s=4,alpha=0.4,color="#2980b9")
    lim=max(abs(qt).max(),abs(qf).max())*1.05
    ax.plot([-lim,lim],[-lim,lim],"r--",lw=1.5,label="ideal (FA=tabular)")
    ax.set_xlabel("q tabular (centred)"); ax.set_ylabel("q FA (centred)")
    ax.set_title("FA vs Tabular q-values"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); fig.savefig(f"{prefix}_qscatter.png",dpi=130); plt.close(fig)
    print(f"Saved: {prefix}_entropy/loss/visitation/policy/qscatter .png")


if __name__=="__main__":
    env,mlp,X,q_tab,entropies,q_errs,sweeps_log,losses,H_tab=train(
        n_sweeps=1000, lr=3e-3, eval_freq=50, seed=42)
    plot(env,mlp,X,q_tab,entropies,q_errs,sweeps_log,losses,H_tab)