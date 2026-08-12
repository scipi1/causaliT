# Query-norm capacity $F$ and the fan-in prior

Status: DESIGN (no code written yet). This document fixes the equations and the
config surface before implementation.

Objective: formalize solutions to inject prior information on the expected number of incoming edge in the DAG (its degree) 

Scope. It (i) re-derives the query normalisation in the paper notation, with the
global score scale written $F$, (ii) proves that $F$ is nothing other than a
*fan-in capacity measured in edges*, (iii) shows that the current `auto` rule
sets that capacity to $N$ (i.e. no cap at all), (iv) specifies how to declare a
smaller capacity as a **prior on the in-degree** without killing the
initialisation, and (v) derives how that prior should be set for the random
ER-$k$ DAGs produced by `scm_ds/random_scm.py`.

Supersedes the notation of `QUERY_FANIN_SCALE_BUDGET.md`. Here $F$ **multiplies**
the cosine, so $F=$ `sqrt(query_fanin_scale)`; see Appendix A.

---

## 1. Notation and the forward path

| symbol | meaning | code |
|---|---|---|
| $N$ | number of candidate parents (structural keys) | `n_keys` |
| $i,\ j$ | child (query row) index, candidate parent (key) index | |
| $\mathbf{k}_j$ | structural key of candidate parent $j$ | |
| $\mathbf{q}_i$ | raw structural query of child $i$ | |
| $\mathbf{u}_i$ | unit query direction, $\mathbf{u}_i=\mathbf{q}_i/\lVert\mathbf{q}_i\rVert$ | `F.normalize(query)` |
| $M_i$ | learnable per-node query norm, $M_i>0$ | `exp(query_norm_log_scale)` |
| $Z_i$ | per-row divisor of the raw dot product, eq (1) | (implicit) |
| $F$ | global score scale, a **multiplier** of $c_{ij}$ | `sqrt(query_fanin_scale)` |
| $E$ | structural embedding dimension (vanilla divisor $\sqrt{E}$) | `d_model` head dim |
| $\theta_{ij}$ | angle between $\mathbf{q}_i$ and $\mathbf{k}_j$ | |
| $c_{ij}$ | directional alignment $\cos\theta_{ij}=\langle\mathbf{u}_i,\mathbf{k}_j\rangle$ | |
| $\ell_{ij}$ | structural logit of edge $j\to i$ | `scores` |
| $s_{ij}$ | Binary-Concrete random variable, eq (2a) | |
| $\bar{s}_{ij}$ | stretched Binary Concrete, eq (2b) | |
| $z_{ij}$ | gate value, the clamp of $\bar{s}_{ij}$, eq (2c) | `gates` |
| $\pi_{ij}$ | edge posterior, $\Pr[z_{ij}>0]$, eq (2f) | |
| $T$ | additive gate offset | `init_edge_offset` |
| $\tau,\gamma,\zeta$ | Binary-Concrete temperature and stretch interval | `init_tau`, `init_gamma`, `init_zeta` |
| $\kappa$ | DERIVED: logit at which the gate opens, eq (2e) | |
| $\kappa_1$ | DERIVED: logit at which the gate saturates, eq (2e) | |
| $p$ | a target edge posterior | `query_centroid_max_p` |
| $x(p)$ | threshold logit for posterior $p$, eq (3) | (new helper) |
| $K$ | **capacity**: number of parents affordable at posterior $p$ | new |
| $K_{\text{init}}$ | capacity at $M_i=1$, i.e. the one that fixes $F$ | `query_capacity_init` |
| $K^{\star}$ | prior (declared) in-degree, i.e. the target capacity | `max_in_edges` |
| $\mu$ | target of the penalty on $M_i$ | `query_norm_target` |
| $\lambda_{\text{qn}}$ | weight of that penalty | `lambda_query_norm` |
| $f$ | realised fan-in of a row (number of parents held) | |

Vectors are bold. $\sigma(\cdot)$ is the logistic sigmoid, $[\,\cdot\,]_+$ the
ReLU, $\Pr[\cdot]$ a probability, $\Pi_{\mathcal{K}}$ the orthogonal projector
onto $\mathcal{K}=\mathrm{span}(\mathbf{k}_1,\dots,\mathbf{k}_N)$.

The structural keys are orthonormal by construction (orthonormal frame /
orthogonal key projection):

$$\langle \mathbf{k}_a,\mathbf{k}_b\rangle=\delta_{ab} \tag{0}$$

The logit is defined by the normalized dot product of keys and queries.
$$
\ell_{ij}=\frac{\langle\mathbf{q}_i,\mathbf{k}_j\rangle}{Z_i}=\frac{\lVert\mathbf{q}_i\rVert\,c_{ij}}{Z_i}
$$ 

The normalization constant is defined by
$$ 
Z_i=\frac{\lVert\mathbf{q}_i\rVert}{M_iF} \tag{1}
$$

N.B. the $\lVert\mathbf{q}_i\rVert$ cancels, which is the whole content of (1):

$$\ell_{ij}=M_i F c_{ij}\tag{2}$$

The query efforts are channeled into learning the alignment $c_{ij}$, which we scale by 
1) global scalar $F$, which depends on the *size* of the causal problem (currently $N$ but we want to make it selectable). In a nutshell, when we initialize the query with the key centroid, the logit magnitude per key decreases with the number of keys. This term ensures that the gates stays at a desired level at initialization independently on the size of the problem.
2) a learnable row-wise term. It is initialized at 1, so that the initial logit per keys are the one ensured by $F$. Over training, $M$ is incentivized to stay below a maximum value (default 1), or even decrease. This allow the scores to increase beyond the constraint of alignment, if needed, preventing the abrupt parents-misalignment due to maximum budget observed in the experiments.


Settings, code and comments

`free_query_embedding=True`: initialise a separate embedding for the query. It should be the default when we use "orthogonal_embeddings"

`query_norm`: applies the normalization but this should also be the default now

$T$ should be computed automatically when both cross-/self-attention are selected.
**[Implemented, 2026-08]** — `experiment.init_edge_offset: auto` resolves
$T = \ln(e^{x-\kappa}+2)$ at data-load time (`resolve_init_edge_offset`), the
value that lowers the cross gate onto the DIRECTED self posterior $p^*/2$ at
init; a float pins a legacy ablation, $0$ disables the offset.  Crucially, $T$
no longer enters $x(p^*)$ in ANY mode: $F$ is T-free
($x(p^*) = \mathrm{logit}(p^*) + \kappa$ everywhere below), so the capacity
calculus never sees the offset.




### The Hard-Concrete gate

**Source settings** (everything else in this section is derived from them): the
temperature $\tau$ and the stretch interval $[\gamma,\zeta]$ with $\gamma<0<1<\zeta$
(`init_tau`, `init_gamma`, `init_zeta`), the gate offset $T$
(`init_edge_offset`) and the target posterior $p$ (`query_centroid_max_p`).

The gate is built in three steps: a Binary-Concrete variable $s_{ij}$, its
stretch $\bar{s}_{ij}$ onto $[\gamma,\zeta]$, and the clamp $z_{ij}$. With
logistic noise $L=\log U-\log(1-U)$, $U\sim\mathcal{U}(0,1)$ (training time;
$L\equiv 0$ at eval time),

$$s_{ij}=\sigma\!\Bigl(\frac{\ell_{ij}-T+L}{\tau}\Bigr) \tag{2a}$$

$$\bar{s}_{ij}=s_{ij}\,(\zeta-\gamma)+\gamma \tag{2b}$$

$$z_{ij}=\mathrm{clamp}\bigl(\bar{s}_{ij},\ 0,\ 1\bigr) \tag{2c}$$

Inverting the affine map (2b) gives the two events that matter - the gate being
open and the gate being saturated - as thresholds on $s_{ij}$:

$$\bar{s}_{ij}>0\iff s_{ij}>\frac{-\gamma}{\zeta-\gamma},\qquad \bar{s}_{ij}\ge 1\iff s_{ij}\ge\frac{1-\gamma}{\zeta-\gamma} \tag{2d}$$

$s_{ij}$ in (2a) is an increasing function of $\ell_{ij}-T+L$, so applying
$\tau\,\mathrm{logit}(\cdot)$ to (2d) re-expresses both thresholds **in logit
units**. These are the only two derived constants of the gate:

$$\kappa=\tau\,\mathrm{logit}\!\Bigl(\frac{-\gamma}{\zeta-\gamma}\Bigr)=\tau\ln\!\Bigl(\frac{-\gamma}{\zeta}\Bigr),\qquad \kappa_1=\tau\,\mathrm{logit}\!\Bigl(\frac{1-\gamma}{\zeta-\gamma}\Bigr)=\tau\ln\!\Bigl(\frac{1-\gamma}{\zeta-1}\Bigr) \tag{2e}$$

The edge posterior follows: $z_{ij}>0$ iff $\bar{s}_{ij}>0$ iff
$\ell_{ij}-T+L>\kappa$ by (2d)-(2e), and $L$ is logistic and symmetric, so

$$\pi_{ij}=\Pr[z_{ij}>0]=\Pr\bigl[L>\kappa-(\ell_{ij}-T)\bigr]=\sigma\bigl(\ell_{ij}-T-\kappa\bigr) \tag{2f}$$

For the deterministic gate ($L\equiv 0$, eval time) the same two thresholds read

$$z_{ij}>0 \iff \ell_{ij}>T+\kappa \tag{2g}$$

$$z_{ij}=1 \iff \ell_{ij}\ \ge\ T+\kappa_1 \tag{2h}$$


Given the  current numerical default in `query_norm.py`: $\tau=0.5$, $\gamma=-1.1$, $\zeta=1.1$, one can compute $\kappa=0$ (it simplifies 2f) and $\kappa_1$. 


Note that $T+\kappa$ is the *opening* threshold (2g) and $T+\kappa_1$ the
*saturation* threshold (2h): for $\ell_{ij}\ge T+\kappa_1$ we have
$\bar{s}_{ij}\ge 1$, so $z_{ij}=1$ and the clamp is flat,
$\partial z_{ij}/\partial\ell_{ij}=0$. Any term that reaches the logit *through
the gate value* $z_{ij}$ therefore stops giving gradient on that edge, while
terms written on the posterior $\pi_{ij}$ - such as $\ell_0$, which uses (2f) -
keep pulling.

**What do those variables control?**

`query_centroid_max_p = 0.8209` $=\sigma(\kappa_1)$ at the defaults: the
posterior at which the deterministic gate saturates, i.e. the smallest $p$ whose
threshold logit already gives $z=1$. `DEFAULT_CENTROID_MAX_P = 0.9` sits
slightly above it.



### Ensure the desired posterior at the centroid

We said that, when the query is initialised at the centroid of $K$ keys, the
alignment per key is $1/\sqrt{K}$, so the posterior per key depends on $K$. Let
$p^*$ be a target posterior. By inverting 2f, it is not hard to see that 


$$\pi_{ij}\ge p^* \iff \ell_{ij}\ge \underbrace{\mathrm{logit}(p^*)+T+\kappa}_{x(p^*)} \tag{3}$$




### Centroid initialisation: setting $F$ from a declared capacity

`query_centroid_init` places the query at the centroid of a set $S$ of $K$ keys
($S$ = all $N$ keys today):

$$\mathbf{q}_i=\frac{1}{K}\sum_{j\in S}\mathbf{k}_j$$

Its squared norm is the dot product of the sum with itself. Expanding the
product of the two sums into a double sum, pulling out the scalar prefactor and
applying orthonormality (0) term by term:

$$\lVert\mathbf{q}_i\rVert^{2}=\Bigl\langle\frac{1}{K}\sum_{a\in S}\mathbf{k}_a,\ \frac{1}{K}\sum_{b\in S}\mathbf{k}_b\Bigr\rangle=\frac{1}{K^{2}}\sum_{a\in S}\sum_{b\in S}\langle\mathbf{k}_a,\mathbf{k}_b\rangle=\frac{1}{K^{2}}\sum_{a\in S}\sum_{b\in S}\delta_{ab}$$

The Kronecker delta kills every cross term $a\ne b$ and leaves $1$ on each of
the $K$ diagonal terms $a=b$, so the double sum equals $K$ and

$$\lVert\mathbf{q}_i\rVert^{2}=\frac{K}{K^{2}}=\frac{1}{K},\qquad \lVert\mathbf{q}_i\rVert=\frac{1}{\sqrt{K}}$$

Let's now compute the cosine between the centroid-initialized query and a given key $k_m$.
$$
\begin{align}
c_{im}&= \frac{q_i}{\Vert q_i\Vert_2}\cdot \frac{k_m}{\Vert k_m\Vert_2}\\
&=\frac{1}{\sqrt{K}}\sum_{j\in \mathcal{S}}\langle k_j, k_m \rangle\\
&=\frac{1}{\sqrt{K}}\sum_{j\in \mathcal{S}}\delta_{jm}\\
&=\begin{cases}
\frac{1}{\sqrt{K}}& \text{if} && m\in\mathcal{S}\\
0& \text{if} && m\notin\mathcal{S}\\
\end{cases}
\end{align}
\tag{3a}
$$


By substituting $c_{ij}=1/\sqrt{K}$ into (2) and setting $M_i=1$

$$
\begin{align}
F=\frac{\ell_{ij}}{M_i c_{ij}}\Big|_{M_i=1,\ c_{ij}=1/\sqrt{K}}=\ell_{ij}\sqrt{K}
\end{align}
$$

Here we use (3) to select the $\ell_{ij}$ needed to ensure a posterior $p^*$
$$
F=x(p^*)\sqrt{K} \tag{3b}
$$

where $x(p^*)$ is fully computable, given the parameters $T$, $\kappa$ and the chosen $p^*$.


## Implementation

**Current solution:** the code checks the size of the input data and set $K=N$, i.e. all possible keys.

**Desiderata:** restrict $K$ to the maximum number of *plausible* keys $K^{\star}$, in some datasets this information is available.

The *naive* approach would be to set $K=K^{\star}$ but the centroid initialization consider **ALL** $N$ keys, because we don't know a priori which ones are the real parent. Setting $K=K^{\star}<N$ would not ensure the desired posterior $p^*$. A better solution is to add a penalty term in the loss function so that, over training, the model learns to allocate the posterior $p^*$ on $K^{\star}$ keys.

The idea is to incorporate such regularization into the already trainable $M_i$, leaving $F=x(p^*)\sqrt{N}$ from (3b) fixed. The division of labour is then: $F$ absorbs the problem SIZE $N$ (it keeps the init posterior at $p^*$ whatever $N$ is), and $M_i$ absorbs the PRIOR, as the ratio $K^{\star}/N$, per row.




By substituting $F=x(p^*)\sqrt{N}$ in Lemma 1, equation 5, we obtain that a row can then reach $x(p^*)$ on $K^\star$ keys only if $M_i\ge\sqrt{K^\star/N}$. Therefore, the proposed regularization on $M_i$ becomes.

$$\mu=\sqrt{\frac{K^{\star}}{N}},\qquad R=\lambda_{\text{qn}}\sum_i\bigl[M_i-\mu\bigr]_+^{2} \tag{3c}$$

i.e. exactly `collect_query_norm_penalty` with `query_norm_target` set to $\mu$
instead of the current $1$ (which is (3c) at $K^{\star}=N$: a penalty that prices
no fan-in at all). Section 2 justifies the step from (5) to (3c) - why a row-wise
*scale* can cap a *count*; Section 3 collects the numbers the implementation
needs; Section 4 derives $K^{\star}$ for the synthetic DAGs.


**Scheduler Implementation**
$$K(t)=N+(K^{\star}-N)\,\rho(t),\qquad \rho:0\to1 \text{ monotone} \tag{3d}$$

$$\mu(t)=\sqrt{\frac{K(t)}{N}},\qquad R(t)=\lambda_{\text{qn}}\sum_{i}\bigl[M_i-\mu(t)\bigr]_+^{2} \tag{3e}$$

$\rho(0)=0$ gives $\mu=1$ exactly (today's behaviour), so the squeeze starts from
the un-modified state and no initialisation is ever destroyed. Linear in $K$ is
chosen rather than linear in $\mu$ so that the schedule is interpretable in edges.
Gradualness is not cosmetic: by the "limits of the claim" paragraph of Section 2,
a fast squeeze deflates a row instead of pruning it, because re-concentrating the
budget on fewer keys takes gradient steps that an instantaneous target does not
grant.


## 2. Theoretical backup

Section 1 fixes $F$ from *one* direction, the centroid. Training then moves the
query, so $c_{ij}$ becomes arbitrary and (3a) no longer applies. The claim behind
(3c) - that shrinking a *scale* $M_i$ caps a *number of parents* - therefore needs
a statement that holds for every direction. That is the content of this section,
and it is the theoretical contribution: it bounds the fan-in a row can hold, and
shows the bound is tight.

### Lemma 0 - The budget identity

By (0), for any query row $i$,

$$\sum_{j=1}^{N}c_{ij}^{2}=\bigl\lVert \Pi_{\mathcal{K}}\mathbf{u}_i\bigr\rVert^{2}\ \le\ 1 \tag{4a}$$

with equality iff $\mathbf{u}_i\in\mathrm{span}(\mathbf{k}_1,\dots,\mathbf{k}_N)$.
Equality holds in practice because (i) `query_centroid_init` puts
$\mathbf{q}_i$ in that span at epoch 0, and (ii) the gradient of any score-based
loss w.r.t. $\mathbf{q}_i$ is a linear combination of the $\mathbf{k}_j$, so the
iterates stay in the span. Q_NORM observation O2 measures
$\sum_j c_{ij}^2=1.0000$ from roughly epoch 9 onwards. We therefore use

$$\sum_{j=1}^{N}c_{ij}^{2}=1 \tag{4b}$$

This is the crux: **the directional budget of a query row is conserved.** A row
cannot align with many keys at once; it can only redistribute a fixed unit of
squared cosine. The only quantity that can inflate every logit of a row at once
is $M_i$ - a scale, not a redistribution - which is why $M_i$ is what the prior
prices.

### Lemma 1 - Capacity bound

Fix a row $i$ and a target posterior $p^*$, and let
$\mathcal{A}_i(p^*)=\{\,j:\pi_{ij}\ge p^*\,\}$ be the parents it holds at that
confidence. Then

$$\bigl\lvert\mathcal{A}_i(p^*)\bigr\rvert\ \le\ \left(\frac{M_i\,F}{x(p^*)}\right)^{2} \tag{5}$$


*Proof* 

By (2) and (3), $j\in\mathcal{A}_i(p^*)$ iff $M_iFc_{ij}\ge x(p^*)$, i.e.
iff

$$c_{ij}\ \ge\ \underbrace{\frac{x(p^*)}{M_i\,F}}_{c^*}>0$$

Where the last inequality follows by the solver imposition of positive $x(p^*)$. Each $c^2_{ij}$ in (4b) is therefore at least $(c^*)^2$, if we have $\bigl\lvert\mathcal{A}_i(p^*)\bigr\rvert$ terms in total, it follows that

$$
\begin{align}
\bigl\lvert\mathcal{A}_i(p^*)\bigr\rvert (c^*)^2&\leq 1 \tag{6.1}\\
\bigl\lvert\mathcal{A}_i(p^*)\bigr\rvert &\leq \frac{1}{(c^*)^2}=\left(\frac{M_i\,F}{x(p^*)}\right)^{2} \tag{6.2}\\
\end{align}
$$

The bound is also **attained**, so it is not vacuous: for any parent set $S$ with
$\lvert S\rvert=K\le(M_iF/x(p^*))^{2}$, the direction
$\mathbf{u}_i=K^{-1/2}\sum_{j\in S}\mathbf{k}_j$ has $c_{ij}=1/\sqrt{K}\ge c^*$
for $j\in S$ and $c_{ij}=0$ otherwise, by the same computation as (3a). A row can
therefore sit exactly at capacity, which is what makes $\mu$ a meaningful target
rather than a loose inequality. $\blacksquare$

**Remark (what a row at capacity looks like).** At attainment the non-parents sit
at $c_{ij}=0$, i.e. $\pi_{ij}=\sigma(-T-\kappa)$ - $0.25$ in split mode, $0.5$ in
homogeneous mode - and $z_{ij}=0$ by (2g) whenever $T\ge 0$. So capacity $K$ buys
"$K$ parents at posterior $p^*$, everything else at the neutral floor". Pushing
non-parents *below* that floor requires $c_{ij}<0$, which spends budget from (4b)
and therefore *reduces* the number of parents affordable at $p^*$.

**Limits of the claim.** Lemma 1 bounds the *count* of edges above $p^*$; it says
nothing about *which* edges, and nothing about posteriors below $p^*$. In
particular $M_i$ is a scale, not a redistributor: lowering it deflates every logit
of the row uniformly and does not preferentially delete the weakest edge. The
prior becomes a *fan-in* prior only indirectly - with a smaller $M_i$, keeping any
edge above $x(p^*)$ needs a larger $c_{ij}$, and by (4b) larger cosines can only be
bought by concentrating the budget on fewer keys. That indirection is the
mechanism claim under test, and the reason the squeeze must be gradual; see the
first falsifier in Section 5.

**[Human revised ends here]**

---

## 3. Implementation notes

Three numbers the implementation needs, all consequences of Section 1-2.

### 3.1 The init gate, and why $F$ must not be shrunk (the guard)

> **Note (2026-08):** the numbers below are the historical arm with $T = \ln 3$
> folded into $x(p^*)$ ($x = 2.6211$).  The resolved derivation is now T-free,
> $x(p^*) = 1.5225$, so eq (8) holds with $T = 0$ — and with $\kappa = 0$ the
> init gate then survives at EVERY size ($z > 0$ for any $\ell > 0$), which
> strengthens the conclusion.  The point of the guard is unchanged: do not
> shrink $F$.

The tempting shortcut is to set $F=x(p^*)\sqrt{K^{\star}}$ and be done. It fails,
because the initialisation is the centroid of **all** $N$ keys, where
$c_{ij}=1/\sqrt{N}$ by (3a), not $1/\sqrt{K^{\star}}$. Substituting into (2) with
$M_i=1$:

$$\ell_{\text{init}}=x(p^*)\sqrt{\frac{K^{\star}}{N}},\qquad \pi_{\text{init}}=\sigma\!\left(x(p^*)\sqrt{\frac{K^{\star}}{N}}-T-\kappa\right) \tag{8a}$$

$$z_{\text{init}}>0 \iff x(p^*)\sqrt{\frac{K^{\star}}{N}}>T+\kappa \qquad\text{by (2g)} \tag{8b}$$

The init signal decays like $\sqrt{K^{\star}/N}$. Split mode, $p^*=0.8209$
($x=2.6211$, $T=1.0986$, $\kappa=0$), $K^{\star}=10$:

| $N$ | $\ell_{\text{init}}$ | $\pi_{\text{init}}$ | $z_{\text{init}}$ |
|---|---|---|---|
| $10$ | $2.621$ | $0.821$ | $1.00$ |
| $50$ | $1.172$ | $0.518$ | $0.08$ |
| $100$ | $0.829$ | $0.433$ | $\mathbf{0}$ |
| $400$ | $0.414$ | $0.335$ | $\mathbf{0}$ |

By (8b) the init gate survives only while $K^{\star}/N>(T/x(p^*))^{2}=0.1757$,
i.e. $K^{\star}>70$ at $N=400$: below that the deterministic gate passes
**nothing** at initialisation - the failure mode already recorded for
`query_fanin_scale` $=12.07$ ($F=3.47$) in `QUERY_FANIN_SCALE_BUDGET.md`. Keeping
$F=x(p^*)\sqrt{N}$ and moving $\mu$ instead, as in (3c), avoids this entirely
($\mu(0)=1$, so epoch 0 is bit-identical to today). Eq (8) is nevertheless
implemented as a **startup guard**, because a user may still set
`query_capacity_init` by hand.

### 3.2 The schedule clock

$\rho(t)$ in (3d) must be measured in *structure-phase* epochs counted from an
anchor installed at the FIRST structure phase, not from the global epoch. Under
adaptive training everything runs in a single `fit()`, so `current_epoch` is
global and a raw window is consumed by the (long) reconstruct phases - the lesson
recorded in the `query_norm.py` header ("the budget saturates much LATER than any
preset window") and the fix already used for `_descendant_warmup_anchor` in
`adaptive_trainer.py`. The penalty acts on the structural stream
(`query_norm_log_scale` is routed as a structural parameter), so structural time
is the correct clock.

### 3.3 Calibrating $\lambda_{\text{qn}}$

Substituting the minimal scale $M_i=\sqrt{f/N}$ that holds $f$ parents into (3c)
gives the price a row pays for exceeding the prior:

$$R_i(f)=\frac{\lambda_{\text{qn}}}{N}\Bigl(\sqrt{f}-\sqrt{K^{\star}}\Bigr)^{2},\qquad f>K^{\star} \tag{11}$$

$$\frac{\partial R_i}{\partial f}=\frac{\lambda_{\text{qn}}}{N}\left(1-\sqrt{\frac{K^{\star}}{f}}\right)\ \ge 0,\qquad \frac{\partial^{2} R_i}{\partial f^{2}}=\frac{\lambda_{\text{qn}}}{2N}\,\sqrt{K^{\star}}\,f^{-3/2}>0 \tag{12}$$

$R_i(K^{\star})=0$ with zero slope, and (12) is positive and convex, so the prior
is free up to $K^{\star}$ and then hardens smoothly - the behaviour wanted from a
*prior* rather than a constraint, since a genuinely high-in-degree node can buy
its parents whenever the structural loss pays more than (11).

The catch is that $\lambda_{\text{qn}}$ **does not transfer across $N$**, for two
independent reasons:

1. *the total grows with $N$.* (3c) is a SUM over rows while the reconstruction
   loss is a mean, so at the start of the squeeze the penalty is
   $\approx N(1-\mu)^{2}$: $0.09$ at $n=10,K^{\star}=8$ but $284$ at
   $n=400,K^{\star}=10$ per unit $\lambda_{\text{qn}}$. The same
   $\lambda_{\text{qn}}$ is a nudge in one setting and dominant in the other.
2. *one parent gets cheaper as $N$ grows.* The $1/N$ in (11) is not incidental:
   capacity enters as the FRACTION $K/N$, so one edge is worth $1/N$ of scale.
   Five extra parents cost $1.09\cdot10^{-2}\lambda_{\text{qn}}$ at $n=50$ but
   $1.31\cdot10^{-3}\lambda_{\text{qn}}$ at $n=400$ - $8\times$ less, exactly
   where the prior matters most.

The two compound: as $N$ grows the penalty pushes every $M_i$ down harder (1)
while pricing the over-spend it exists to price ever less (2). Switching the
reduction from `sum` to `mean` divides everything by $N$ and therefore fixes (1)
only; (2) survives untouched, because it is a *ratio* between two settings. What
fixes both is to write the penalty in **capacity units** instead of scale units.
Using $K_i=M_i^{2}N$ from Lemma 1 and $\mu^{2}=K^{\star}/N$,

$$\frac{K_i-K^{\star}}{K^{\star}}=\left(\frac{M_i}{\mu}\right)^{2}-1 \qquad\Longrightarrow\qquad R=\lambda_{\text{qn}}\,\mathrm{mean}_i\left[\left(\frac{M_i}{\mu}\right)^{2}-1\right]_+^{2} \tag{12a}$$

Same parameter, same target $\mu$, same activation condition $M_i>\mu$; the
excess is simply measured as a ratio, so "this row holds 50 % more parents than
the prior" costs the same at every $n$. In code it is one line inside
`overspend_penalty`. It is *not* the default (it changes every existing run) and
it does not harden the prior at $f=K^{\star}$ - both forms are flat there. Until
it is adopted, $\lambda_{\text{qn}}$ must be an explicit axis of every
experiment, never inherited.

---

## 4. Choosing $K^{\star}$ for random ER-$k$ DAGs

Section 1 assumes the prior in-degree is known. For the synthetic benchmarks it
is *not* handed over directly: the sampler is parameterised by a mean, and the
maximum is what a fan-in prior must cover.

### What the sampler actually controls

`scm_ds/random_scm.py::_sample_dag` draws a random topological order, then picks
$m=\mathrm{round}(\text{degree}\cdot n)$ edges uniformly **without replacement**
from the $n_{\text{slots}}=\binom{n}{2}$ forward pairs. So `degree` fixes the
**total** edge count; in-degree, out-degree and depth are emergent. In
particular `degree` is a **mean** in-degree, not a maximum:

$$\mathbb{E}\bigl[\text{in-degree averaged over nodes}\bigr]=\frac{m}{n}=\text{degree} \tag{13a}$$

### In-degree law

Indexing nodes by topological position $i=0,\dots,n-1$, node $i$ has $i$
admissible parents, each present with probability
$p_e=m/\binom{n}{2}=2\,\text{degree}/(n-1)$. The draw is without replacement, so
the exact law is hypergeometric; the Binomial below is the standard
approximation, exact as $n\to\infty$:

$$D_i\sim\mathrm{Bin}(i,\,p_e),\qquad p_e=\frac{2\,\text{degree}}{n-1},\qquad \mathbb{E}[D_i]=i\,p_e \tag{13b}$$

$$\mathbb{E}[D_{n-1}]=2\,\text{degree} \tag{13c}$$

Averaging (13b) over positions returns (13a), and the root fraction is
$\Pr[D=0]=\frac{1}{n}\sum_i(1-p_e)^{i}$, which is precisely the existing
`expected_er_roots`. The **pooled** in-degree CDF and its quantile are

$$\Pr[D\le d]=\frac{1}{n}\sum_{i=0}^{n-1}\Pr\bigl[\mathrm{Bin}(i,p_e)\le d\bigr] \tag{14a}$$

$$Q(\alpha)=\min\bigl\{\,d:\Pr[D\le d]\ge\alpha\,\bigr\} \tag{14b}$$

$Q(\alpha)$ is the recommended estimator for `max_in_edges`: *the prior covers a
fraction $\alpha$ of the nodes*. `er_indegree_quantile(n_nodes, degree, alpha)`
will implement (14). Evaluated, and compared with the `p95 in` column of the
Monte Carlo below, which estimates the same quantity up to the discreteness of
$d$ and the difference between a pooled quantile and the mean of per-graph
quantiles:

| $n$ | degree | $Q(0.95)$ from (14) | MC `p95 in` | diff |
|---|---|---|---|---|
| $10$ | $4$ | $8$ | $7.71$ | $+0.29$ |
| $50$ | $4$ | $9$ | $9.10$ | $-0.10$ |
| $100$ | $4$ | $10$ | $9.33$ | $+0.67$ |
| $400$ | $1$ | $3$ | $3.03$ | $-0.03$ |
| $400$ | $2$ | $5$ | $5.34$ | $-0.34$ |
| $400$ | $4$ | $10$ | $9.56$ | $+0.44$ |

All within one edge, so (14) is a usable closed form and the Binomial
approximation to the without-replacement draw is harmless at these sizes.

### Monte-Carlo verification (300 DAGs per row, exact sampler rule)

| $n$ | degree | $m$ | slots | fill % | mean in | p95 in | max in | depth | roots % |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | 10 | 45 | 22.2 | 1.00 | 2.66 | 3.08 | 2.9 | 40.3 |
| 20 | 1 | 20 | 190 | 10.5 | 1.00 | 2.81 | 3.64 | 3.5 | 41.8 |
| 50 | 1 | 50 | 1225 | 4.1 | 1.00 | 3.11 | 4.38 | 4.5 | 42.6 |
| 100 | 1 | 100 | 4950 | 2.0 | 1.00 | 3.11 | 4.87 | 5.0 | 42.9 |
| 400 | 1 | 400 | 79800 | 0.5 | 1.00 | 3.03 | 6.01 | 6.3 | 43.1 |
| 10 | 2 | 20 | 45 | 44.4 | 2.00 | 4.42 | 4.91 | 4.8 | 21.9 |
| 20 | 2 | 40 | 190 | 21.1 | 2.00 | 4.84 | 5.83 | 5.9 | 22.8 |
| 50 | 2 | 100 | 1225 | 8.2 | 2.00 | 5.21 | 6.96 | 7.1 | 23.6 |
| 100 | 2 | 200 | 4950 | 4.0 | 2.00 | 5.37 | 7.67 | 8.3 | 24.2 |
| 400 | 2 | 800 | 79800 | 1.0 | 2.00 | 5.34 | 9.16 | 10.0 | 24.5 |
| 10 | 4 | 40 | 45 | **88.9** | 4.00 | 7.71 | 8.18 | 8.1 | 11.3 |
| 20 | 4 | 80 | 190 | 42.1 | 4.00 | 8.46 | 9.79 | 9.6 | 11.8 |
| 50 | 4 | 200 | 1225 | 16.3 | 4.00 | 9.10 | 11.44 | 12.2 | 11.8 |
| 100 | 4 | 400 | 4950 | 8.1 | 4.00 | 9.33 | 12.85 | 14.0 | 12.3 |
| 400 | 4 | 1600 | 79800 | 2.0 | 4.00 | 9.56 | **14.61** | **17.6** | 12.5 |

The `roots %` column reproduces `expected_er_roots` ($43/24/12$ % for
ER-1/2/4) and `mean in` reproduces (13a) exactly, so the law (13) is verified
against the sampler.

### Three conclusions

1. **ER-4 does not cap the in-degree at 4.** The mean is $4$ by construction,
   but the maximum is $8.2$ at $n=10$ and $14.6$ at $n=400$. Setting
   `max_in_edges` $=$ `degree` would starve the downstream nodes, whose expected
   in-degree is already $2\,\text{degree}$ by (13c). Use $Q(0.95)$ from (14) -
   about $10$ for ER-4 - or the realised maximum for an oracle arm.
2. **`degree` says nothing about depth.** ER-4 has longest path $8$ at $n=10$
   and $17.6$ at $n=400$, not $4$.
3. **ER-4 at $n=10$ fills 88.9 % of all admissible forward slots** - it is
   nearly the complete DAG on 10 nodes, so $Q(0.95)=8=N-2$ and any fan-in prior
   there is close to vacuous. An independent reason to expect the effect only at
   larger $n$, and a reason not to read a null result at $n=10$ as a refutation.

### Prior strength at a glance

Split mode, $p^*=0.8209$, $K^{\star}=Q(0.95)$ taken from the MC column:

| $n$ | degree | $K^{\star}$ | $\mu=\sqrt{K^{\star}/N}$ | logit shrink $1/\mu$ | $R_i(K^{\star}+5)/\lambda_{\text{qn}}$ |
|---|---|---|---|---|---|
| $50$ | $4$ | $9.1$ | $0.427$ | $2.3$ | $1.09\cdot10^{-2}$ |
| $100$ | $4$ | $9.3$ | $0.305$ | $3.3$ | $5.36\cdot10^{-3}$ |
| $400$ | $4$ | $9.6$ | $0.155$ | $6.5$ | $1.31\cdot10^{-3}$ |
| $400$ | $2$ | $5.3$ | $0.115$ | $8.7$ | $2.06\cdot10^{-3}$ |

The last column is the whole $\lambda_{\text{qn}}$-calibration problem in one
place: the same five extra parents cost $8\times$ less at $n=400$ than at
$n=50$.

---

## 5. Config surface, guards and pre-registered predictions

### Config

No $F$: everything in edges.

```yaml
experiment:
  normalize_query: true
  query_centroid_max_p: 0.8209   # the posterior p* of eq (3); = sigmoid(kappa_1), eq (2e)
  query_capacity_init: auto      # K_init at M_i = 1, in EDGES; auto = n_keys -> sets F, eq (3b)
  max_in_edges: null             # K* prior capacity in EDGES; null = disabled (= n_keys) -> sets mu, eq (3c)
  query_fanin_scale: auto        # DERIVED = F^2. An explicit float still wins (legacy configs).
training:
  lambda_query_norm: 1.0e-3      # must be re-calibrated with N, see Section 3.3
  max_in_edges_anneal_epochs: 0  # STRUCTURE epochs to squeeze n_keys -> max_in_edges (0 = immediate)
  max_in_edges_anneal_idle_epochs: 0
```

Resolution order, in `populate_seq_lengths_from_dataset` where $N$ is known:

1. an explicit numeric `query_fanin_scale` is honoured verbatim and everything
   below is skipped (byte-identical legacy behaviour);
2. else $K_{\text{init}}=$ `query_capacity_init` (`auto` $\to N$) and
   `query_fanin_scale` $=F(K_{\text{init}})^{2}=K_{\text{init}}\,x(p^*)^{2}$ by (3b);
3. $K^{\star}=$ `max_in_edges` (`null` $\to N$), $\mu_{\text{end}}=\sqrt{K^{\star}/N}$ by (3c);
4. validate $1\le K_{\text{init}}\le N$ and $1\le K^{\star}\le N$;
5. **init-gate guard**: compute $\ell_{\text{init}},\pi_{\text{init}},z_{\text{init}}$
   from (8) with $K^{\star}\to K_{\text{init}}$; if $z_{\text{init}}=0$ while
   `query_centroid_init` is on, raise with the numbers and the two fixes (raise
   $K_{\text{init}}$, or drop $T$).

Startup log line, readable, replacing the raw `query_fanin_scale`:

```
[query-norm] n_keys=400 | capacity_init=400 edges | max_in_edges=10 edges
             F=52.42 (query_fanin_scale=2748.0) | x(p*)=2.6211 @ p*=0.8209 | mu_end=0.1581
             pi_init=0.8209 z_init=1.000 | anneal=30 struct epochs
```

Arms:

| arm | `query_capacity_init` | `max_in_edges` | effect |
|---|---|---|---|
| control (today) | `auto` | `null` | bit-identical to current behaviour |
| annealed prior | `auto` | $Q(0.95)$ | the prior of (3c)-(3e) |
| oracle | `auto` | realised max in-degree | same, upper bound on the gain |
| hard prior (ablation) | $Q(0.95)$ | $Q(0.95)$ | shrinks $F$ too; the guard of 3.1 may refuse it |

### Interaction with the rest of the system

* **`query_centroid_init`** - unchanged by construction, since $\mu(0)=1$ and
  $F=x(p^*)\sqrt{N}$. It is what breaks if $F$ is shrunk instead, eq (8).
* **homogeneous mode** - no cross gate exists, so `init_edge_offset` has no
  consumer ($T=0$).  Since 2026-08 F is T-free in ALL modes ($x(p^*)=1.5225$,
  never the $2.6211$ of the old T-in-F derivation), and with the offset off the
  init gate can never die by (2g). All equations hold with $T=0$.
* **split mode, prior x offset** - a pinned $T>0$ raises the CROSS-side
  threshold logit from $x(p^*)$ to $x(p^*)+T$, so the capacity that $\mu(t)$
  prices (Lemma 1, calibrated at $x(p^*)$) is under-delivered on the cross gate
  by $(x/(x+T))^2$ (~0.2 at the matched value): the prior would over-prune
  S->X parents.  `FaninPriorSchedule` therefore anneals
  $T(t)=T_0\,(1-\rho(t))$ to zero on the SAME structure clock
  (`training.anneal_edge_offset: auto`, the default, only alongside an active
  prior; `true` forces it without one, `false` disables): the directed-level
  init balance holds early, and the calibration above is exact again at the end
  of the squeeze - where, the direction gate having committed, the end state is
  again balanced at $p^*$.  Side effect: as $T\to 0$ the cross non-parent floor
  rises from $\sigma(-T)$ to $\sigma(0)=0.5$; any edge with a negative logit
  stays below the 0.5 eval threshold.
* **$\ell_0$ / `lambda_l0`** - complementary: $\ell_0$ prices *posterior mass*,
  the fan-in prior prices *how many edges can be confident at once*. The prior
  costs nothing at eval time and does not shift the $\ell_0$ threshold.
* **gradient routing** - `query_norm_log_scale` is a structural parameter, so
  the squeeze acts on the structural stream only. Unchanged.
* **`shared_query`** - a single `log_scale` shared by the cross and self blocks
  must be de-duplicated by parameter id when the target is written,
  symmetrically to `collect_query_norm_penalty`.
* **adaptive phases** - anchor as in Section 3.2; the target must be re-applied on
  every phase switch (a cheap idempotent write in `on_train_epoch_start`).
* **sweeps** - `max_in_edges` is a *prior*, therefore an OPT-IN extra arm, never
  a silent default: the main arms must not receive an inductive bias the
  baselines lack. In the dagsweep it can be derived from the `dag.degree`
  *generation setting* via (14); deriving it from the realised graph is
  ground-truth leakage and admissible only in an arm explicitly named `oracle`.
* **`validate_dimensions` / `fanin_saturating`** - both currently rewrite $F$
  from $N$; they must be made capacity-aware so they cannot silently restore
  $K=N$ and delete the feature.

### Predictions and falsifiers (recorded before the runs)

Primary prediction: with $K^{\star}=Q(0.95)$ annealed as in (3d)-(3e), DAG-retrieval
metrics (SHD, `precision_cross`, `tpr`) improve, and the improvement **grows
with $N$** - about zero at $n=10$ (Section 4, conclusion 3) and largest at
$n=400$.

Mechanism prediction: the mean learned in-degree falls towards $K^{\star}$ while
$M_i$ settles near $\mu_{\text{end}}$ for most rows and above it for the
genuinely high-in-degree (late topological) rows, i.e. the buy-out in (11) is
exercised selectively.

Falsifiers:

* `tpr` falls roughly uniformly across rows while $M_i\to\mu$ for all rows: the
  squeeze is *deflating* rows rather than pruning them (the risk flagged at the
  end of Section 2), and the mechanism claim is refuted;
* $M_i$ stays at $1$ and nothing moves: $\lambda_{\text{qn}}$ is too small
  (Section 3.3) - re-run on the $\lambda_{\text{qn}}$ axis before concluding
  anything;
* the effect does not scale with $N$: the capacity of Lemma 1 is not the binding
  constraint, and the feature should be abandoned rather than tuned.

New diagnostics required to evaluate the above: `query_norm/target_mu` ($\mu(t)$),
`query_norm/cap_target_edges` ($K(t)$ from (3d)),
`query_norm/cap_actual_edges` ($\mathrm{mean}_i M_i^{2}N$), and the learned
in-degree distribution alongside the existing `mean_M` / `max_M`.

---

## Appendix A. Where every symbol lives in the code

Reachability of the Section 1 quantities, as of today. Everything is set from
`experiment.*` unless the config key says otherwise; `training.*` keys are read
by the forecaster.

| symbol | config key | code path |
|---|---|---|
| $N$ | (none: derived from the dataset) | `populate_seq_lengths_from_dataset`, $N=$ `n_source + n_input` |
| $F$ | `query_fanin_scale` (holds $F^{2}$), `auto` | `resolve_query_fanin_scale` -> module attr `query_fanin_scale`; used as `scale_s = math.sqrt(...)` in `gated_cross_attention.py` / `gated_self_attention.py` / `commutator_self_attention.py` |
| $M_i$ | `query_norm_learnable`, `query_norm_init_scale` | `make_query_norm_log_scale` -> parameter `query_norm_log_scale`; applied in `apply_query_norm` as `q_hat * M` |
| $\mu$ | `query_norm_target` | module attr `query_norm_target`, read by `overspend_penalty` / `collect_query_norm_penalty` |
| $\lambda_{\text{qn}}$ | `training.lambda_query_norm` | `attention_selector_forecaster.py` / `self_selector_forecaster.py`, `qn_reg = lambda * penalty` |
| $\tau$ | `init_tau` | module attr `self.beta` (the code still uses the old name $\beta$ for the temperature) |
| $\gamma,\zeta$ | `init_gamma`, `init_zeta` | module attrs `self.gamma`, `self.zeta` |
| $T$ | `init_edge_offset` | module attr `edge_offset`; resolved by `resolve_init_edge_offset` (`auto` = matched $\ln(e^{x-\kappa}+2)$, float = pinned, $0$ = off); never enters $F$; inert when `homogeneous_nodes=True` |
| $p^*$ | `query_centroid_max_p` (default `DEFAULT_CENTROID_MAX_P = 0.9`) | local `max_p` in `resolve_query_fanin_scale` |
| $x(p^*)$ | (derived) | local `x` in `query_fanin_scale_from_centroid_p` |
| $\kappa$ | (derived) | local `stretch` in `query_fanin_scale_from_centroid_p` |
| $\kappa_1$ | (derived) | `x_sat` in `euler_sweep/search_space.py` |
| $c_{ij}$, $\ell_{ij}$ | - | `F.normalize(q_s)` then `log_alpha = einsum(q_s, key) * scale_s` |
| $\pi_{ij}$, $z_{ij}$ | - | `p_edge_on`, `z` in the gated attention forward |
| centroid init | `query_centroid_init` (requires `free_query_embedding`) | `_maybe_init_query_centroid` -> `model.init_query_at_key_centroid`, deferred to the first training batch |
| $K_{\text{init}}$, $K^{\star}$ | `query_capacity_init`, `max_in_edges` | NOT IMPLEMENTED - this document |
| $Q(\alpha)$ | (derived from `dag.degree`) | `er_indegree_quantile` (new), eq (14) |

`query_fanin_scale` is kept as the code symbol (no rename) so that every
existing config, checkpoint and sweep rule keeps working; it becomes a *derived*
quantity that users are not expected to set. Note the two name mismatches to
watch for when reading the code: the temperature $\tau$ is `self.beta`, and
`query_fanin_scale` is $F^{2}$, not $F$.

### Notation history

Earlier plain-text draft: the stretch constant was written `c` and is now
$\kappa$ (freeing `c` for the cosine $c_{ij}$), the prior in-degree was written
`q` and is now $K^{\star}$ (freeing $\mathbf{q}$ for the query vector), and the
edge posterior was written `P` and is now $\pi$ (freeing $\Pr$ for
probabilities).

Second pass (alignment with the reference notes): the Concrete temperature is
$\tau$ (was $\beta$), the Binary-Concrete random variable is $s_{ij}$ and its
stretched version $\bar{s}_{ij}$, and $z_{ij}$ is reserved for the clamped gate.
The gate block reads settings -> construction (2a)-(2c) -> thresholds on $s$
(2d) -> derived constants $\kappa,\kappa_1$ (2e) -> posterior (2f) ->
deterministic thresholds (2g), (2h). The score normalisation was factorised into
the single per-row divisor $Z_i=\lVert\mathbf{q}_i\rVert/(M_iF)$ of eq (1), and
the global constant became a **multiplier** $F$ (the old document used the divisor
$Z=1/F$ and wrote `query_fanin_scale` as $F$), so
$F_{\text{new}}=1/Z_{\text{old}}=\sqrt{F_{\text{old}}}$.

Third pass (this restructure): the capacity theorem and the budget identity were
merged into Lemma 1 of Section 2, whose role is now stated explicitly - it is
what extends the single-direction derivation of Section 1 to an arbitrary query
direction; the design formula is used only as $K\to F$ (3b); the penalty target
(3c) was moved up into the Implementation block; and the target posterior is
$p^*$ throughout. Presentation only; the code is unchanged.

Fourth pass: the "route (a) / route (b)" framing was dropped - there is one
design (move $\mu$, keep $F$), and shrinking $F$ survives only as the ablation arm
and as the startup guard of Section 3.1. What remained of the deleted sections was
folded into Section 3 (init gate, schedule clock, $\lambda_{\text{qn}}$), so eq
(8) and (11)-(12) keep their numbers; (12a) is new. The implementation plan is
Appendix C.

---

## Appendix B. Open questions

1. How should the penalty be normalised in $N$ (Section 3.3)? There are two
   distinct $N$-dependences and they need different fixes: `mean` instead of
   `sum` removes the growth of the total (reason 1) but not the $1/N$ in the
   per-edge price (reason 2), which is a ratio and therefore survives any global
   rescaling. Only the capacity form (12a) removes both. Proposal: add
   `query_norm_penalty: absolute | mean | capacity`, default `absolute`
   (today's behaviour), and report all three in the diagnostics so the choice is
   made on data rather than taste. Note (12a) does NOT harden the prior at
   $f=K^{\star}$ - both forms are flat there.
2. Is $\rho(t)$ linear enough? A cosine or exponential-in-$\mu$ squeeze may be
   gentler at the end. Proposal: implement linear in $K$, and keep the shape
   behind a single enum if the linear version deflates rows.
3. Should the squeeze act on $F$ itself instead (capacity $N\to K^{\star}$ in the
   forward pass)? It avoids the penalty entirely but rescales all rows globally
   and fights the already-learned $M_i$; not planned.
4. Is $\alpha=0.95$ the right convention for $Q(\alpha)$, or should the prior be
   the expected maximum in-degree ($\approx 14.6$ for ER-4 at $n=400$)? The
   oracle arm answers this empirically.

---

## Appendix C. Implementation plan

Two stages. Stage 1 is pure functions plus config resolution: no training loop is
touched, so every number in this document becomes a unit test. Stage 2 is the
schedule and the diagnostics. The invariant across both: with
`max_in_edges: null` and `query_capacity_init: auto` the behaviour must be
**bit-identical to today**.

### Stage 1 - resolution, helpers, guard (no training)

`causaliT/utils/query_norm.py`

1. `x_of_p(p, tau, gamma, zeta, edge_offset) -> float` - eq (3), i.e.
   $\mathrm{logit}(p)+T+\kappa$ with $\kappa$ from (2e). Extract it from the body
   of `query_fanin_scale_from_centroid_p`, which then calls it (no numerical
   change; the existing `test_query_fanin_auto.py` must still pass untouched).
2. `kappa(tau, gamma, zeta)` and `kappa_1(tau, gamma, zeta)` - eq (2e). The
   second one currently lives as `x_sat` in `euler_sweep/search_space.py`;
   re-point that call site so the constant has one definition.
3. `f_from_capacity(k, x) -> float` = $x\sqrt{k}$, eq (3b), and
   `capacity_from_f(f, x) -> float` = $(f/x)^2$, its inverse. Both in $F$ units;
   the caller squares for `query_fanin_scale`.
4. `mu_from_capacity(k_star, n_keys) -> float` = $\sqrt{K^{\star}/N}$, eq (3c).
5. `init_gate_at_centroid(k_init, n_keys, p, tau, gamma, zeta, edge_offset)`
   -> $(\ell_{\text{init}},\pi_{\text{init}},z_{\text{init}})$ from (8a) and
   (2a)-(2c) with $L=0$. Test against the Section 3.1 table.
6. extend `resolve_query_fanin_scale` with the 5-step order of Section 5. It must
   return `k_init`, `k_star`, `mu_end` and the three init-gate numbers so the
   caller can log them; and it must still return `None` when
   `query_fanin_scale` is an explicit float.
7. the guard: raise `ValueError` when $z_{\text{init}}=0$ and
   `query_centroid_init` is on, with $N$, $K_{\text{init}}$,
   $\ell_{\text{init}}$, $T$, $\kappa$ and the two remedies in the message.
8. `er_indegree_quantile(n_nodes, degree, alpha=0.95) -> int` - eq (14), pure
   Binomial, no scipy (`math.comb`). Test against the six rows of the Section 4
   table.

`causaliT/training/config_utils.py`

9. write `experiment.query_norm_target = mu_end` and
   `experiment.query_capacity_init` / `max_in_edges` back as resolved integers,
   then emit the Section 5 log line.

`causaliT/euler_sweep/euler_sweep/search_space.py`

10. `validate_dimensions` must stop recommending $F$ from $N$ when
    `query_capacity_init` is set; it currently would flag a deliberate capacity
    as a stale value and "fix" it back to $N$.

Tests (`tests/test_query_fanin_capacity.py`, new): round-trip
`capacity_from_f(f_from_capacity(k)) == k`; the Section 3.1 init-gate table; the
guard raising at $N=100,K_{\text{init}}=10$ and not raising at $K=N$; the
Section 4 quantile table; and `resolve_query_fanin_scale` unchanged on a legacy
config with an explicit float.

### Stage 2 - the schedule and the diagnostics

`causaliT/utils/query_norm.py`

11. `capacity_schedule(step_in_struct_epochs, anneal_epochs, idle_epochs, n_keys, k_star)`
    -> $(K(t),\mu(t))$ from (3d)-(3e), linear in $K$, clamped to $[K^{\star},N]$,
    returning $\mu=1$ while $t<$ `idle_epochs`.
12. optional `penalty_form` argument on `overspend_penalty` implementing (12a)
    behind the `query_norm_penalty` enum of Appendix B, question 1; default
    unchanged.

`causaliT/training/forecasters/attention_selector_forecaster.py` (and the
self-selector twin)

13. a structure-phase anchor, mirroring `_descendant_warmup_anchor`: record the
    global epoch at the first structure phase, count $t$ from it.
14. in `on_train_epoch_start`, write `module.query_norm_target = mu(t)` on every
    module owning a learnable `query_norm_log_scale`, de-duplicated by
    `id(parameter)` for `shared_query`. Idempotent, so a phase switch can simply
    re-run it.
15. log `query_norm/target_mu`, `query_norm/cap_target_edges`,
    `query_norm/cap_actual_edges` ($\mathrm{mean}_iM_i^2N$) next to the existing
    `mean_M` / `max_M`, plus the realised in-degree at $p^*$ so the mechanism
    prediction of Section 5 is measurable.

Tests: `mu(0)==1` and `mu(anneal_epochs)==sqrt(k_star/n)`; the target is written
once per shared parameter; a two-epoch smoke run with `max_in_edges: null`
reproduces the control loss exactly; the structural clock does not advance during
a reconstruct phase.

### Out of scope for both stages

The `oracle` arm (needs the realised graph plumbed through the sweep, and is
ground-truth leakage by construction), the dagsweep wiring of $Q(0.95)$ from
`dag.degree`, and the non-linear $\rho$ shapes of Appendix B question 2.
