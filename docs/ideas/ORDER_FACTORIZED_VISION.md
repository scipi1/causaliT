# Order-Factorized Causal Attention

Status: design proposal, not an implemented or validated method.
Companions: `docs/MAIN.md` (P1/P2), `docs/PAPER_TEMPLATE.md`, and
`docs/documentation/ATTENTION_TEMPERATURES.md`. 

## 1. Current model and its limitations

The current attention gates use Hard-Concrete sampling [Louizos et al.,
2018]. A gate is a continuous value in $[0,1]$, with positive probability
of being exactly zero or one. It is not itself a Bernoulli variable.
Its inclusion indicator is Bernoulli: an edge is included when its gate
is greater than zero.

Write $d$ for the number of nodes, $X$ for the observed data, $Z$ for the
collection of gates, and $q$ for their learned sampling distribution.
A sigmoid gives a separate inclusion probability for each edge. A softmax
instead gives normalized row weights; those weights alone are neither
independent inclusion probabilities nor a sampled parent set.

Independent gate sampling defines a joint distribution: the product of
the gate distributions. The limitation is not the absence of a joint,
but the dependencies it cannot represent. Shared parameters and shared
training losses do not, by themselves, couple independently sampled gates.
Pairwise sharing of existence gates or direction noise can create local
dependencies; that still does not impose a shared global order.

For example, two edges can each have probability 0.5 while being
independent, always appearing together, or never appearing together.
Their marginals alone do not distinguish these possibilities. In an
independent model, conditioning one edge on the other leaves its
probability unchanged.

The current NOTEARS-style penalty discourages cycles. For weighted
adjacency $W$, one standard form is

$$
h(W)=\operatorname{tr}\bigl(\exp(W\circ W)\bigr)-d,
$$

where $\circ$ is elementwise multiplication, $\exp$ is the matrix
exponential, and $\operatorname{tr}$ is the sum of diagonal entries.
This is an optimization constraint, not a rule for sampling graphs.
With finite penalties, cycles can remain. Independent sampling can
assign positive probability to cycles whenever the edge support permits them.

NOTEARS does backpropagate into edge weights. Its penalty supplies
cycle-breaking pressure, not evidence about the causal direction.
The data-fitting or independence loss also affects which edges survive.
It is therefore misleading to say that NOTEARS chooses an order without
data. The actual distinction is that the current model has no explicit
ordering variable to train, inspect, or sample.

The proposal adds that variable. It does not automatically add causal
information: that information must come from the structural training
objective and its assumptions.

## 2. Proposed model: gates conditional on an order

### Conventions and parameters

All mathematical adjacency entries in this document use the convention
$Z_{ij}$ means edge $i\to j$. Implementation must transpose masks if
attention rows are targets and columns are sources.

Let $\sigma(i)$ be the position of node $i$ in a permutation of the nodes.
Position 1 comes first. An edge $i\to j$ is allowed only if
$\sigma(i)<\sigma(j)$. This convention applies to gates and exclusion masks.

Let $r\in\mathbb{R}^d$ be learnable node scores. Larger scores favor earlier
positions. They parameterize an order sampler; they are not a sampled
order. A sampled permutation is $\sigma$, and its relaxed counterpart
is a matrix $P$. Sinkhorn normalization has no learnable parameters of
its own: it transforms learned scores into a relaxed assignment.

Node scores are one possible parameterization, not a requirement.
A free node-by-position score matrix can represent richer uncertainty.
The choice remains open; the pipeline below uses node scores as a baseline.

Scope: self-attention only (homogeneous nodes, square $d\times d$
adjacency). The design is a NEW standalone attention module; it does not
modify `GatedSelfAttention` and does not inherit its direction-side
features (direction gate, direction bias, etc.).

The current model has one attention layer and one head. Place one order
sampler there. Queries and keys are computed from global variable
descriptors, so the structure is global: it is identical for every
observation and every batch. Consequently all structural noise (Gumbel
order noise and gate noise) is drawn ONCE per gradient step, never per
observation, and shared by the whole minibatch and by every forward pass
inside that step. If multiple layers or heads are introduced later,
revisit order sharing rather than silently assigning separate causal
orders to them.

### Hard order and conditional gates

Define the order mask

$$
M^\sigma_{ij}=\mathbb{1}[\sigma(i)<\sigma(j)],
$$

where $\mathbb{1}$ is the indicator, equal to one when its condition holds.
Sample gates only on allowed edges, and set all other gates to zero.
The proposed joint distribution is

$$
q(\sigma,Z)=q(\sigma\mid r)\prod_{i\ne j}q(z_{ij}\mid\sigma).
$$

Existence is symmetric ($\pi_{ij}=\pi_{ji}$), so the baseline draws ONE
existence sample per unordered pair and uses it for both directions
(as the current `GatedSelfAttention` already does). The product above is
therefore over unordered pairs; the order alone decides which of the two
directions is allowed.

Every graph sampled with the hard mask is a directed acyclic graph (DAG):
following an edge always increases position, so a path cannot return to
its starting node. Self-edges are forbidden.

Conditioning on an order is not the same as conditioning on the rest of
the graph. Nevertheless, after averaging over uncertain orders, edges
can become dependent because they share the same sampled order.
For example, $1\to2$ and $2\to3$ constrain the possible position of node 3
relative to node 1. The model cannot simultaneously include $3\to1$.

An existence gate cannot override a hard forbidden edge.
Instead, the structural loss can change the order distribution so that
another sampled order permits that edge.

## 3. Training with relaxed orders and gates

### Gumbel-Sinkhorn order sampling

Let $\Theta\in\mathbb{R}^{d\times d}$ be node-by-position scores.
A simple construction from $r$ is

$$
\Theta_{ik}=-(r_i-\mu_k)^2,
\qquad
\mu_k=\frac{d+1}{2}-k.
$$

The fixed levels $\mu_k$ decrease with position, so larger $r_i$ favor
earlier positions. They are centered at zero. Centering $r$ removes an
irrelevant common offset. The spacing of these levels sets the score scale.

For fixed $r$, row terms $-r_i^2$ and column terms $-\mu_k^2$ cancel in
assignment comparisons and ideal Sinkhorn normalization. The relevant
interaction is $2r_i\mu_k$. Thus the levels are not independent bins
that can all attract nodes to one position: assignment normalization
forces positions to compete.

Let $E$ be a matrix of independent standard Gumbel noise, $s$ its noise
scale, and $\tau_{\mathrm{perm}}>0$ the relaxation temperature. Define

$$
P=\operatorname{Sinkhorn}\left(
\exp\left(\frac{\Theta+sE}{\tau_{\mathrm{perm}}}\right)\right).
$$

The exponential here is elementwise. Sinkhorn alternately normalizes rows
and columns. With sufficient iterations, $P$ is approximately doubly
stochastic: its entries are nonnegative and each row and column sums to one.
Use a log-domain implementation to avoid overflow.

$P$ is a relaxed sample, not automatically a matrix of exact posterior
position probabilities. At evaluation, maximum-weight assignment
(Hungarian matching) gives a hard permutation. Sampling fresh noise and
rounding defines an implicit distribution over permutations.

Reducing temperature sharpens the relaxation. It does not remove Gumbel
noise or necessarily make the distribution of hard orders concentrate.
Noise scale and score gaps also control ordering uncertainty.

### A direct relaxed order mask

Define a fixed position mask $T\in\mathbb{R}^{d\times d}$ by

$$
T_{k\ell}=\mathbb{1}[k<\ell],
\qquad M=PTP^\top.
$$

For a hard permutation matrix, $M$ is exactly the allowed-edge mask.
For a relaxed matrix, it is a differentiable extension of that mask,
not an exact pairwise precedence probability. Explicitly set its diagonal
to zero. Soft masks can contain cycles; only hard permutation masks give
the DAG guarantee.

Symmetry at initialization: for doubly stochastic $P$,
$M_{ij}+M_{ji}=1-\sum_k P_{ik}P_{jk}$. Near a uniform $P$ this gives
$M_{ij}\approx M_{ji}\approx (d-1)/(2d)\approx 0.5$: every pair is
half-open in both directions (a soft two-cycle). Orientation emerges only
as the structural gradient breaks this symmetry; monitor it.

At finite temperature, Sinkhorn entries are generally positive, so most
off-diagonal mask entries can also be positive. A small mask value suppresses
an edge without making it exactly absent. Relaxed training support can
therefore be denser than hard evaluation support. Measure this gap in
reconstruction, independence loss, and graph extraction. A hard-forward,
soft-backward mask is a later comparison, not the initial baseline.

An alternative uses expected positions $m_i=\sum_{k=1}^d kP_{ik}$ and a
sigmoid of $m_j-m_i$. That is another chosen smooth mask, not a derived
Sinkhorn probability formula. The direct matrix construction is the
baseline proposed here; compare the alternatives experimentally.

### Conditional adjacency

Let $Q$ and $K$ be the query and key representations. After adapting the
attention indexing to the source-to-target convention, let $R$ denote the
raw score matrix. Use its symmetric part for existence:

$$
S=\frac{R+R^\top}{2},\qquad
\pi_{ij}=\operatorname{sigmoid}(S_{ij}-b).
$$

Here $b$ combines the existing offsets in the Hard-Concrete inclusion
readout, and $\pi_{ij}$ is the probability that an unmasked gate is positive.
Let $G$ be a sampled Hard-Concrete gate matrix. Training uses

$$
Z_{\mathrm{relaxed}}=G\odot M,
$$

where $\odot$ is elementwise multiplication and $M$ is the sampled relaxed
order mask. Its probability-weighted readout is $H_{ij}=\pi_{ij}M_{ij}$;
this is not the expected continuous gate value or an exact hard-graph
inclusion probability at finite relaxation temperature.

The antisymmetric QK component is omitted for now as non-essential.
The learned order supplies orientation, and the structural independence
objective trains that order. Pairwise direction preferences can be
revisited if experiments reveal a specific need.

The structural loss backpropagates through $M$ and $P$ into the order
parameters, as well as into the existence-gate parameters.

### Two forward methods and adaptive-training phases

The adaptive trainer alternates phases: structure phases train only the
structural parameters (regressor frozen), reconstruction phases train only
the regressor (structure frozen). The module therefore exposes two forward
paths, selected by a phase hook (e.g. `set_order_forward("relaxed"|"hard")`
called by `PhaseController`, like the existing `set_phase_active` hooks):

- `forward_relaxed` (structure phases): Gumbel-Sinkhorn $P$, soft $M$,
  Hard-Concrete $G$, $Z_{\mathrm{relaxed}}=G\odot M$. Gradients reach $r$
  and the existence parameters.
- `forward_hard` (reconstruction phases and Monte Carlo graph sampling):
  at every step, draw ONE hard DAG from the current learned distribution,
  $Z_{\mathrm{binary}}=B\odot M^\sigma$ (below). The regressor is thus
  trained on a fresh binary DAG per step, the order-based analogue of the
  current "dense adjacency with learned dropout" reconstruction regime.

Cross-fitting only splits the data between the two phases and does not
change this design. Because noise is global per step, any extra forward
within one step reuses the same sampled structure.

### Binary graph sampling

For each Monte Carlo sample (and each reconstruction step), draw fresh standard Gumbel noise $E$ and
solve maximum-weight assignment on $\Theta+sE$. This produces a hard
permutation matrix $P_{\mathrm{hard}}$. Define

$$
M^\sigma=P_{\mathrm{hard}}TP_{\mathrm{hard}}^\top,
\qquad
B_{ij}\sim\operatorname{Bernoulli}(\pi_{ij}),
\qquad
Z_{\mathrm{binary}}=B\odot M^\sigma.
$$

$B$ is the binary existence matrix, sampled once per unordered pair and
mirrored ($B_{ij}=B_{ji}$). Every resulting graph is a DAG.
Bernoulli sampling matches the inclusion indicators of Hard-Concrete,
not its continuous gate magnitudes. The hard assignment uses
`scipy.optimize.linear_sum_assignment`.
No direction filter or direction temperature is needed.

Repeated samples estimate edge frequencies, relative-order frequencies,
and graph-dependent predictions. They describe the learned sampler's
variability, not automatically a calibrated Bayesian posterior.

This explicit graph-sampling operation is separate from ordinary prediction
evaluation. The latter needs a declared deterministic or sampled policy;
that policy remains open. Do not implicitly turn every evaluation forward
pass into a Monte Carlo graph draw.

### Pipeline pseudocode

This is design pseudocode, not runnable Python. Parameters are initialized
once, outside the training loop. All matrices use source-to-target indexing.

```text
# Inputs: d nodes; Q and K representations; gate offsets and temperatures.
# transpose adapts attention indexing to source-to-target indexing if needed.
# init_r is a chosen flat, random, or prior-defined vector of length d.
r = learnable_parameter(init_r)
mu = [(d + 1) / 2 - k for k in 1..d]  # fixed decreasing position levels
T = [[int(k < l) for l in 1..d] for k in 1..d]

# forward_relaxed (structure phases): one relaxed order per gradient step,
# shared by the whole minibatch and every forward in the step.
# noise_scale controls perturbations; tau_perm controls relaxation sharpness.
r_centered = r - mean(r)
Theta[i, k] = -(r_centered[i] - mu[k])**2
E = sample_standard_gumbel(shape=(d, d))
P = log_sinkhorn((Theta + noise_scale * E) / tau_perm)
# log_sinkhorn takes log-scores and returns the normalized matrix P.
M = P @ T @ transpose(P)
M = zero_diagonal(M)

# Shared score geometry and conditional gate samples.
# raw_scores performs the existing QK scaling and index adaptation.
R = raw_scores(Q, K)
S = (R + transpose(R)) / 2
# hc_logit_offset combines existing offsets for gate sampling.
# hc_beta, hc_gamma, hc_zeta are the existing Hard-Concrete parameters.
gate_logits = S - hc_logit_offset
G = sample_hard_concrete(gate_logits, hc_beta, hc_gamma, hc_zeta)
Z_relaxed = zero_diagonal(G * M)      # elementwise product, conditional on P

# Edge-based descendant exclusion (Section 4), same step's samples.
# Hard order from the SAME noise E; existence evidence from the same gates.
P_step = maximum_weight_assignment(Theta + noise_scale * E)
A_hard = (G > threshold) * (P_step @ T @ transpose(P_step))   # DAG
# descendant_mask.py uses child <- parent layout, hence the transpose.
exclusion_mask = build_hsic_pair_mask(transpose(A_hard), s_seq_len=0,
                                      homogeneous_nodes=True)  # detached
# Regressor parameters receive reconstruction gradients.
# Structural parameters, including r, receive independence gradients.
# Routed losses and their weights are defined in Section 4.
# No order-uncertainty regularizer is used by default.

# forward_hard (reconstruction steps, Monte Carlo sampling): one hard DAG
# per step, no gradient to structural parameters.
# Theta is the learned node-by-position score matrix defined above.
E_eval = sample_standard_gumbel(shape=(d, d))
P_hard = maximum_weight_assignment(Theta + noise_scale * E_eval)
M_hard = P_hard @ T @ transpose(P_hard)
# hard_concrete_inclusion_prob includes the stretch/temperature correction.
pi = hard_concrete_inclusion_prob(gate_logits, hc_beta, hc_gamma, hc_zeta)
B = mirror_upper(sample_bernoulli(pi))  # one draw per unordered pair
Z_binary = zero_diagonal(B * M_hard)
```

Repeat the evaluation block with fresh order and gate samples to estimate
variability over orders and graphs. Do not reuse training samples as the
Monte Carlo evaluation sample set.

## 4. Training signals, order variability, and exclusion masks

### Routed reconstruction and independence objectives

Write $\psi$ for downstream regressor parameters and $\phi$ for structural
parameters, including order and gate parameters. The existing
`gradient_routing` machinery separates their updates. A schematic choice is

$$
L_\psi=L_{\mathrm{reconstruction}},
\qquad
L_\phi=\lambda_I L_{\mathrm{independence}}
+\lambda_0 L_{\mathrm{sparsity}}.
$$

The nonnegative $\lambda$ values are loss weights. The regressor fits the
current structure; the structure uses residual independence, measured by
HSIC (Hilbert-Schmidt independence criterion), plus sparsity.
For the initial baseline, independence updates all structural parameters,
including existence parameters and $r$; sparsity updates the existence
parameters. Reconstruction updates only regressor parameters, with no
structural reconstruction mixing. The unmasked proposal penalty has no
direct gradient into $r$. Keep the differentiable path through the
regressor to the conditional gates when computing structural gradients.
Routing blocks unwanted parameter updates, not this required gradient path.

An additive noise model (ANM) assumes each variable is a function of its
parents plus noise independent of those parents. Under suitable model
assumptions, independence can distinguish directions that observational
conditional-independence tests cannot distinguish. This is not universal:
linear-Gaussian models and other exceptions remain ambiguous. Regression
quality, finite data, confounding, and loss masking also affect recovery.
HSIC-trained structure therefore has causal motivation, not a guarantee.

### What distribution do we learn?

The order and gate parameters define a learned sampling distribution
$q_\phi(\sigma,Z)$. Its form specifies how orders and conditional gates
are sampled; it is not a likelihood for the observed data. The method
trains this distribution through independence, fitting, and sparsity
objectives. It does not require a data likelihood or a Bayesian posterior.

Reconstruction loss can sometimes be interpreted as a negative
log-likelihood under an explicit noise model. That does not make the
routed independence-based training a likelihood method. We make no
ELBO or calibrated-posterior claim here.

The initial sparsity term retains the current unmasked existence-proposal
penalty. It measures expected proposal count, not expected final masked
edge count and not automatically a KL divergence. Preserve its existing
normalization initially. A masked-edge penalty is a later comparison.

### Order variability: noise, scores, and temperature

Fresh Gumbel noise can produce different orders. For example, node 1 may
consistently come first while nodes 2 and 3 exchange positions. These
samples permit different edges and propagate order variability downstream.

Score gaps relative to the noise scale determine how much samples vary.
The baseline fixes the noise scale and learns the scores through the
structural objective. Temperature controls how close a relaxed sample is
to a permutation; it does not by itself control agreement between samples.

No uncertainty-specific regularizer is required by default. We report
variation across sampled orders, not Bayesian posterior uncertainty.
Whether this variation reflects ambiguity in the data must be evaluated.
Add scale control or diversity regularization only if a concrete training
failure motivates it; neither automatically calibrates uncertainty.

### Edge-based descendant exclusion

The HSIC loss is a sum of nodal tests, RESIT-style [Peters et al., 2014]:
the residual $r_i$ of node $i$ is tested against the VALUES of candidate
non-descendants. Under an ANM, $r_i=e_i$ is independent of every
non-descendant but necessarily dependent on descendants and on $X_i$
itself, so those pairs must be excluded.

**Rejected alternative: order-only exclusion.** Excluding every node later
in the sampled order needs no closure, but it gives the first node an
empty test set and a free zero loss regardless of the data, and it drops
informative tests between unrelated nodes (e.g. two sources, which must
be placed one after the other).

**Baseline.** Build the exclusion from actual edges, using the existing
`causaliT/utils/descendant_mask.py::build_hsic_pair_mask` (threshold
variant only; the budgeted variant is not used):

1. Hard order $P_{\mathrm{step}}$ from the SAME Gumbel noise $E$ as the
   relaxed $P$ of the step, so gates and mask refer to one sampled order.
2. Existence evidence from the same step's gates (thresholded).
3. $A=\mathrm{existence}\odot M^\sigma$ is a DAG; transpose it to the
   child $\leftarrow$ parent layout of `descendant_mask.py`, compute the
   full transitive closure, and exclude $\mathrm{Desc}(i)\cup\{i\}$.
   `is_cyclic` must always be false (free orientation check).

A node is excluded from a test only if it is reached through active
edges. A source whose every other node is a descendant gets an empty
row, i.e. complete exclusion, which is correct for a parentless node and
was observed to work well in the current model. Existing safeguards
(warm-up anchor, minimum kept fraction, first-order degrade, EMA) are kept
and set through configuration.

**Dense start.** The closure of a complete DAG equals the order-only mask.
Adaptive training starts with a reconstruction phase, so this can only
occur at the beginning of the first structure phase, when the order is
still near-flat. The descendant mask requires edge evidence (threshold
combined with the order mask) and is activated by the existing guardrail
once a proper order is available.

**Detachment: the mask is a consequence, not a lever.** Minimizing
residual dependence changes the selection set (gates); the selection set
moves the order through the gate path; edges and order then determine the
descendant mask. Detaching (default) preserves this direction of
causation and blocks creating edges only to delete penalty terms. An
attached variant is an optional ablation; it is coherent only because
sparsity acts on existence gates alone. Monitor active tests per node and
nodes collapsing to the front of the order.

### Worked example: V-structure $s_1\to c\leftarrow s_2$

With $s_1\perp s_2$ and the true graph, in either relative order of the
sources: $\mathrm{Desc}(s_1)=\mathrm{Desc}(s_2)=\{c\}$,
$\mathrm{Desc}(c)=\emptyset$. Tests: $\mathrm{HSIC}(s_2,s_1)=0$,
$\mathrm{HSIC}(s_1,s_2)=0$, $\mathrm{HSIC}(r_c,\{s_1,s_2\})=0$ under ANM.

- **Tied sources.** The loss is identical for both source orders, so their
  relative order is not identifiable and should remain variable.
  Relative-order frequencies are reported only for pairs connected by a
  directed path in the sampled graphs.
- **No Berkson effect.** Source tests exclude $c$ and no test conditions on
  $c$, provided both edges into $c$ are active.
- **Chain $s_1\to c\to s_2$.** $r_{s_2}=s_2-h(c)$ is dependent on $s_1$; the
  test fails (also linear-Gaussian: V-structures are identifiable).
- **Missing $s_2\to c$.** $r_c$ retains the effect of $s_2$ and fails its test
  against $s_2$, pushing the gate on.
- **$c$ placed first.** $r_c=c$ fails against $s_1,s_2$; the gradient through
  the soft $M_{s_1c}$ moves $s_1$ before $c$.
- **Redundant $s_1\to s_2$.** Loss unchanged; no reward from the detached
  mask; sparsity removes it.

Risks: (i) competing reverse-direction gradients until $s_1\to c$ is
active (RESIT-like; resolved by the ANM asymmetry, may need warm-up);
(ii) non-additive colliders ($c=s_1s_2+n$) require sufficient regressor
capacity; (iii) hard exclusion with soft gates is a mismatch, acceptable
because the mask is detached.

## 5. Comparison and contribution positioning

### Comparison with the current architecture

| Component | Current | Proposed baseline |
|---|---|---|
| Ordering | Implicit in learned edges | Explicit sampled order; Sinkhorn during training |
| Gate dependencies | Independent or pairwise-shared sampling | Shared order couples gates after averaging over orders |
| Acyclicity | Finite optimization penalty | Guaranteed for hard order masks, not relaxed masks |
| Structural signal | Existing routed losses, including HSIC where enabled | Same available signals also train order parameters |
| Exclusion masks | Derived from adjacency-based estimates (may be cyclic) | Detached closure of hard (existence x order) DAG; never cyclic |
| Reconstruction phases | Dense adjacency with learned gate dropout | One hard DAG sampled per step from the learned distribution |
| Sparsity | Hard-Concrete inclusion penalty | Retain unmasked proposal penalty; masked-edge accounting is a later comparison |
| Objective interpretation | Depends on actual losses and routing | Routed independence/fitting; no automatic ELBO claim |

Related findings to revisit experimentally: F10 (undecided query alignment),
F3/P1/P2 (optimization instability and ordering), F11 (dense proposals),
and OQ1 (independence signal does not imply structure recovery).

### Contribution hypotheses and related work

Order-based learning [Friedman & Koller, 2003; Koivisto & Sood, 2004;
Teyssier & Koller, 2005] motivates separating order uncertainty from
parent-set selection. Classical methods generally factor by parent sets
or nodes, not necessarily independent individual edges. Order priors
can weight graphs differently because a graph may admit many orders
[Koivisto, 2006; Ellis & Wong, 2008].

Gumbel-Sinkhorn [Mena et al., 2018] supplies differentiable assignment
sampling. 
NOTEARS [Zheng et al., 2018] is the penalty-based comparator. Hard-Concrete
[Louizos et al., 2018] supplies gate sampling and inclusion readouts.

Compare the design with DiBS, BCD-Nets, DAG-GFlowNet, and differentiable
DAG sampling [Lorch et al., 2021; Cundy et al., 2021; Deleu et al., 2022;
Charpentier et al., 2022]. These use different graph distributions,
mechanisms, and inference approximations. Avoid claiming that penalties
cannot define distributions, or that all prior ordering scores are
unconnected to edge models. Reisach et al. motivate tests for data-scale
and optimization artifacts, not a proof that this design avoids them.

### Design decisions grounded in existing work

| Decision | Grounding | Difference in this design |
|---|---|---|
| Order x edges given the order | BCD-Nets [Cundy et al., 2021]; VI-DP-DAG [Charpentier et al., 2022]; permutahedron [Zantedeschi et al., 2023] | Edges from attention QK geometry; structure trained by routed residual HSIC, not likelihood/ELBO |
| Symmetric existence + separate orientation | ENCO [Lippe et al., 2022] (edge-existence and orientation parameters) | Orientation from a global order, so hard samples are acyclic by construction |
| Order then parent selection | CAM [Buhlmann et al., 2014]; SCORE [Rolland et al., 2022]; NoGAM/DAS [Montagna et al., 2023]; CORL [Wang et al., 2021] | Order and selection learned jointly and differentiably, not in two stages |
| Residual independence as structural signal | RESIT [Peters et al., 2014]; regression by dependence minimization [Mooij et al., 2009] | RESIT finds the order greedily; here the order follows from minimizing residual dependence through the selection set |
| Relaxed permutation sampling | Gumbel-Sinkhorn [Mena et al., 2018]; Gumbel-sort / Plackett-Luce [Grover et al., 2019; Charpentier et al., 2022] | Compared by ablation |
| Descendant exclusion of tests | RESIT sink/source logic [Peters et al., 2014] | Exclusion from closure of sampled hard DAG, detached |

Order factorization itself is NOT a contribution: BCD-Nets and VI-DP-DAG
already combine a sampled permutation with order-masked edges. Any novelty
lies in the combination below.

Potential contributions to validate:

1. Attention gates conditioned on sampled orders, using shared QK geometry
   over global variable descriptors and a routed residual-HSIC signal.
2. A consistent order-to-gate pipeline with edge-based, detached
   descendant exclusion, and phase-separated training (relaxed structure
   phases, sampled hard DAGs for reconstruction phases).
3. A comparison of learned orders and symmetric existence gates. Symmetric
   existence is not automatically a recovered skeleton or moral graph;
   these are different graph objects.
4. Propagating graph uncertainty into average treatment effect estimates.
   Sampling a learned graph distribution does not by itself guarantee
   calibrated posterior or effect intervals.

Claims of priority or improved recovery require a literature review and
experiments. Adjacent comparisons include DCDI, AVICI, and DECI.

## 6. Open questions and current decisions

- **Training-time order sampling.**
  Q: Does Sinkhorn run during training, or only at evaluation?
  A: During training. The sampled relaxed order conditions the gates;
  structural gradients pass through it. Hard orders are used for DAG reports.

- **Order parameterization.**
  Q: Node scores $r$, a free node-by-position matrix, or a data-dependent
  predictor? How much uncertainty can each represent?
  A: Open. Node scores are the baseline in this document. A free matrix
  is a meaningful alternative; amortization is relevant across datasets.

- **Order-to-mask construction.**
  Q: Use $PTP^\top$, expected-position sigmoids, or hard-forward/soft-backward
  masks?
  A: Open experimentally. $PTP^\top$ is the proposed direct baseline;
  only a hard permutation mask guarantees DAG support.

- **Gate semantics.**
  Q: Is a separate asymmetric direction gate needed? How are binary gates
  sampled at evaluation?
  A: No direction gate in the baseline. Sample Bernoulli existence with
  learned Hard-Concrete inclusion probabilities, then apply the sampled
  hard order mask. One existence draw per unordered pair (symmetric).

- **Phase-dependent forward.**
  Q: Which graph does each adaptive-training phase use?
  A: Structure phases: relaxed forward only. Reconstruction phases: one
  hard DAG sampled per step from the learned distribution. Selected by a
  phase hook in `PhaseController`.

- **Noise granularity.**
  Q: Per observation or per step?
  A: Per gradient step, shared by the batch: the structure is global.
- **Structural training signal.**
  Q: Must structure train on reconstruction? Does existence also use HSIC?
  A: The baseline routes independence to existence parameters and $r$,
  sparsity to existence parameters, and reconstruction only to the regressor.
  No structural reconstruction mixing initially; it is a later ablation.

- **Order variability.**
  Q: Does fixed sampling noise preserve useful alternative orders, or cause
  premature commitment or unnecessary variability? Do score magnitudes grow?
  A: Open. Fix noise scale initially and use no order-specific regularizer.
  Evaluate first; add regularization only for an observed failure.

- **Temperature and noise schedules.**
  Q: How should permutation temperature, noise scale, and gate temperatures
  interact? How many Sinkhorn iterations are needed?
  A: Annealing is accepted; the schedule is open. Sharpness and ordering
  uncertainty are distinct. Track row/column normalization errors.

- **Initialization and stability.**
  Q: Flat, random, prior-defined, or field-derived initialization? Is the
  order-to-mask-to-loss gradient stable?
  A: Open. Check gradient magnitudes, score gaps, and sensitivity to seeds.

- **Descendant exclusion.**
  Q: Order-only or edge-based? Detached or attached?
  A: Edge-based closure of the step's hard DAG via `build_hsic_pair_mask`
  (threshold variant; budgeted variant not used); complete exclusion of
  sources allowed. Detached by default, attached as ablation. Safeguards
  and activation guardrail are configured, not redesigned.

- **Sparsity accounting.**
  Q: Penalize unmasked gate proposals or expected final masked edges?
  A: Retain the current unmasked existence-proposal penalty for the baseline.
  A final masked-edge penalty remains an experimental comparison.

- **Evaluation and relaxation bias.**
  Q: How many Monte Carlo samples, what rounding/threshold rules, and how
  large is the relaxed-to-hard performance gap?
  A: Monte Carlo graph sampling uses fresh perturbed-score hard assignments
  and Bernoulli existence indicators. Sample counts and ordinary prediction
  policy remain open. Measure graph cycles, mask convergence, structure quality,
  and uncertainty calibration separately. Annealing is not a general
  guarantee that every approximation error vanishes.

- **Interventions.**
  Q: How should treatment information train order and mechanisms?
  A: Deferred. The current design focuses on observational data.

## 7. Concrete baseline implementation plan

This plan fixes the initial architecture, not the numerical settings or
recovery claims. Before coding, inspect the existing attention, configuration,
gradient-routing, HSIC masking, and evaluation interfaces to locate the
integration points and preserve existing conventions.

### Phase 1: Order sampler

- Create a new standalone module
  `causaliT/core/modules/order_factorized_attention.py` (self-attention,
  homogeneous nodes). Register learnable node scores $r$ and fixed
  position levels and mask $T$.
- Implement centered scores, Gumbel perturbations, log-domain Sinkhorn,
  the relaxed mask, and hard assignment on perturbed scores.
- Expose initialization, fixed noise scale, temperature schedule, and
  normalization iteration count through configuration. Numerical defaults
  must be declared and checked in the small baseline experiment.
- Draw order and gate noise once per gradient step, shared by the batch
  and cached for every forward in the step. Track row/column normalization
  errors and score magnitudes.
- Expose `forward_relaxed`, `forward_hard`, a phase switch
  `set_order_forward("relaxed"|"hard")`, and `sample_graphs(n, seed)`.

### Phase 2: Attention integration

- Add an opt-in order-factorized mode. Keep the current mode unchanged
  for backward compatibility and comparison.
- Retain symmetric Hard-Concrete existence gates. Omit the asymmetric
  direction factor in the new mode and use $G\odot M$ instead.
- Adapt source/target indexing explicitly and forbid self-edges.
- Reject `kappa > 0` and `acyclicity_constraint.enabled` in the new mode.
  Do not claim that relaxed training support is acyclic; hard samples are.
- Add a `PhaseController` hook: relaxed forward in structure phases, hard
  sampled DAG per step in reconstruction phases.

### Phase 3: Gradient routing and exclusion

- Classify $r$ as structural. Route independence gradients to order and
  existence parameters; retain the unmasked proposal sparsity penalty.
- Route reconstruction only to regressor parameters. Verify that the
  structural loss can differentiate through the regressor into the gates.
- Feed the step's hard (thresholded existence x order) DAG, transposed to
  child <- parent layout, into `_build_hsic_descendant_mask` /
  `build_hsic_pair_mask`. Optional attached variant for the ablation. Do
  not detach the mask used in conditional adjacency.
- Preserve existing sparsity normalization and test HSIC index conventions,
  retained comparison counts, missing-parent cases, and empty-graph cases.

### Phase 4: Evaluation interfaces

- Separate ordinary prediction evaluation from an explicit Monte Carlo
  graph-sampling operation. Declare the ordinary prediction policy before
  implementation; do not silently make existing evaluation stochastic.
- For each graph sample, draw fresh order noise, solve hard assignment on
  perturbed scores, sample Bernoulli existence indicators, and apply the
  hard order mask. No threshold is required for these binary graph samples.
- Return sampled graphs and summarize edge and relative-order frequencies.
  Allow reproducible seeds and configurable sample counts.
- If predictions use sampled binary graphs, compare them with relaxed
  training predictions rather than assuming the two regimes are equivalent.

### Phase 5: Validation and small baseline experiment

- Add short unit tests for indexing, mask orientation, zero diagonals,
  Sinkhorn normalization, and acyclicity of hard graph samples.
- Test gradients into $r$ and existence parameters, regressor/structure
  routing, and exclusion detachment without breaking the gate gradient path.
- Test seeded sampling, fresh Monte Carlo draws, and unchanged behavior
  when the new mode is disabled. Include configuration/checkpoint handling.
- Run a small synthetic experiment before recovery benchmarks. Track
  independence and reconstruction losses, proposal sparsity, order
  variability, numerical stability, and relaxed-to-hard performance gaps.
- Add a 3-node collider test ($s_1\to c\leftarrow s_2$): source tie and
  exclusion behaviour; assert `is_cyclic == False` for hard samples.
- Treat temperature schedules, exclusion safeguards, and sampler settings
  as experimental questions. Request approval before time-consuming tests.

### Ablations

| Ablation | Baseline | Alternative |
|---|---|---|
| Exclusion gradient | Detached | Attached (soft closure) |
| Order sampler | Gumbel-Sinkhorn on node-by-position scores | Plackett-Luce (Gumbel-perturbed $r$, sort) |
| Relaxed mask | $PTP^\top$ | Straight-through hard mask; expected-position sigmoid |
| HSIC aggregation | Existing nodal sum/mean | Normalized by active tests per node |
| Exclusion robustness | No edge dropout | `edge_dropout` in `build_hsic_pair_mask` |
| Sparsity | Unmasked proposal penalty | Masked-edge penalty |

## 8. References

- [Louizos et al., 2018] - Louizos, Welling & Kingma, "Learning Sparse
  Neural Networks through L0 Regularization", ICLR 2018 (Hard-Concrete).
- [Friedman & Koller, 2003] - "Being Bayesian About Network Structure",
  Machine Learning Journal, 2003.
- [Koivisto & Sood, 2004] - "Exact Bayesian Structure Discovery in
  Bayesian Networks", JMLR 2004.
- [Teyssier & Koller, 2005] - "Ordering-Based Search: A Simple and
  Effective Algorithm for Learning Bayesian Networks", UAI 2005.
- [Koivisto, 2006] - "Advances in Exact Bayesian Structure Discovery in
  Bayesian Networks" (partial order MCMC), UAI 2006.
- [Ellis & Wong, 2008] - "Learning Causal Bayesian Network Structures from
  Experimental Data", JASA 2008.
- [Mena et al., 2018] - "Learning Latent Permutations with Gumbel-Sinkhorn
  Networks", ICLR 2018.
- [Zheng et al., 2018] - "DAGs with NO TEARS: Continuous Optimization for
  Structure Learning", NeurIPS 2018.
- [Reisach et al., 2021, 2023] - "Beware of the Simulated DAG!" (2021) and
  "A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive
  Noise Models" (2023).
- [Lorch et al., 2021] - "DiBS: Differentiable Bayesian Structure
  Learning", NeurIPS 2021.
- [Cundy et al., 2021] - Cundy, Grover & Ermon, "BCD-Nets: Scalable
  Variational Approaches for Bayesian Causal Discovery", NeurIPS 2021.
- [Deleu et al., 2022] - "Bayesian Structure Learning with Generative Flow
  Networks" (DAG-GFlowNet), AISTATS 2022; and "Bayesian Learning of Causal
  Structure and Mechanisms with GFlowNets and Variational Bayes",
  arXiv:2211.02763, 2022.
- [Charpentier et al., 2022] - Charpentier, Kibler & Gunnemann,
  "Differentiable DAG Sampling" (VI-DP-DAG), ICLR 2022.
- [Peters et al., 2014] - Peters, Mooij, Janzing & Scholkopf, "Causal
  Discovery with Continuous Additive Noise Models" (RESIT), JMLR 2014.
- [Mooij et al., 2009] - Mooij, Janzing, Peters & Scholkopf, "Regression
  by Dependence Minimization and its Application to Causal Inference in
  Additive Noise Models", ICML 2009.
- [Buhlmann et al., 2014] - Buhlmann, Peters & Ernest, "CAM: Causal
  Additive Models, High-Dimensional Order Search and Penalized
  Regression", Annals of Statistics 2014.
- [Rolland et al., 2022] - "Score Matching Enables Causal Discovery of
  Nonlinear Additive Noise Models" (SCORE), ICML 2022.
- [Montagna et al., 2023] - "Causal Discovery with Score Matching on
  Additive Models with Arbitrary Noise" (NoGAM), CLeaR 2023; and
  "Scalable Causal Discovery with Score Matching" (DAS), CLeaR 2023.
- [Wang et al., 2021] - "Ordering-Based Causal Discovery with
  Reinforcement Learning" (CORL), IJCAI 2021.
- [Lippe et al., 2022] - Lippe, Cohen & Gavves, "Efficient Neural Causal
  Discovery without Acyclicity Constraints" (ENCO), ICLR 2022.
- [Zantedeschi et al., 2023] - "DAG Learning on the Permutahedron",
  ICLR 2023.
- [Grover et al., 2019] - Grover, Wang, Zweig & Ermon, "Stochastic
  Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
