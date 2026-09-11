# Linear Value Embedding for the Causal Selection Component

Rationale note for the `linear_per_node` source-value embedding, as used in
`experiments/6_INVESTIGATIONS/BILEVEL_GATE/gate_d20_linear` and
`gate_bkd_loo_d20_linear`.  Originated as an intuition (conversation,
2026-09): attention is a weighted average — a linear operation in its
inputs — so if the value embedding is also linear, the whole causal
selection component is linear in the nodal values, and superposition
holds where it matters.

## 1. The precise statement

With `linear_per_node` the source value stream is

    h_j = x_j * w_j        (per-node direction w_j in R^d, no bias)

i.e. a rank-1, information-preserving map of the scalar value (x_j is
recoverable from h_j up to the scale |w_j|).  Attention aggregation is

    m_i = sum_j A_ij * V h_j = sum_j (A_ij * V w_j) * x_j

so the message arriving at node i is a **linear function of the raw nodal
values**, with input-dependent coefficients c_ij = A_ij * V w_j.  The
causal-selection component is an *input-dependent linear operator* on x,
and superposition holds exactly (holding attention weights fixed):

    m_i(x + x') = m_i(x) + m_i(x')
    A_ij V h_j + A_ik V h_k = V (A_ij w_j x_j + A_ik w_k x_k)

i.e. a hidden vector "representing the sum of two nodes" literally is the
sum of their hidden vectors.

All nonlinearity is quarantined into two well-defined places:

1. the attention **weights** (softmax over QK scores — the *selection*
   nonlinearity), and
2. the downstream **output MLP** (the *mechanism* nonlinearity).

The model factorizes as: linear transport of values -> nonlinear routing
-> nonlinear readout.

## 2. Why this is more than aesthetics

### 2.1 Faithfulness of the HSIC / commit / LOO evidence

The whole structural apparatus (per-row HSIC on residuals, centroid
commits, LOO contrasts) asks: "does key j carry unique information about
row i's residual?"  With an MLP value embedding, the encoder can
*manufacture or destroy* dependence before attention ever sees it — two
variables can be made perceptually similar by a learned encoder, and HSIC
measured downstream confounds "the encoder folded the signal away" with
"the edge is absent".  With a linear (rank-1) embedding the value stream
is information-preserving, so dependence structure measured in hidden
space is faithful to dependence in the data.  For a *selection* component
whose job is to expose conditional dependence, an information-losing
encoder is a confound; a linear one (almost) is not.

### 2.2 The linear-Gaussian null model is exact

If the true SCM were linear Gaussian, the Bayes-optimal predictor
E[X_i | parents] is linear in the parents, so an architecture that is
linear-in-values with attention as coefficient matrix can represent the
truth *without the output MLP doing any work*, and A converges to a
NOTEARS/DAGMA-style weighted adjacency (cf. the `linear_per_node`
docstring: "NOTEARS/DAGMA-style first-layer behaviour").  That is a
controllable sanity regime.  The benchmark datasets are nonlinear
Gaussian, so the output MLP still models the mechanisms — but the
selection layer operating in a linear regime gives attention coefficients
a direct interpretation as partial-regression-like weights, which is what
the centroid-commit readout presumes when it quantizes queries to key
centroids.

### 2.3 Coherent centroid arithmetic

The commit controller's core object is a *centroid* of key embeddings,
c(S').  Averaging is only meaningful in a space where sums/means
correspond to sums of the underlying quantities.  With MLP embeddings,
mean_j h_j(x_j) has no relation to any function of the x_j; with linear
embeddings, sum_j h_j = sum_j x_j w_j — the centroid of the
representations *is* the representation of the aggregate.  Subset-centroid
semantics become well-defined.

## 3. Where the argument breaks / what to watch

- **Softmax still renormalizes.**  Superposition holds for the value path
  *given* A, but A itself is scale-sensitive through the scores.  With
  `remove_query/key_projection: true` and orthogonal keys this is
  mitigated, but routing stays nonlinear: "fully linear from values to
  hidden" really means "linear conditional on the gate state".
- **Rank-1 bottleneck.**  Each node's value stream lives on a line in
  R^d: maximally faithful, minimally expressive.  Per-node *normalization*
  is impossible — a large-magnitude node dominates attention output norms,
  and scale differences between nodes cannot be corrected by the encoder.
  Watch warmup `train_x_mae` against the mlp_per_node arms for a capacity
  cost.
- **The nonlinearity has to live somewhere.**  For nonlinear data the
  output MLP must absorb everything the encoder used to provide; if
  reconstruction degrades, that is the trade, not a bug.
- **Synergy with BKD / LOO.**  Linear value transport means dropping key j
  removes exactly A_ij V w_j x_j from the message — the LOO contrast
  measures a *cleanly attributable* delta (linear in x_j).  LOO measures
  unique contribution; linearity guarantees the measured contribution is
  not an encoder artifact.  This is the motivation for pairing the two
  changes in `gate_bkd_loo_d20_linear`.

## 4. Diagnostic use of the linear arms

Because A becomes interpretable as a NOTEARS-style weighted adjacency (up
to per-node directions), these runs give a cheap extra check: compare the
*attention-score* SHD against the *commit-driven* structure.  If linearity
helps, the two should agree more closely than in the mlp arms; any
disagreement localizes the failure to either the gate (accepts too rare)
or the readout (dense-subset bias) — exactly the ambiguity left open by
`gate_bkd_d20_12411404` (727 probes / 14 accepts, ~1e-3 noise-level
deltas, commit subset mean 19 -> 13.65/19).
