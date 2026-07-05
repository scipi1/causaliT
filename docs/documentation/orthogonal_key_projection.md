# Orthogonal Structural Embeddings and the Isometric Key Projection

**Scope.** This document explains the mathematical formulation behind the
`orthogonal_struct_embedding` and `key_projection_type="orthogonal"` options of
`AttentionSelectorLayer`, *why* they are needed, and *how* orthogonality is
actually achieved and preserved end-to-end through the attention logit.

**Code references.**
- `causaliT/core/modules/orthogonal_embedding.py` — `OrthogonalMaskEmbedding`
- `causaliT/core/modules/orthogonal_linear.py` — `OrthogonalLinear` (Cayley)
- `causaliT/core/architectures/attention_selector/model.py` — wiring
- `tests/test_atsel_orthogonal_key_projection.py` — property tests

---

## 1. Background: the combined cross-attention selector

`AttentionSelectorLayer` uses a **single** cross-attention block. For each child
variable $X_i$ we form a query from its (value-blanked) identity, and we offer
all candidate parents $[S_1,\dots,S_{L_S},\,X_1,\dots,X_{L_X}]$ as keys/values:

$$
Q = q(X_i), \qquad
K = \big[\, k(S_1),\dots,k(S_{L_S}),\; k(X_1),\dots,k(X_{L_X}) \,\big].
$$

The attention logit between child $i$ and candidate parent $j$ is

$$
\mathrm{score}_{ij} \;=\; \big(e_q(i)\,W_Q\big)\,\big(e_k(j)\,W_K\big)^{\!\top},
\tag{1}
$$

where $e_q(i),\,e_k(j)\in\mathbb{R}^{d_{\text{model}}}$ are the structural
(identity) embeddings and $W_Q,\,W_K$ are the shared query/key projections.
Splitting the resulting matrix by column gives the two learned adjacencies:

$$
A[:,:,{:}L_S] \;\to\; S\to X \ \text{(exogenous parents)}, \qquad
A[:,:,L_S{:}] \;\to\; X\to X \ \text{(endogenous, diagonal masked to }0).
$$

---

## 2. The problem we are solving

### 2.1 Empirical observation

In `experiments/2_ARCH_STUDY/ORTH_EMBED_STUDY/...`, replacing the standard
lookup embedding with `OrthogonalMaskEmbedding` on the structural (Q/K) stream
**helped the $S\to X$ block but not the $X\to X$ block**. We know a clean
solution exists: the oracle (`control_shd=0`) sits at (or very near) the minimum
of the HSIC landscape for this architecture
(`experiments/1_FOUNDATIONS/3_ORACLE/2_ATT_SEL`). So the objective is correct;
the difficulty is that the *parameterisation* couples edges that should be
learnable independently.

### 2.2 The confounding: $X$ plays two roles with one embedding

Each $X_i$ appears in the combined attention **twice**:

- as a **query** ($X_i$ as a *child* selecting its own parents), and
- as a **key** ($X_i$ as a *candidate parent* offered to every other $X_j$).

Source nodes $S$ only ever appear as keys. With a **single shared** structural
embedding $e(X_i)$ feeding both the query and the key roles, any gradient that
updates $e(X_i)$ to improve $X_i\leftarrow S$ (query role) simultaneously
perturbs $e(X_i)$ in its key role, i.e. it changes how $X_i$ is offered as a
parent to all $X_j$. The two mechanisms

$$
X_i \leftarrow S \quad(\text{$X_i$ as child, query}), \qquad
X_j \leftarrow X_i,\; j\neq i \quad(\text{$X_i$ as parent, key})
$$

are therefore **not free to be learned independently** — a structural
confounding baked into the parameter sharing. This is a plausible reason the
$X\to X$ block did not benefit from orthogonal embeddings while $S\to X$ did.

Two complementary remedies address different halves of this coupling:

1. **Decouple the query from the key** for $X$ — give the query its own
   embedding (`free_query_embedding`, `FreeQueryEmbedding`). This separates
   "$X_i$ as child" from "$X_i$ as parent" at the *embedding* level.
2. **Preserve embedding orthogonality through the projection** — the subject of
   the rest of this document. Even with orthogonal embeddings, the *projected*
   keys can leak information across variables unless $W_K$ is constrained.

---

## 3. Orthogonal structural embeddings

### 3.1 Construction (`OrthogonalMaskEmbedding`)

Give each variable a **disjoint block** of $k$ dimensions of the
$d_{\text{model}}$ space via a fixed binary mask
$m_j\in\{0,1\}^{d_{\text{model}}}$, and embed its value with a shared linear map,
gated by the mask:

$$
e(\text{var}_j,\, v) \;=\; s\,\big(W_v\,v\big)\odot m_j,
\qquad s=\sqrt{\tfrac{d_{\text{model}}}{k}}.
\tag{2}
$$

The masks tile the space without overlap. When $S$ and $X$ share the same
$d_{\text{model}}$ (they are concatenated as keys), the two groups must occupy
**disjoint** partitions, achieved via `mask_start_dim`:

$$
k=\Big\lfloor\tfrac{d_{\text{model}}}{L_S+L_X}\Big\rfloor,\qquad
S:\ [0,\,L_S k),\qquad
X:\ [L_S k,\,(L_S+L_X)k).
$$

### 3.2 Why the raw keys are orthogonal

Let $\operatorname{supp}(m_j)$ denote the non-zero index set of $m_j$. Because
$\operatorname{supp}(m_i)\cap\operatorname{supp}(m_j)=\varnothing$ for $i\neq j$,
the element-wise product annihilates every cross term:

$$
\big\langle e(\text{var}_i),\, e(\text{var}_j)\big\rangle
= s^2\!\sum_{d} (W_v v_i)_d (W_v v_j)_d\,(m_i)_d (m_j)_d
= 0,
\quad\text{since } (m_i)_d (m_j)_d = 0\ \ \forall d.
\tag{3}
$$

This holds for **any** input values and for **any** learned $W_v$ — the masks
are fixed buffers (not trainable), so orthogonality is a hard architectural
invariant, not something that must be learned or frozen.

> **Note on "freezing".** A natural question is *which parameter we freeze to
> keep orthogonality*. The answer: **nothing needs freezing**. Orthogonality
> comes from the fixed disjoint masks; the only trainable part $W_v$ cannot
> break (3) because disjoint supports kill the cross terms regardless of $W_v$.

---

## 4. The leak: projection destroys orthogonality

The attention logit (1) is **not** computed on the raw keys
$k_j := e_k(j)$; it uses the **projected** keys $k_j W_K$. Writing the
query-side vector $q_i := e_q(i) W_Q$, we have
$\mathrm{score}_{ij} = q_i\,(k_j W_K)^{\top}$, and the geometry that decides
"does key $i$ look like key $j$" is the inner product of projected keys:

$$
\big\langle k_i W_K,\; k_j W_K\big\rangle
= k_i\,\big(W_K^{\top} W_K\big)\,k_j^{\top}.
\tag{4}
$$

With an **unconstrained** $W_K$ (a plain `nn.Linear`), the Gram matrix
$G := W_K^{\top} W_K$ is an arbitrary symmetric positive-(semi)definite matrix.
Even if $k_i \perp k_j$ (i.e. $k_i k_j^{\top}=0$), the bilinear form
$k_i\,G\,k_j^{\top}$ is generally **non-zero**:

$$
k_i \perp k_j \;\;\not\Longrightarrow\;\; k_i\,G\,k_j^{\top}=0
\qquad(\text{unless } G = c\,I \text{ on the relevant subspace}).
\tag{5}
$$

So the embedding-level orthogonality (3) is silently thrown away at the
projection step. Off-diagonal entries of $G$ mix the disjoint blocks and let
information about variable $i$ bleed into the logit for variable $j$ — exactly
the cross-variable coupling the orthogonal embeddings were meant to remove.

> **Common algebra slip.** Orthogonality of two vectors is a statement about
> their **inner product** $k_i k_j^{\top}=0$ (a scalar), *not* about their
> **outer product** $k_j^{\top} k_i$ (a rank-1 matrix). The outer product is
> never zero for non-zero vectors; the only orthogonality-implied fact about it
> is that its **trace** vanishes,
> $\operatorname{tr}\!\big(k_j^{\top} k_i\big) = k_i k_j^{\top} = 0$. The object
> that must be controlled to preserve orthogonality after projection is the Gram
> matrix $W_K^{\top} W_K$ in (4).

---

## 5. The fix: constrain $W_K$ to an isometry

### 5.1 Condition

We want the projection to **preserve inner products**:

$$
\big\langle k_i W_K,\; k_j W_K\big\rangle = \big\langle k_i,\, k_j\big\rangle
\qquad \forall\, k_i, k_j.
\tag{6}
$$

By (4) this is equivalent to requiring $W_K$ to be an **isometry**:

$$
\boxed{\,W_K^{\top} W_K = I\,}.
\tag{7}
$$

Under (7), orthogonal raw keys stay orthogonal after projection:

$$
k_i k_j^{\top} = 0
\;\;\Longrightarrow\;\;
\big\langle k_i W_K,\, k_j W_K\big\rangle = k_i\, I\, k_j^{\top} = 0.
\tag{8}
$$

The model keeps full freedom to **rotate** the shared representation to fit the
data (an isometry is exactly a rotation/reflection, possibly into a
higher-dimensional space); it simply may not **shear** it in a way that
collapses the disjoint blocks together.

### 5.2 Dimension requirement $d_{qk}\ge d_{\text{model}}$

An isometry from $\mathbb{R}^{d_{\text{model}}}$ maps onto an orthonormal set of
$d_{\text{model}}$ columns living in $\mathbb{R}^{d_{qk}}$. Such a set exists
only if the codomain has at least as many dimensions:

$$
W_K\in\mathbb{R}^{d_{qk}\times d_{\text{model}}},\quad W_K^{\top}W_K = I
\;\;\Longrightarrow\;\; d_{qk}\ge d_{\text{model}}.
\tag{9}
$$

`AttentionSelectorLayer` enforces this with an explicit guard; requesting
`key_projection_type="orthogonal"` with $d_{qk}<d_{\text{model}}$ raises a
`ValueError`.

### 5.3 Parametrisation (`OrthogonalLinear`, Cayley transform)

We need a *differentiable, unconstrained* parametrisation of the isometry so it
can be trained by gradient descent. `OrthogonalLinear` uses the **Cayley
transform**. Fill the strict upper triangle of a matrix
$\tilde A\in\mathbb{R}^{d_{qk}\times d_{qk}}$ with unconstrained parameters, and
make it **skew-symmetric**:

$$
A = \tilde A - \tilde A^{\top}, \qquad A^{\top} = -A.
\tag{10}
$$

For any skew-symmetric $A$, the Cayley map yields an orthogonal matrix

$$
Q = (I - A)(I + A)^{-1}, \qquad Q^{\top} Q = I,
\tag{11}
$$

with $Q = I$ when $A = 0$, so training starts near the identity. When
$d_{qk} > d_{\text{model}}$ we take the first $d_{\text{model}}$ columns of $Q$
(a subset of an orthonormal basis is still orthonormal):

$$
W_K = Q_{[:,\,:d_{\text{model}}]} \;\;\Longrightarrow\;\; W_K^{\top} W_K = I.
\tag{12}
$$

An **optional learnable scalar** $s>0$ (`orthogonal_key_scale`) rescales the
projection, $W_K \leftarrow s\,W_K$. This preserves orthogonality (zero stays
zero) while letting the logit magnitude / temperature adapt; inner products are
then preserved up to $s^2$:

$$
\big\langle k_i\,(sW_K),\; k_j\,(sW_K)\big\rangle
= s^2\,\big\langle k_i,\, k_j\big\rangle.
\tag{13}
$$

For $i\neq j$ the right-hand side is still $0$, so orthogonality is exact
regardless of $s$.

---

## 6. End-to-end guarantee

Chaining the results, for $i\neq j$:

$$
\underbrace{k_i k_j^{\top} = 0}_{\text{disjoint masks, (3)}}
\;\;\wedge\;\;
\underbrace{W_K^{\top} W_K = I}_{\text{isometric } W_K,\,(7)}
\;\;\Longrightarrow\;\;
\underbrace{\big\langle k_i W_K,\, k_j W_K\big\rangle = 0}_{\text{preserved, (8)}}.
$$

So the structural key of one variable contributes **nothing** to the attention
logit of another variable through shared-representation leakage. Combined with
`free_query_embedding` (which decouples the $X$ query from the $X$ key), the
architecture can learn $X_i\leftarrow S$ and $X_j\leftarrow X_i$ as independent
decisions — removing the confounding of Section 2 — while still being able to
reach the oracle solution that we know minimises HSIC.

**What this does *not* do.** Orthogonality here concerns the *identity/structure*
channel only. The value stream ($V$, residual, FFN, MLP head) is untouched, so
reconstruction capacity is unchanged. Orthogonality also does not by itself
*select* the correct parents — it removes a spurious coupling so that the HSIC /
sparsity signals can drive selection cleanly.

---

## 7. How to use it

In `config_attention_selector.yaml` (or any experiment override):

```yaml
experiment:
  # Make raw S/X structural keys mutually orthogonal (disjoint d_model blocks).
  orthogonal_struct_embedding: true

  # Preserve that orthogonality through the key projection (W_K^T W_K = I).
  key_projection_type: "orthogonal"   # "linear" (default) | "orthogonal"
  orthogonal_key_scale: true          # learnable scalar s (keeps ⊥, scales logits)

  # Required for the isometry to exist:
  d_qk: <value >= d_model>

  # Recommended for a clean, interpretable structure:
  n_heads: 1

  # Complementary decoupling of the X query from the X key role:
  free_query_embedding: true
```

**Defaults are unchanged** (`key_projection_type: "linear"`), so this is an
opt-in for ablations. The most meaningful configuration is
`orthogonal_struct_embedding=True` + `key_projection_type="orthogonal"` +
`n_heads=1`.

---

## 8. Verification

`tests/test_atsel_orthogonal_key_projection.py` asserts the properties derived
above:

- **Construction / wiring** — `key_projection` is `OrthogonalLinear` iff the
  flag is set; `verify_orthonormality()` holds with the scale on and off.
- **Isometry guard** — $d_{qk} < d_{\text{model}}$ with `"orthogonal"` raises.
- **Orthogonality preservation, eq. (8)** — projecting two orthogonal raw keys
  keeps their inner product at $\approx 0$, whereas an unconstrained (perturbed)
  `nn.Linear` breaks it.
- **Inner-product preservation up to scale, eq. (13)** —
  $\langle Wx, Wy\rangle = s^2\,\langle x, y\rangle$.
- **Forward/backward** — shapes are unchanged and the Cayley $A$-parameters
  (`skew_params`) receive gradient.

---

## 9. Symbol table

| Symbol | Meaning |
|---|---|
| $L_S,\,L_X$ | number of source ($S$) and observed ($X$) variables |
| $d_{\text{model}}$ | model / embedding dimension |
| $d_{qk}$ | query–key projection dimension (per head) |
| $k$ | dims per variable in the orthogonal partition, $k=\lfloor d_{\text{model}}/(L_S+L_X)\rfloor$ |
| $m_j$ | fixed binary mask (support $=$ variable $j$'s block) |
| $e_k(j),\,e_q(i)$ | structural key / query embedding |
| $W_Q,\,W_K$ | shared query / key projections |
| $G = W_K^{\top} W_K$ | key Gram matrix; $G=I \Leftrightarrow$ isometry |
| $A$ | skew-symmetric matrix (Cayley), $A^{\top}=-A$ |
| $Q=(I-A)(I+A)^{-1}$ | orthogonal matrix from the Cayley transform |
| $s$ | optional learnable positive scale on $W_K$ |
