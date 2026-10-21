# Appendix — Closed-Form Hodge Decomposition of the Direction Field

Companion to `docs/ideas/ORDER_FACTORIZED_VISION.md` (section 3). Derives
the closed-form projection of the antisymmetric direction field
$A \in \mathbb{R}^{d \times d}$, $A_{ij} = -A_{ji}$ (the `A_anti` term of
`GatedSelfAttention`), onto the gradient-flow subspace on the complete
graph $K_d$, following [Jiang et al., 2011].

## A.1 Least-squares projection onto gradient flows

The Hodge decomposition $A = G + C$ is obtained by projecting $A$ onto the
subspace of gradient flows $G_{ij} = s_j - s_i$ for a node potential
$s \in \mathbb{R}^d$:

$$
s^\star = \arg\min_{s \in \mathbb{R}^d} \; \sum_{i < j} \big( A_{ij} - (s_j - s_i) \big)^2,
\qquad G_{ij} = s^\star_j - s^\star_i, \qquad C = A - G .
$$

Let the **divergence** (net outflow) of $A$ at node $k$ be

$$
b_k \;=\; \sum_{j} A_{kj} .
$$

## A.2 Normal equations

The residual $r_{ij} = A_{ij} - (s_j - s_i)$ is antisymmetric because $A$
is. Differentiating the objective with respect to $s_k$, pairs containing
$k$ contribute $2 r_{kj}$ each (pairs $(i,k)$ contribute $-2 r_{ik} =
2 r_{ki}$), giving

$$
\frac{\partial}{\partial s_k} \sum_{i<j} r_{ij}^2
= 2 \sum_{j \neq k} \big( A_{kj} - s_j + s_k \big)
= 2 \big( b_k + d\, s_k - \textstyle\sum_j s_j \big) .
$$

Setting the gradient to zero yields the graph-Laplacian system

$$
L s^\star = -b,
\qquad
L = d I - \mathbf{1}\mathbf{1}^{\!\top}
$$

where $L$ is the Laplacian of the complete graph $K_d$. The system is
determined up to an additive constant (gauge freedom of the potential); fix
it by the centered gauge $\sum_j s_j = 0$.

## A.3 Closed form on $K_d$

For the complete graph, the Laplacian pseudoinverse is centering scaled by
$1/d$:

$$
L^{\dagger} = \frac{1}{d}\left( I - \frac{1}{d}\mathbf{1}\mathbf{1}^{\!\top} \right).
$$

Since the divergence of any antisymmetric matrix sums to zero
($\sum_k b_k = \sum_{k,j} A_{kj} = 0$), the centering is a no-op and the
solution collapses to

$$
\boxed{\; s^\star_k = -\frac{1}{d} \sum_{j} A_{kj} = -\frac{b_k}{d} \;}
$$

i.e. **the potential is minus the scaled row sum (divergence) of the
field**. Then

$$
G_{ij} = s^\star_j - s^\star_i = \frac{1}{d}\big( b_i - b_j \big),
\qquad
C = A - G ,
$$

all in closed form, $O(d^2)$, no iteration.

## A.4 Sign conventions

A **source** node has all-outgoing edges, so $b_k > 0$ large and
$s^\star_k$ very **negative**; a sink has $s^\star_k$ maximal. Two
consistent conventions:

1. **Potential convention** (used above): $s = -b/d$, causal order
   ascending — "$i$ precedes $j$ iff $s_i < s_j$" (source = minimum
   potential), with $G_{ij} = s_j - s_i$.
2. **Ranking-score convention** (used in the main document): define
   $r = b/d$ (plain scaled row sums), causal order **descending** — "$i$
   precedes $j$ iff $r_i > r_j$" (source = maximum row sum), with
   $G_{ij} = r_i - r_j$.

The two are related by $r = -s$. Convention 2 avoids the minus sign and
matches the intuition "source = all-outgoing rows = maximal row sum"; the
row sums $r$ feed the Gumbel-Sinkhorn / Plackett-Luce relaxation directly.

## A.5 Curl energy in closed form

By orthogonality of the Hodge decomposition, $\|A\|_F^2 = \|G\|_F^2 +
\|C\|_F^2$. On $K_d$ with $s^\star$ centered, the gradient-flow energy is

$$
\|G\|_F^2 = \sum_{i \neq j} (s^\star_j - s^\star_i)^2 = 2 d \, \|s^\star\|_2^2 ,
$$

so the acyclicity diagnostic costs one Frobenius-norm difference — no
explicit projection is ever materialized:

$$
\boxed{\; \|C\|_F^2 = \|A\|_F^2 - 2 d \, \|s^\star\|_2^2 \;}
$$

$\|C\|_F^2 = 0$ iff $A$ is a pure gradient flow, i.e. iff the direction
field is exactly a total order.

## A.6 Beyond the complete graph

The row-sum shortcut relies on $K_d$. Once the direction field is masked by
the existence cone (section 3 of the main document), the flow lives on a
sparser graph $\mathcal{G}$ and the closed form generalizes to

$$
s^\star = -L_{\mathcal{G}}^{\dagger}\, b ,
$$

with $L_{\mathcal{G}}$ the Laplacian of $\mathcal{G}$; the potentials then
require solving a sparse Laplacian system instead of a row sum.

## A.7 Triangle refinement (optional diagnostic)

The curl component further decomposes (non-uniquely) into per-triangle
cyclic flows:

$$
C = \sum_{i < j < k} w_{ijk}\, \chi_{ijk},
$$

where $\chi_{ijk}$ is the unit circulation on triangle $(i,j,k)$. This
identifies *which* triples carry the inconsistency rather than only its
total mass $\|C\|_F^2$ — useful for debugging non-transitive direction
fields during training.

## Reference

- [Jiang et al., 2011] — Jiang, Lim, Yao & Ye, "Statistical Ranking and
  Combinatorial Hodge Theory", Mathematical Programming, 2011.
