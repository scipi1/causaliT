# Architecture Versions

This file crystallizes the model architecture versions of causaliT. It
documents **architecture only** — optimization objectives, loss functions,
schedules and adaptive-training controllers are out of scope.

---

## Arch v3 — LATEST (2026-09-19)

**Reference config:**
`experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film2_evalbkd/config.yaml`

**One-liner:** single-head GatedCross/GatedSelf attention over
`svfa`-composed linear_per_node value embeddings + max-norm-1 variable
embeddings, with orthogonal structure embeddings, free centroid-initialized
queries, no Q/K projections, learnable per-node query-norm budgets, feeding
deep (4-layer, hidden-64) per-node decoder MLPs FiLM-conditioned at every
layer on the detached applied-adjacency row.

### 1. Model object

| Key | Value |
|---|---|
| `model.model_object` | `AttentionSelectorLayer` |

Single selector layer with cross- and self-attention stacks.

### 2. Input embeddings (`ds_embed_S` / `ds_embed_X`, identical streams)

| Key | Value | Role |
|---|---|---|
| module 1 `embed` | `linear_per_node` | value |
| module 1 `input_dim` / `embedding_dim` | `1` / `val_emb_hidden (= d_model_set = 20)` | per-variable scalar→vector linear map |
| module 2 `embed` | `nn_embedding` | structure (variable identity) |
| module 2 `embedding_dim` / `max_norm` / `padding_idx` | `var_emb_hidden (= 20)` / `1` / `0` | |
| module 3 `embed` | `mask` (`label: value_missing`) | missing-value mask |
| `comps_embed_S` / `comps_embed_X` (`experiment.comps_embed`) | `svfa` | structure–value fused addition |

### 3. Attention blocks

| Key | Value |
|---|---|
| `experiment.attention_type` | `GatedCrossAttention` |
| `experiment.self_attention_type` | `GatedSelfAttention` |
| `experiment.n_heads` | `1` |
| `experiment.shared_dag_across_heads` | `true` |
| `experiment.init_tau_cross` / `init_tau_self` | `0.5` / `0.5` |
| `experiment.init_gamma` / `init_zeta` | `-1.1` / `1.1` |
| `experiment.dir_tau_self` | `0.6666666667` |
| `experiment.init_edge_offset` | `auto` |

### 4. Query / Key parameterization (structure-aware)

| Key | Value |
|---|---|
| `experiment.struct_embedding_type` | `orthogonal_fixed` |
| `experiment.key_projection_type` | `orthogonal` |
| `experiment.orthogonal_key_scale` | `false` |
| `experiment.free_query_embedding` | `true` |
| `experiment.query_centroid_init` | `true` |
| `experiment.query_centroid_max_p` | `0.8209` |
| `experiment.remove_query_projection` / `remove_key_projection` | `true` / `true` |
| `experiment.homogeneous_nodes` | `true` |
| `experiment.shared_query` / `shared_key` | `false` / `false` |
| `experiment.commutator_direction_mode` | `skew_query` |
| `experiment.query_norm` / `normalize_query` | `true` / `true` |
| `experiment.query_fanin_scale` | `auto` |
| `experiment.query_norm_learnable` | `true` (per-node learnable budget) |
| `experiment.query_norm_init_scale` / `query_norm_target` | `1.0` / `1.0` |
| `experiment.value_structure_injection` / `value_structure_query_injection` | `none` / `none` |
| `experiment.use_gain_softmax` / `gain_data` | `false` / `false` |

### 5. Core dimensions / block settings

| Key | Value |
|---|---|
| `experiment.d_model_set` | `20` |
| `experiment.d_ff` / `d_ff_mult` | `null` → `4.0 × d_model_set` (= 80) |
| `experiment.d_qk` / `d_qk_mult` | `null` → `1.0 × d_model_set` (= 20) |
| `model.kwargs.activation` | `gelu` |
| `model.kwargs.norm` / `use_final_norm` | `none` / `false` |
| all dropouts (`dropout_emb`, `dropout_attn_out`, `dropout_ff`, `dropout_qkv`, `attention_dropout`) | `0.0` |
| `experiment.n_nodes` / `n_source` / `n_input` | `20` / `2` / `18` |

### 6. Downstream regressor — structure-aware expressive per-node MLP with multi-layer FiLM

| Key | Value |
|---|---|
| `experiment.per_node_output` | `true` |
| `experiment.per_node_output_hidden` | `64` |
| `experiment.per_node_output_layers` | `4` (deep per-node decoder head) |
| `model.kwargs.output_mlp_layers` / `output_mlp_activation` | `4` / `relu` |
| `experiment.per_node_adjacency_context` | `film` |

Every per-node decoder MLP is **FiLM-conditioned on the detached
applied-adjacency row** (post top-k / batch-key-dropout). The shared
conditioner emits a `(γ_k, β_k)` pair per hidden activation (packed output,
zero-initialized last layer ⇒ identity mapping at step 0), following Perez
et al. 2018 (feature-wise modulation at every layer). With
`per_node_output_layers: 4`, modulation is applied at each hidden
activation of the deepened per-node head.

### 7. Notes (architecture-visible, controller-driven)

- The model instantiates `batch_key_dropout*` modules
  (`batch_key_dropout: 0.8`, `deterministic: true`, `min_keys: 0`,
  `eval: true`, `eval_seed: 12345`) so that the applied adjacency used by
  the value stream and by the FiLM context can be sparsified by the
  external training controller; at model level `min_keys` stays 0 and the
  schedule is inert without the controller. The ladder behavior itself is
  out of scope here (optimization/training concern).


