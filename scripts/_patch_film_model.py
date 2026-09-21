"""One-off patch: per_node_adjacency_context mode parsing ("concat"/"film")
in AttentionSelectorLayer (attention_selector/model.py)."""
import io

p = "causaliT/core/architectures/attention_selector/model.py"
s = io.open(p, encoding="utf-8").read()

# --- ctor docstring: document the mode strings ------------------------------
old_doc = (
    "        # Adjacency-context injection into the per-node decoder MLP: when\n"
    "        # True, each per-node MLP input is concatenated with the DETACHED\n"
    "        # applied-adjacency row (B, L_q, L_S+L_X) — the gate weights\n"
    "        # actually used on this sample after hard mask, gain, BKD, top-k\n"
    "        # blanking and dropout — so the (nuisance) regressor knows which\n"
    "        # keys were selected, consistently with stochastic key exclusion.\n"
    "        # Requires per_node_output=True.\n"
    "        per_node_adjacency_context: bool = False,\n"
)
new_doc = (
    "        # Adjacency-context injection into the per-node decoder MLP.  Modes:\n"
    "        # False/None = off; True or \"concat\" = each per-node MLP input is\n"
    "        # concatenated with the DETACHED applied-adjacency row (B, L_q,\n"
    "        # L_S+L_X) — the gate weights actually used on this sample after\n"
    "        # hard mask, gain, BKD, top-k blanking and dropout — so the\n"
    "        # (nuisance) regressor knows which keys were selected, consistently\n"
    "        # with stochastic key exclusion.  \"film\" = FiLM conditioning: a\n"
    "        # shared conditioner MLP maps the context to per-channel scale/\n"
    "        # shift of the decoder hidden activation (zero-init => identical\n"
    "        # to the unconditioned decoder at step 0), without widening the\n"
    "        # MLP input.  Requires per_node_output=True.\n"
    "        per_node_adjacency_context: bool = False,\n"
)
assert s.count(old_doc) == 1, "doc anchor not unique"
s = s.replace(old_doc, new_doc)

# --- ctor body: parse mode, validate ----------------------------------------
old_flag = (
    "        self.per_node_output = bool(per_node_output)\n"
    "        self.per_node_adjacency_context = bool(per_node_adjacency_context)\n"
    "        if self.per_node_adjacency_context and not self.per_node_output:\n"
)
new_flag = (
    "        self.per_node_output = bool(per_node_output)\n"
    "        # Adjacency-context mode: False/None -> off; True -> \"concat\"\n"
    "        # (legacy); \"concat\" / \"film\" select the injection mechanism.\n"
    "        _pac = per_node_adjacency_context\n"
    "        if _pac is True:\n"
    "            _pac = \"concat\"\n"
    "        elif _pac in (False, None):\n"
    "            _pac = None\n"
    "        _pac = str(_pac).lower() if _pac is not None else None\n"
    "        if _pac not in (None, \"concat\", \"film\"):\n"
    "            raise ValueError(\n"
    "                f\"per_node_adjacency_context must be one of False / True / \"\n"
    "                f\"'concat' / 'film', got {per_node_adjacency_context!r}\"\n"
    "            )\n"
    "        self.per_node_adjacency_context = _pac is not None\n"
    "        self._adjacency_context_mode = _pac\n"
    "        if self.per_node_adjacency_context and not self.per_node_output:\n"
)
assert s.count(old_flag) == 1, "flag anchor not unique"
s = s.replace(old_flag, new_flag)

# --- head construction: film keeps base width, concat widens ----------------
old_head = (
    "            # Optional adjacency context widens every per-node MLP input by\n"
    "            # the applied-adjacency row width (L_S + L_X).\n"
    "            _ctx_dim = (\n"
    "                (S_seq_len + X_seq_len) if self.per_node_adjacency_context else 0\n"
    "            )\n"
    "            _base_dim = (\n"
    "                (S_seq_len + X_seq_len) if self.raw_value_adjacency else d_model\n"
    "            )\n"
    "            self.forecaster = PerNodeMLPHead(\n"
    "                d_model=_base_dim + _ctx_dim,\n"
    "                out_dim=out_dim,\n"
    "                num_variables=n_out_nodes,\n"
    "                d_hidden=per_node_output_hidden,\n"
    "                activation=output_mlp_activation,\n"
    "                dropout=output_mlp_dropout,\n"
    "                bias=True,\n"
    "            )\n"
)
new_head = (
    "            # Optional adjacency context widens every per-node MLP input by\n"
    "            # the applied-adjacency row width (L_S + L_X) in \"concat\" mode;\n"
    "            # \"film\" mode keeps the base width and conditions via FiLM.\n"
    "            _ctx_dim = (\n"
    "                (S_seq_len + X_seq_len)\n"
    "                if self._adjacency_context_mode == \"concat\"\n"
    "                else 0\n"
    "            )\n"
    "            _film_dim = (\n"
    "                (S_seq_len + X_seq_len)\n"
    "                if self._adjacency_context_mode == \"film\"\n"
    "                else 0\n"
    "            )\n"
    "            _base_dim = (\n"
    "                (S_seq_len + X_seq_len) if self.raw_value_adjacency else d_model\n"
    "            )\n"
    "            self.forecaster = PerNodeMLPHead(\n"
    "                d_model=_base_dim + _ctx_dim,\n"
    "                out_dim=out_dim,\n"
    "                num_variables=n_out_nodes,\n"
    "                d_hidden=per_node_output_hidden,\n"
    "                activation=output_mlp_activation,\n"
    "                dropout=output_mlp_dropout,\n"
    "                bias=True,\n"
    "                film_context_dim=_film_dim,\n"
    "            )\n"
)
assert s.count(old_head) == 1, "head anchor not unique"
s = s.replace(old_head, new_head)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("model.py patched: adjacency-context mode parsing + FiLM head")
