"""One-off patch: add FiLM conditioning to PerNodeMLPHead (mlp_head.py)."""
import io

p = "causaliT/core/modules/mlp_head.py"
s = io.open(p, encoding="utf-8").read()

old_ctor = (
    "        var_id_offset: int = 1,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        assert activation in self.ACTIVATIONS, (\n"
    "            f\"Invalid activation '{activation}'. Choose from {list(self.ACTIVATIONS)}.\"\n"
    "        )\n"
    "        self.d_model = d_model\n"
    "        self.out_dim = out_dim\n"
    "        self.num_variables = num_variables\n"
    "        self.d_hidden = d_hidden\n"
    "        self.var_id_offset = var_id_offset\n"
)
new_ctor = (
    "        var_id_offset: int = 1,\n"
    "        film_context_dim: int = 0,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        assert activation in self.ACTIVATIONS, (\n"
    "            f\"Invalid activation '{activation}'. Choose from {list(self.ACTIVATIONS)}.\"\n"
    "        )\n"
    "        self.d_model = d_model\n"
    "        self.out_dim = out_dim\n"
    "        self.num_variables = num_variables\n"
    "        self.d_hidden = d_hidden\n"
    "        self.var_id_offset = var_id_offset\n"
    "        self.film_context_dim = int(film_context_dim)\n"
)
assert s.count(old_ctor) == 1, "ctor anchor not unique"
s = s.replace(old_ctor, new_ctor)

old_tail = (
    "            for _ in range(num_variables)\n"
    "        ])\n"
    "\n"
    "    def forward(\n"
)
new_tail = (
    "            for _ in range(num_variables)\n"
    "        ])\n"
    "\n"
    "        # FiLM conditioner (shared across nodes; the context row is already\n"
    "        # node-specific).  Zero-init last layer -> gamma=1, beta=0 at init.\n"
    "        if self.film_context_dim > 0:\n"
    "            film_hidden = max(32, 2 * d_hidden)\n"
    "            self.film = nn.Sequential(\n"
    "                nn.Linear(self.film_context_dim, film_hidden),\n"
    "                nn.GELU(),\n"
    "                nn.Linear(film_hidden, 2 * d_hidden),\n"
    "            )\n"
    "            nn.init.zeros_(self.film[-1].weight)\n"
    "            nn.init.zeros_(self.film[-1].bias)\n"
    "        else:\n"
    "            self.film = None\n"
    "\n"
    "    def forward(\n"
)
assert s.count(old_tail) == 1, "tail anchor not unique"
s = s.replace(old_tail, new_tail)

old_fwd = (
    "        if context is not None:\n"
    "            assert context.shape[:2] == x.shape[:2], (\n"
    "                f\"context (B, L) dims {tuple(context.shape[:2])} must \"\n"
    "                f\"match x {tuple(x.shape[:2])}.\"\n"
    "            )\n"
    "            x = torch.cat([x, context], dim=-1)\n"
)
new_fwd = (
    "        if self.film is not None:\n"
    "            # FiLM mode: context modulates the hidden activation\n"
    "            # multiplicatively; it is NOT concatenated.\n"
    "            if context is None:\n"
    "                raise ValueError(\n"
    "                    \"PerNodeMLPHead built with film_context_dim > 0 \"\n"
    "                    \"requires a context tensor in forward().\"\n"
    "                )\n"
    "            assert context.shape[:2] == x.shape[:2], (\n"
    "                f\"context (B, L) dims {tuple(context.shape[:2])} must \"\n"
    "                f\"match x {tuple(x.shape[:2])}.\"\n"
    "            )\n"
    "            assert context.size(-1) == self.film_context_dim, (\n"
    "                f\"FiLM context width {context.size(-1)} != \"\n"
    "                f\"film_context_dim {self.film_context_dim}.\"\n"
    "            )\n"
    "            gb = self.film(context)\n"
    "            gamma = 1.0 + gb[..., : self.d_hidden]\n"
    "            beta = gb[..., self.d_hidden :]\n"
    "        elif context is not None:\n"
    "            assert context.shape[:2] == x.shape[:2], (\n"
    "                f\"context (B, L) dims {tuple(context.shape[:2])} must \"\n"
    "                f\"match x {tuple(x.shape[:2])}.\"\n"
    "            )\n"
    "            x = torch.cat([x, context], dim=-1)\n"
)
assert s.count(old_fwd) == 1, "forward anchor not unique"
s = s.replace(old_fwd, new_fwd)

old_vec = (
    "        all_outs = torch.stack(\n"
    "            [mlp(x) for mlp in self.mlps], dim=2\n"
    "        )  # (B, L, num_variables, out_dim)\n"
)
new_vec = (
    "        if self.film is not None:\n"
    "            # Linear -> Act -> Dropout, then FiLM on the hidden activation,\n"
    "            # then the output projection.  Submodule indexing keeps the\n"
    "            # legacy nn.Sequential checkpoint layout (mlps.i.0/3).\n"
    "            all_outs = torch.stack(\n"
    "                [mlp[3](gamma * mlp[2](mlp[1](mlp[0](x))) + beta)\n"
    "                 for mlp in self.mlps],\n"
    "                dim=2,\n"
    "            )  # (B, L, num_variables, out_dim)\n"
    "        else:\n"
    "            all_outs = torch.stack(\n"
    "                [mlp(x) for mlp in self.mlps], dim=2\n"
    "            )  # (B, L, num_variables, out_dim)\n"
)
assert s.count(old_vec) == 1, "vectorised anchor not unique"
s = s.replace(old_vec, new_vec)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("mlp_head.py patched: FiLM ctor + forward")
