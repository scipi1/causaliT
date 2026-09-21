from pathlib import Path

p = Path(r'c:\Users\ScipioneFrancesco\Documents\Projects\causaliT\causaliT\core\modules\mlp_head.py')
src = p.read_text(encoding='utf-8')

old_sig = '''        var_id_offset: int = 1,
        film_context_dim: int = 0,
    ):'''
new_sig = '''        var_id_offset: int = 1,
        film_context_dim: int = 0,
        n_layers: int = 2,
    ):'''
assert src.count(old_sig) == 1
src = src.replace(old_sig, new_sig)

old_state = '''        self.var_id_offset = var_id_offset
        self.film_context_dim = int(film_context_dim)
'''
new_state = '''        self.var_id_offset = var_id_offset
        self.film_context_dim = int(film_context_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 2:
            raise ValueError(
                f"PerNodeMLPHead n_layers must be >= 2, got {self.n_layers}."
            )
        # Number of FiLM modulation points: one per HIDDEN activation
        # (multi-layer FiLM); the output projection is never modulated.
        self.n_film_points = self.n_layers - 1
'''
assert src.count(old_state) == 1
src = src.replace(old_state, new_state)

old_mlps = '''        # One independent decoder MLP per variable (node).
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden, bias=bias),
                self.ACTIVATIONS[activation](),
                self.dropout,
                nn.Linear(d_hidden, out_dim, bias=bias),
            )
            for _ in range(num_variables)
        ])'''
new_mlps = '''        # One independent decoder MLP per variable (node): (n_layers - 1)
        # hidden blocks of Linear -> act -> drop, then the output projection.
        # n_layers=2 reproduces the legacy layout exactly (submodule indices
        # 0..3), keeping older checkpoints loadable.
        def _build_mlp():
            layers = [
                nn.Linear(d_model, d_hidden, bias=bias),
                self.ACTIVATIONS[activation](),
                self.dropout,
            ]
            for _ in range(self.n_layers - 2):
                layers += [
                    nn.Linear(d_hidden, d_hidden, bias=bias),
                    self.ACTIVATIONS[activation](),
                    self.dropout,
                ]
            layers.append(nn.Linear(d_hidden, out_dim, bias=bias))
            return nn.Sequential(*layers)

        self.mlps = nn.ModuleList([_build_mlp() for _ in range(num_variables)])'''
assert src.count(old_mlps) == 1
src = src.replace(old_mlps, new_mlps)

old_film = '''        # FiLM conditioner (shared across nodes; the context row is already
        # node-specific).  Zero-init last layer -> gamma=1, beta=0 at init.
        if self.film_context_dim > 0:
            film_hidden = max(32, 2 * d_hidden)
            self.film = nn.Sequential(
                nn.Linear(self.film_context_dim, film_hidden),
                nn.GELU(),
                nn.Linear(film_hidden, 2 * d_hidden),
            )'''
new_film = '''        # FiLM conditioner (shared across nodes; the context row is already
        # node-specific).  Outputs a (gamma_k, beta_k) pair for EVERY hidden
        # activation (multi-layer FiLM, Perez et al. 2018), packed as
        # [gamma_1..gamma_K | beta_1..beta_K].  Zero-init last layer ->
        # gamma=1, beta=0 at init.
        if self.film_context_dim > 0:
            film_hidden = max(32, 2 * d_hidden)
            self.film = nn.Sequential(
                nn.Linear(self.film_context_dim, film_hidden),
                nn.GELU(),
                nn.Linear(film_hidden, 2 * d_hidden * self.n_film_points),
            )'''
assert src.count(old_film) == 1
src = src.replace(old_film, new_film)

old_gb = '''            gb = self.film(context)
            gamma = 1.0 + gb[..., : self.d_hidden]
            beta = gb[..., self.d_hidden :]'''
new_gb = '''            gb = self.film(context)
            chunks = gb.chunk(2 * self.n_film_points, dim=-1)
            gammas = [1.0 + c for c in chunks[: self.n_film_points]]
            betas = list(chunks[self.n_film_points :])'''
assert src.count(old_gb) == 1
src = src.replace(old_gb, new_gb)

old_run = '''        if self.film is not None:
            # Linear -> Act -> Dropout, then FiLM on the hidden activation,
            # then the output projection.  Submodule indexing keeps the
            # legacy nn.Sequential checkpoint layout (mlps.i.0/3).
            all_outs = torch.stack(
                [mlp[3](gamma * mlp[2](mlp[1](mlp[0](x))) + beta)
                 for mlp in self.mlps],
                dim=2,
            )  # (B, L, num_variables, out_dim)'''
new_run = '''        if self.film is not None:
            # Per hidden block k: Linear -> Act -> Dropout, then FiLM
            # gamma_k * h + beta_k; the output projection is applied last.
            # Submodule indexing keeps the legacy nn.Sequential checkpoint
            # layout (mlps.i.0/3 when n_layers=2).
            def _run_film(mlp, x_in):
                h = x_in
                for k in range(self.n_film_points):
                    h = mlp[3 * k + 2](mlp[3 * k + 1](mlp[3 * k](h)))
                    h = gammas[k] * h + betas[k]
                return mlp[-1](h)

            all_outs = torch.stack(
                [_run_film(mlp, x) for mlp in self.mlps],
                dim=2,
            )  # (B, L, num_variables, out_dim)'''
assert src.count(old_run) == 1
src = src.replace(old_run, new_run)

p.write_text(src, encoding='utf-8')
print('mlp_head.py patched OK')
