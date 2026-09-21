from pathlib import Path

# ---- Patch 1: adaptive_trainer.py - BKD ladder + constraint reset hook -------
p = Path("causaliT/training/adaptive_trainer.py")
src = p.read_text(encoding="utf-8")

old = '''        self._bkd_managed: bool = any(
            "batch_key_dropout" in cfg
            for cfg in (self.recon_cfg, self.struct_cfg, self.final_cfg,
                        self.warmup_cfg)
        )'''
new = old + '''

        # BKD ladder (constant within a cycle, discrete decrease per cycle).
        # When set, rung k = ladder[min(cycle, len-1)] is applied IDENTICALLY
        # to the reconstruct AND structure phases of cycle k (warmup = rung 0),
        # overriding any static per-phase batch_key_dropout.  Rationale: the
        # frozen-regressor structure phase must test independence under the
        # SAME key-availability regime the regressor was just trained on.
        lad = ad.get("bkd_ladder", None)
        self._bkd_ladder: Optional[list] = (
            [float(v) for v in lad] if lad is not None else None
        )
        if self._bkd_ladder is not None:
            if not self._bkd_ladder or any(p < 0.0 or p > 1.0 for p in self._bkd_ladder):
                raise ValueError(
                    f"adaptive_training.bkd_ladder must be a non-empty list of "
                    f"probabilities in [0, 1], got {self._bkd_ladder}"
                )
            self._bkd_managed = True'''
assert src.count(old) == 1, "ladder config anchor"
src = src.replace(old, new)

old = '''        if phase == "warmup":
            cfg = self.warmup_cfg
        elif phase == "reconstruct":
            cfg = self.recon_cfg
        elif phase == "final_reconstruct":
            cfg = {**self.recon_cfg, **self.final_cfg}
        else:
            cfg = self.struct_cfg
        active = "batch_key_dropout" in cfg'''
new = '''        if phase == "warmup":
            cfg = self.warmup_cfg
        elif phase == "reconstruct":
            cfg = self.recon_cfg
        elif phase == "final_reconstruct":
            cfg = {**self.recon_cfg, **self.final_cfg}
        else:
            cfg = self.struct_cfg
        if self._bkd_ladder is not None:
            # Constant p for the whole cycle (recon + structure alike);
            # _cycle_count counts COMPLETED structure phases, so both phases
            # of cycle k see rung k.
            p_rung = self._bkd_ladder[min(self._cycle_count,
                                          len(self._bkd_ladder) - 1)]
            cfg = {**cfg,
                   "batch_key_dropout": p_rung,
                   "batch_key_dropout_final": p_rung,
                   "batch_key_dropout_annealing_batches": None}
        active = "batch_key_dropout" in cfg'''
assert src.count(old) == 1, "ladder apply anchor"
src = src.replace(old, new)

old = '''        if getattr(pl_module, "nodewise_reset_every_stage", False):
            pl_module.nodewise_reset_stats()'''
new = old + '''

        # HSIC constraint: the violation regime shifts at every phase switch
        # (new cross-fit fold, new BKD rung), so the EMA and the rho-escalation
        # memory must not compare across regimes.  The dual variables
        # themselves persist -- they accumulate evidence over the whole run.
        if getattr(pl_module, "hsic_constraint_enabled", False):
            pl_module.hsic_constraint_on_phase_switch()'''
assert src.count(old) == 1, "reset hook anchor"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("adaptive_trainer patched")

# ---- Patch 2: forecaster - phase-switch reset method --------------------------
p = Path("causaliT/training/forecasters/attention_selector_forecaster.py")
src = p.read_text(encoding="utf-8")

old = '''        self._hsic_constraint_prev_violation = violation
        self.log("hsic/dual_lambda", self._hsic_dual_lambda,'''
new = '''        self._hsic_constraint_prev_violation = violation
        self.log("hsic/dual_lambda", self._hsic_dual_lambda,'''
assert src.count(old) == 1, "method anchor"
method = '''
    def hsic_constraint_on_phase_switch(self) -> None:
        """Reset per-regime constraint memory at an adaptive phase boundary.

        Called by the adaptive trainer at every phase switch: the BKD rung and
        the cross-fit fold change with the phase, so the HSIC EMA and the
        previous-violation memory are not comparable across the boundary.
        The dual variables (lambda, rho) deliberately PERSIST -- they
        accumulate constraint evidence over the whole run.
        """
        self._hsic_constraint_ema = None
        self._hsic_constraint_prev_violation = None

    def _update_hsic_dual(self) -> None:
        """Per-epoch dual ascent for the HSIC constraint (Lagrangian mode).'''
old2 = '''
    def _update_hsic_dual(self) -> None:
        """Per-epoch dual ascent for the HSIC constraint (Lagrangian mode).'''
assert src.count(old2) == 1, "method anchor 2"
src = src.replace(old2, method)
p.write_text(src, encoding="utf-8")
print("forecaster patched")
