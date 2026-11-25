"""
Helpers of ``adaptive_trainer`` (phase schedule and pre-flight stages).

    stage_policy          adaptive_training stage-policy schema / validation
                          (the forecaster reads ``resolve_gate_opts``)
    staging_utils         plain-container, score margin, cross-fit split, JSON
    dropout_selection     pre-flight regressor-dropout selection
    order_initialization  one-off residual-HSIC order initialization

Coding Assistant: new PhaseController features that need their own module go
here; keep ``adaptive_trainer.py`` the orchestrator.
"""
