"""
Training infrastructure shared by ``trainer`` and ``adaptive_trainer``.

    config_utils        dataset metadata -> config (sequence lengths, ...)
    experiment_control  config loading, update_config, combination sweeps
    optimizer_factory   optimizers / schedulers from config
    gradient_routing    structural vs reconstruction parameter split
    training_summary    training_summary.json writer / reader

Coding Assistant: put helpers here only if BOTH trainers (or the forecaster
and a trainer) need them; trainer-specific code goes to adaptive_utils/ or
forecaster_utils/.
"""
