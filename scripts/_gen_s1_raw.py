"""Generate the RAW (unnormalized) sibling of random_n20_k4_er_nonlinear_gaussian_s1.

Same RandomSCMConfig (same seed -> identical graph and raw draws) as the
``minmax`` and ``standardize_node`` (nzscore) variants; only the normalization
differs: ``normalize=False`` keeps the raw sampled values, preserving the
marginal-scale profile (the "causal stairs"; Reisach et al. 2021) that
node-wise z-scoring removes.  The name suffix is required because the folder
name does not encode the normalization choice.
"""

from pathlib import Path

from scm_ds.random_scm import RandomSCMConfig
from causaliT.euler_sweep.euler_sweep.dag_provider import ensure_dag_dataset

cfg = RandomSCMConfig(
    n_nodes=20,
    degree=4,
    linearity="nonlinear",
    noise="gaussian",
    noise_scale=0.1,
    nonlinear_fns=["sin", "tanh"],
    permute_labels=True,
    rescale_by_indegree=True,
    seed=1,
    source_noise="uniform",
    weight_range=[0.5, 2.0],
    name="random_n20_k4_er_nonlinear_gaussian_s1_raw",
)

name = ensure_dag_dataset(
    cfg,
    data_root=Path("data"),
    gen_kwargs={
        "n_samples": 5000,
        "normalize": False,
        "mode": "flat",
        "compute_ate": False,
    },
)
print("dataset:", name)
