"""Generate the node-wise z-scored sibling of random_n20_k4_er_nonlinear_gaussian_s1.

Same RandomSCMConfig (same seed -> identical graph and raw draws); only the
normalization differs: "standardize_node" (per-node z-score) instead of the
legacy global "minmax".  The name suffix is required because the folder name
does not encode normalize_method.
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
    name="random_n20_k4_er_nonlinear_gaussian_s1_nzscore",
)

name = ensure_dag_dataset(
    cfg,
    data_root=Path("data"),
    gen_kwargs={
        "n_samples": 5000,
        "normalize_method": "standardize_node",
        "mode": "flat",
        "compute_ate": False,
    },
)
print("dataset:", name)
