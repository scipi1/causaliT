"""
Dataset loading for the external benchmarks.

The benchmarks (NOTEARS, DAGMA, PC) are plain structure learners: they expect
one ``(n_samples, n_nodes)`` design matrix and know nothing about causaliT's
token layout.  This module performs that single conversion, and it is the only
place where the on-disk format is interpreted.

**Node ordering is the critical contract.**  causaliT stores the SCM as two
token tensors, ``s`` (source variables) and ``x`` (intermediate variables), each
of shape ``(n_samples, seq_len, n_features)`` with feature 0 holding the value.
The benchmark matrix concatenates them **sources first**::

    columns 0 .. L_S-1        source variables      (S1, S2, ...)
    columns L_S .. L_S+L_X-1  intermediate variables (X1, X2, ...)

which is exactly the ordering assumed by the canonical DAG blocks: a square
``(N, N)`` adjacency in this ordering is what ``query_dag_blocks`` slices into
``cross`` (S->X) and ``self`` (X->X) via its homogeneous ``Rule 1``.  Any other
column order would silently permute the ground-truth comparison, so the loader
returns the labels alongside the matrix and the runner records them.

There is no S/X asymmetry at fit time: the estimator sees one SCM over ``N``
variables, just as it would in its own paper.  The split into blocks happens
only for evaluation, so causaliT's metrics remain comparable.
"""

from dataclasses import dataclass, field
from os.path import exists, join
from typing import List, Optional, Tuple

import numpy as np

#: Feature channel holding the variable's value in the causaliT token tensors.
VALUE_CHANNEL = 0

#: Split -> candidate npz filenames, in order of preference.  ``ds.npz`` is the
#: unsplit single-file layout produced by older datasets.
SPLIT_FILES = {
    "train": ("ds_train.npz", "ds.npz"),
    "test": ("ds_test.npz", "ds.npz"),
    "all": ("ds.npz", "ds_train.npz"),
}


@dataclass
class BenchmarkData:
    """
    One design matrix ready for a structure learner.

    Attributes:
        X: ``(n_samples, N)`` matrix, columns ordered ``[S..., X...]``.
        L_S: Number of source variables (leading columns).
        L_X: Number of intermediate variables (trailing columns).
        labels: Variable names, aligned with the columns of ``X``.
        split: Which split was loaded.
        source_file: The npz file the data came from.
        standardized: Whether columns were z-scored.
        n_dropped: Samples dropped because they contained non-finite values.
    """

    X: np.ndarray
    L_S: int
    L_X: int
    labels: List[str] = field(default_factory=list)
    split: str = "train"
    source_file: str = ""
    standardized: bool = False
    n_dropped: int = 0

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.X.shape[1])

    def summary(self) -> str:
        return (
            f"{self.split}: X{self.X.shape} (L_S={self.L_S}, L_X={self.L_X}) "
            f"from {self.source_file}"
            + (", standardized" if self.standardized else "")
            + (f", dropped {self.n_dropped} non-finite samples" if self.n_dropped else "")
        )


def _values(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Extract the value channel of a causaliT token tensor as ``(n_samples, L)``.

    Accepts ``(n, L, F)`` (standard) and ``(n, L)`` (already flat).
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, :, VALUE_CHANNEL].astype(float)
    if arr.ndim == 2:
        return arr.astype(float)
    raise ValueError(
        f"Unexpected token tensor with {arr.ndim} dimensions (shape {arr.shape}); "
        "expected (n_samples, seq_len, n_features) or (n_samples, seq_len)."
    )


def _resolve_split_file(dataset_dir: str, split: str) -> str:
    """Return the npz path for *split*, falling back to the single-file layout."""
    candidates = SPLIT_FILES.get(split)
    if candidates is None:
        raise ValueError(
            f"Unknown split '{split}'; expected one of {sorted(SPLIT_FILES)}."
        )
    for name in candidates:
        path = join(dataset_dir, name)
        if exists(path):
            return path
    raise FileNotFoundError(
        f"No data file for split '{split}' in {dataset_dir} "
        f"(looked for {', '.join(candidates)})."
    )


def load_benchmark_data(
    datadir_path: str,
    dataset_name: str,
    split: str = "train",
    standardize: bool = True,
    max_samples: Optional[int] = None,
    metadata: Optional[dict] = None,
    seed: int = 0,
) -> BenchmarkData:
    """
    Load one dataset as a single design matrix with canonical column ordering.

    Args:
        datadir_path: Data root containing ``<dataset_name>/``.
        dataset_name: Dataset folder name.
        split: ``train`` (default), ``test`` or ``all``.  Benchmarks are fitted
            on the training split so they see exactly the data the models were
            trained on.
        standardize: Z-score each column.  Both NOTEARS and DAGMA assume
            comparable scales across variables (their L1 penalty is shared), and
            the papers standardize their simulated data, so this defaults to
            True.  Note that standardizing cannot change the graph: it is a
            per-node affine rescaling.
        max_samples: Optional cap, applied by taking the first ``max_samples``
            rows (deterministic; the rows are already i.i.d.).
        metadata: Dataset metadata dict; used only for variable labels.
        seed: Unused placeholder kept for signature stability.

    Returns:
        A :class:`BenchmarkData` instance.

    Raises:
        FileNotFoundError: no npz for the requested split.
        ValueError: the file lacks the ``x`` (intermediate) tensor.
    """
    dataset_dir = join(datadir_path, dataset_name)
    path = _resolve_split_file(dataset_dir, split)

    with np.load(path, allow_pickle=False) as handle:
        keys = set(handle.files)
        if "x" not in keys:
            raise ValueError(
                f"{path} does not contain an 'x' array (found {sorted(keys)})."
            )
        x_vals = _values(handle["x"])
        s_vals = _values(handle["s"]) if "s" in keys else None

    if x_vals is None:  # defensive; 'x' presence was checked above
        raise ValueError(f"Could not read intermediate variables from {path}.")

    # --- Canonical ordering: sources first, then intermediates -----------
    if s_vals is not None:
        L_S = int(s_vals.shape[1])
        matrix = np.concatenate([s_vals, x_vals], axis=1)
    else:
        L_S = 0
        matrix = x_vals
    L_X = int(x_vals.shape[1])

    # --- Labels ----------------------------------------------------------
    var_info = (metadata or {}).get("variable_info", {}) or {}
    src_labels = list(var_info.get("source_labels") or [])
    inp_labels = list(var_info.get("input_labels") or [])
    if len(src_labels) != L_S:
        src_labels = [f"S{i + 1}" for i in range(L_S)]
    if len(inp_labels) != L_X:
        inp_labels = [f"X{i + 1}" for i in range(L_X)]
    labels = src_labels + inp_labels

    # --- Drop non-finite rows (a single NaN breaks the L-BFGS-B solvers) --
    finite_rows = np.isfinite(matrix).all(axis=1)
    n_dropped = int((~finite_rows).sum())
    if n_dropped:
        matrix = matrix[finite_rows]

    if max_samples is not None and matrix.shape[0] > int(max_samples):
        matrix = matrix[: int(max_samples)]

    standardized = False
    if standardize:
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        # Constant columns (std == 0) stay centred at zero instead of becoming
        # NaN; the benchmarks then simply find no edges for them.
        std = np.where(std > 1e-12, std, 1.0)
        matrix = (matrix - mean) / std
        standardized = True

    return BenchmarkData(
        X=np.ascontiguousarray(matrix, dtype=float),
        L_S=L_S,
        L_X=L_X,
        labels=labels,
        split=split,
        source_file=path,
        standardized=standardized,
        n_dropped=n_dropped,
    )


def train_val_split(
    X: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random row split, used only for the optional held-out score.

    The benchmarks are fitted with fixed paper hyperparameters, so no model
    selection happens here; the held-out matrix serves purely as a reported
    diagnostic (out-of-sample fit of the learned structure).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n))) if n > 1 else 0
    return X[idx[n_val:]], X[idx[:n_val]]


__all__ = ["BenchmarkData", "load_benchmark_data", "train_val_split", "VALUE_CHANNEL"]
