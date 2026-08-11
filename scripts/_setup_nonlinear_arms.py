"""Generate the investigation configs from the published svfa template, into
experiments/6_INVESTIGATIONS/NONLINEARITIES/.

All arms share the published svfa config_atsel.yaml; only the local data
pointer and one treatment per arm differ, so any difference in the X4 readout
is attributable to the treatment:
  - baseline:     template as-is
  - larger_mlp:   deeper + wider shared output head (capacity hypothesis)
  - nl_value_emb: shared MLP value embedding instead of the linear map (gate
                  hypothesis: the linear value map cannot represent the
                  nonlinear contribution, so the gate shuts the edge)

Usage:  python scripts/_setup_nonlinear_arms.py
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "experiments/7_PUBLISH/ATE/results/svfa_9425659/config_atsel.yaml"
OUT = ROOT / "experiments/6_INVESTIGATIONS/NONLINEARITIES"

# Local data pointer: the local scm2_continuous is pre-split into
# ds_train.npz / ds_test.npz (no single ds.npz).
DATA_EDITS = [
    ("  dataset: ds_scm1\n", "  dataset: scm2_continuous\n"),
    ("  train_file: null\n", "  train_file: ds_train.npz\n"),
    ("  test_file: null\n", "  test_file: ds_test.npz\n"),
]

# larger_mlp: deepen + widen the shared output head (the capacity hypothesis).
MLP_EDITS = [
    ("  output_mlp_layers : 1\n", "  output_mlp_layers : 3\n"),
    ("    output_mlp_hidden: ${experiment.d_ff}\n", "    output_mlp_hidden: 128\n"),
]

# nl_value_emb: shared MLP value embedding (embed: mlp; defaults hidden 64,
# gelu) instead of the linear map, in BOTH ds_embed_S and ds_embed_X.
NL_VALUE_BLOCK_OLD = (
    "        - idx: ${data.feature_indices.value}\n"
    "          embed: linear\n"
    "          label: value\n"
    "          role: value\n"
    "          kwargs:\n"
    "            input_dim: 1\n"
    "            embedding_dim: ${model.embed_dim.val_emb_hidden}\n"
)
NL_VALUE_BLOCK_NEW = NL_VALUE_BLOCK_OLD.replace("embed: linear", "embed: mlp")


def apply_edits(text: str, edits) -> str:
    for old, new in edits:
        assert old in text, f"template line not found: {old!r}"
        text = text.replace(old, new, 1)
    return text


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    baseline = apply_edits(template, DATA_EDITS)
    larger = apply_edits(template, DATA_EDITS + MLP_EDITS)

    nl_value = apply_edits(template, DATA_EDITS)
    assert nl_value.count(NL_VALUE_BLOCK_OLD) == 2, (
        "expected exactly 2 value-embedding blocks (ds_embed_S, ds_embed_X)"
    )
    nl_value = nl_value.replace(NL_VALUE_BLOCK_OLD, NL_VALUE_BLOCK_NEW)

    arms = [("baseline", baseline), ("larger_mlp", larger), ("nl_value_emb", nl_value)]
    for name, text in arms:
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config_atsel.yaml").write_text(text, encoding="utf-8")
        print(f"  wrote {d / 'config_atsel.yaml'}")

    print("Done. Launch with:")
    for name, _ in arms:
        print(f"  python -m causaliT.cli adaptivetrain --exp_id 6_INVESTIGATIONS/NONLINEARITIES/{name}")


if __name__ == "__main__":
    main()
