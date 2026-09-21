"""One-off patch: fix gradient assertion in test_film_forward_and_gradients."""
import io

p = "tests/test_atsel_adjacency_context.py"
s = io.open(p, encoding="utf-8").read()

old = (
    "    loss = pred_x.square().mean()\n"
    "    loss.backward()\n"
    "    g = model.forecaster.film[0].weight.grad\n"
    "    assert g is not None and g.abs().sum() > 0, \"no gradient into conditioner\"\n"
)
new = (
    "    loss = pred_x.square().mean()\n"
    "    loss.backward()\n"
    "    # The conditioner's LAST (zero-init) layer receives gradient immediately\n"
    "    # (its grad is delta x hidden, independent of its own zero weights);\n"
    "    # the first layer's grad is exactly 0 at identity init by construction.\n"
    "    g = model.forecaster.film[-1].weight.grad\n"
    "    assert g is not None and g.abs().sum() > 0, \"no gradient into conditioner\"\n"
)
assert s.count(old) == 1, "anchor not unique"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched test gradient assertion")
