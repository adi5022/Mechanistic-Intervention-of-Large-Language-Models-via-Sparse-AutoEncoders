"""
Step D5 building blocks: add the translated change at SEVERAL layers of the receiver at once.

The change made by the edit in small (layer 8) is translated once per receiver layer with that layer's own map (small 8 -> medium q) and added
to blocks.q.hook_resid_pre for every q in the chosen layer set. In a "split" configuration each of the n layers gets dose/n of its own
translated change, so the total budget equals a single-layer dose; "full" gives every layer the whole dose (a bigger push, exploratory only).
"""
import torch

SINGLE_LAYERS = [4, 8, 12, 16, 20]
LAYER_SETS = [(12, 16), (8, 12, 16), (12, 16, 20), (8, 12, 16, 20), (4, 8, 12, 16, 20)]


def make_configs():
    """[(name, layers, mode)] with mode 'single', 'split' or 'full'."""
    cfgs = [(f"L{q}", (q,), "single") for q in SINGLE_LAYERS]
    cfgs += [("+".join(map(str, ls)) + " split", ls, "split") for ls in LAYER_SETS]
    cfgs += [("+".join(map(str, ls)) + " full", ls, "full") for ls in LAYER_SETS]
    return cfgs


def layer_share(layers, mode):
    """Multiplier applied to the dose at each layer."""
    return 1.0 / len(layers) if mode == "split" else 1.0


def run_injected_multi(model, tokens, deltas, chunk=64):
    """Last-position logits [P, V] when deltas[q] [P, L, d] is added to blocks.q.hook_resid_pre for every layer q in `deltas`.
    Only the last position is unembedded (same numbers as the last position of a full forward pass)."""
    out = []
    for i in range(0, len(tokens), chunk):
        grab = {}

        def keep_last(x, hook):
            grab["h"] = x[:, -1:].clone()

        hooks = []
        for q, d in deltas.items():
            hooks.append((f"blocks.{q}.hook_resid_pre", lambda resid, hook, d=d[i:i + chunk]: resid + d))
        hooks.append(("ln_final.hook_normalized", keep_last))
        with torch.no_grad():
            model.run_with_hooks(tokens[i:i + chunk], return_type=None, fwd_hooks=hooks)
            out.append(model.unembed(grab["h"])[:, 0].float())
    return torch.cat(out)
