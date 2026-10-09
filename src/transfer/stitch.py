"""
Helpers for step A2: read states at one layer, translate them, and run a model with its own residual stream replaced.

Stitching: run the receiver, but at layer m replace its residual stream (positions 1 onward; position 0, the start token, stays real)
with a translated state, and measure next-word loss. Three reference points:
    clean  = the receiver's own loss, unchanged                           (best possible)
    floor  = states replaced by the receiver's average state              (all text information destroyed)
    wrong  = states replaced by the translation of a DIFFERENT sequence   (right kind of state, wrong text)
    loss recovered = (floor - stitched) / (floor - clean):   0 = as bad as destroying the information, 1 = as good as the real state.
"""
import torch
import torch.nn.functional as F


def states(model, tokens, layer):
    """blocks.<layer>.hook_resid_pre at every position: [B, L, d_model]."""
    name = f"blocks.{layer}.hook_resid_pre"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == name, stop_at_layer=layer + 1)
    return cache[name]


def translate(h, W, mu_src, mu_dst):
    return (h - mu_src) @ W + mu_dst


def _loss_sum(logits, tokens):
    """Summed cross-entropy of predicting token t+1 from position t, for positions 1..L-2 (the start-token position is skipped)."""
    lg = logits[:, 1:-1].float()
    tg = tokens[:, 2:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="sum").item(), tg.numel()


def loss_with_replacement(model, tokens, layer, replacement=None):
    """(summed loss, number of predictions). replacement=None = the unchanged model."""
    name = f"blocks.{layer}.hook_resid_pre"

    def hook(resid, hook):
        out = resid.clone()
        out[:, 1:] = replacement[:, 1:].to(out.dtype)
        return out

    with torch.no_grad():
        logits = model(tokens) if replacement is None else model.run_with_hooks(tokens, fwd_hooks=[(name, hook)])
    return _loss_sum(logits, tokens)


def stitching(src_model, dst_model, ids, src_layer, dst_layer, W, mu_src, mu_dst, batch=16, clean=None):
    """Mean next-word loss of the receiver under: clean, translated state, average state (floor), translated state of the wrong sequence.
    `clean` can be passed in (it depends only on the receiver). Returns a dict of mean losses and the share of loss recovered."""
    acc = {"clean": [0.0, 0], "stitched": [0.0, 0], "floor": [0.0, 0], "wrong": [0.0, 0]}

    def add(key, r):
        acc[key][0] += r[0]; acc[key][1] += r[1]

    for i in range(0, len(ids), batch):
        b = ids[i:i + batch].to(next(dst_model.parameters()).device)
        repl = translate(states(src_model, b, src_layer), W, mu_src, mu_dst)
        if clean is None:
            add("clean", loss_with_replacement(dst_model, b, dst_layer, None))
        add("stitched", loss_with_replacement(dst_model, b, dst_layer, repl))
        add("floor", loss_with_replacement(dst_model, b, dst_layer, mu_dst.expand_as(repl)))
        if len(b) > 1:
            add("wrong", loss_with_replacement(dst_model, b, dst_layer, torch.roll(repl, 1, dims=0)))
        del repl
    out = {k: v[0] / v[1] for k, v in acc.items() if v[1] > 0}
    if clean is not None:
        out["clean"] = clean
    out["recovered"] = (out["floor"] - out["stitched"]) / max(out["floor"] - out["clean"], 1e-9)
    return out
