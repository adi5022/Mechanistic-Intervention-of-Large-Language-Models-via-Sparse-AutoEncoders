"""
Gradient-descent per-feature editing for the Hybrid tab (branch gradient-descent-editing).

Instead of muting one pool of features at one strength and boosting another pool at another strength, every candidate
feature gets its OWN multiplier m_k = 1 + a_k (a_k < 0 mutes, a_k > 0 boosts, a_k = 0 leaves it alone, a_k in [-1, 2]).
The multipliers are tuned for THIS prompt by gradient descent on the real GPT-2 (blocks `layer`..11 run from the saved
residual state), starting from "no edit". This is the same procedure as `oracle_feature` in src/feature_models.py
(Entry 27), for one prompt, with the same loss:

    loss = rank hinge (target must beat the strongest other token by `margin`)
         + lam_kl   * KL(clean || edited) over all tokens except target and blocker, only once the target leads
         + lam_size * sum_k |a_k| * w_k          (w_k = candidate's share of the total max activation)

The best iterate is kept (a successful one with the lowest loss, otherwise the lowest loss). The reported result is then
re-checked through the real hook path (the same delta-patch hook the sweep uses), so the number shown does not depend on
the training-time shortcut.
"""
import math

import torch
import torch.nn.functional as F

from src.editing import get_top_active_features
from src.hooks import make_scale_map_hook

BETA_MAX = 2.0
Z_ZERO = -math.log(BETA_MAX)          # sigmoid(Z_ZERO) = 1 / (1 + BETA_MAX)  ->  a = 0 (no edit)


def _to_a(z):
    return -1.0 + (1.0 + BETA_MAX) * torch.sigmoid(z)


def _last_logits(model, resid, tokens, layer):
    """Logits at the last prompt position, running only blocks `layer`..11 from the (edited) residual state."""
    res = model(resid, start_at_layer=layer, stop_at_layer=model.cfg.n_layers, tokens=tokens)
    x = model.ln_final(res[:, -1:, :])[:, 0]
    return (x @ model.W_U + model.b_U)[0]


def _score(last, clean_last, target_id, blocker_id):
    """rank, target prob, KL(clean || edited) over all tokens except target and blocker (renormalised), margin."""
    lp = F.log_softmax(last, dim=-1)
    tl = lp[target_id]
    rank = int((lp > tl).sum().item()) + 1
    mask = torch.zeros_like(last, dtype=torch.bool)
    mask[target_id] = True
    mask[blocker_id] = True
    lq = F.log_softmax(clean_last.masked_fill(mask, float("-inf")), dim=-1)
    lpe = F.log_softmax(last.masked_fill(mask, float("-inf")), dim=-1)
    kl = torch.where(mask, torch.zeros_like(lq), lq.exp() * (lq - lpe)).sum()
    other = last.clone()
    other[target_id] = float("-inf")
    margin = last[target_id] - other.max()
    return rank, tl.exp(), kl, margin


def run_gradient_descent_edit(model, sae, clean_ctx, target_token_id, prompt, layer, hook_name, top_n=200,
                              positions="all", steps=100, lr=0.1, lam_kl=3.0, lam_size=0.005, margin_target=0.3,
                              on_step=None):
    """Tune one multiplier per candidate feature for this prompt. Returns a dict with the multipliers, the per-step
    history, and the result re-checked through the real hook path."""
    device = clean_ctx.resid_all.device
    cands = get_top_active_features(model, sae, prompt, top_n=top_n, clean_ctx=clean_ctx, positions=positions)
    fids = [f for f, _ in cands]
    K = len(fids)
    if K == 0:
        raise ValueError("No active candidate features for this prompt and candidate source.")

    resid = clean_ctx.resid_all.detach()                                       # [1, seq, d_model]
    tokens = clean_ctx.tokens
    blocker_id = int(torch.argmax(clean_ctx.clean_probs).item())
    with torch.no_grad():
        acts = sae.encode(resid[0])[:, fids]                                    # [seq, K] candidate activations, all positions
        wd = sae.W_dec[fids].detach()                                           # [K, d_model]
        amax = acts.max(dim=0).values                                           # [K]
        w = amax / amax.sum().clamp_min(1e-9)
        clean_last = _last_logits(model, resid, tokens, layer)

    z = torch.full((K,), Z_ZERO, device=device).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    best_key, best_a, best_step = float("inf"), None, 0
    history = []
    for step in range(steps + 1):
        a = _to_a(z)
        delta = torch.einsum("pk,k,kd->pd", acts, a, wd).unsqueeze(0)
        last = _last_logits(model, resid + delta, tokens, layer)
        rank, prob, kl, margin = _score(last, clean_last, target_token_id, blocker_id)
        lead = float(margin.item() >= margin_target)
        loss = F.relu(margin_target - margin) + lam_kl * kl * lead + lam_size * (a.abs() * w).sum()
        key = (1e6 if rank > 1 else 0.0) + float(loss.item())
        history.append({"step": step, "rank": rank, "prob": float(prob.item()), "kl": float(kl.item()), "loss": float(loss.item())})
        if key < best_key:
            best_key, best_a, best_step = key, a.detach().clone(), step
        if on_step is not None:
            on_step(step, steps, rank)
        if step == steps:
            break
        # autograd.grad w.r.t. z only: .backward() would also fill weight gradients for every GPT-2 parameter
        z.grad, = torch.autograd.grad(loss, z)
        opt.step()

    # Re-check the chosen multipliers through the real hook path (full model, same hook the sweep uses).
    scale_map = {fid: float(1.0 + ak) for fid, ak in zip(fids, best_a.tolist())}
    model.reset_hooks()
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_scale_map_hook(scale_map, sae))])
    model.reset_hooks()
    real_last = logits[0, -1]
    real_rank, real_prob, real_kl, _ = _score(real_last, clean_last, target_token_id, blocker_id)
    real_top1 = int(torch.argmax(real_last).item())

    return {
        "fids": fids, "a": best_a.tolist(), "activation_max": amax.tolist(), "best_step": best_step, "history": history,
        "blocker_id": blocker_id, "n_candidates": K,
        "train_path": {"rank": history[best_step]["rank"], "prob": history[best_step]["prob"], "kl": history[best_step]["kl"]},
        "real_path": {"rank": real_rank, "prob": float(real_prob.item()), "kl": float(real_kl.item()), "top1_id": real_top1},
        "settings": {"top_n": top_n, "positions": positions, "steps": steps, "lr": lr, "lam_kl": lam_kl,
                     "lam_size": lam_size, "margin": margin_target},
    }
