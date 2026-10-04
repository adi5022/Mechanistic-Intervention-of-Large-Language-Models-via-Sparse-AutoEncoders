"""
Gradient-descent per-feature editing (branch gradient-descent-editing; Entries 28 to 30).

Instead of muting one pool of features at one strength and boosting another pool at another strength, every candidate
feature gets its OWN multiplier m_k = 1 + a_k (a_k < 0 mutes, a_k > 0 boosts, a_k = 0 leaves it alone, a_k in [-1, 2]).
The multipliers are tuned for THIS prompt by gradient descent on the real GPT-2 (blocks `layer`..11 run from the saved
residual state), starting from "no edit". This is the same procedure as `oracle_feature` in src/feature_models.py
(Entry 27), for one prompt, with the same loss:

    loss = rank hinge (target must beat the strongest other token by `margin`)
         + lam_kl   * KL(clean || edited) over all tokens except target and blocker, only once the target leads
         + lam_size * sum_k |a_k| * w_k          (w_k = candidate's share of the total max activation)

OPTIONAL ADDITIVE EDIT (`additive=True`, Entry 30). A multiplier cannot switch on a feature that is silent (0 * m = 0).
The additive knob adds an amount c_k >= 0 of a feature's decoder direction at the LAST position only, for features that are
exactly silent there:

    new volume = old volume * multiplier + added amount
    candidates : the top `add_top_m` silent features by a first-order score (the gradient of the target margin with respect
                 to the layer-`layer` residual at the last position, dotted with each feature's decoder direction), among
                 those with a positive score
    amount     : c_k = cap * (sigmoid(s_k) - sigmoid(ADD_S0)) for s_k >= ADD_S0 (else 0); s_k starts at ADD_S0, so every amount starts at exactly 0
                 (a tiny non-zero start is NOT harmless: the candidates are chosen because they help, so they add coherently);
                 cap = the loudest candidate activation on the real prompt positions (start-of-text token excluded)
    extra loss : lam_add * sum_k c_k / cap          (prefers few, small additions)

OPTIONAL `kl_always=True`: the KL term is charged on every step, not only once the target already leads (the default,
inherited from Entries 26/27, does not discourage a failing edit from distorting the output).

The best iterate is kept (a successful one with the lowest loss, otherwise the lowest loss). The reported result is then
re-checked through the real hook path (the delta-patch hook the sweep uses, plus the additive hook), so the number shown
does not depend on the training-time shortcut.
"""
import math

import torch
import torch.nn.functional as F

from src.editing import get_top_active_features
from src.hooks import make_scale_and_add_hook, make_scale_map_hook

BETA_MAX = 2.0
Z_ZERO = -math.log(BETA_MAX)          # sigmoid(Z_ZERO) = 1 / (1 + BETA_MAX)  ->  a = 0 (no edit)
ADD_S0 = -6.0                         # the additive amounts start at EXACTLY 0 (no edit): c = cap * max(0, sigmoid(s) - sigmoid(ADD_S0))
ADD_REPORT_FRAC = 0.01                # an addition is reported / counted when it is at least this share of the cap


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


def silent_candidates(model, sae, resid, tokens, layer, target_id, top_m):
    """Top `top_m` features that are exactly silent at the last position and whose first-order effect on the target margin
    is positive. Returns (feature ids [M], first-order scores [M]); M can be smaller than top_m."""
    r = resid.detach().clone().requires_grad_(True)
    last = _last_logits(model, r, tokens, layer)
    other = last.clone()
    other[target_id] = float("-inf")
    margin = last[target_id] - other.max()
    g = torch.autograd.grad(margin, r)[0][0, -1]                                # [d_model]
    with torch.no_grad():
        scores = sae.W_dec @ g                                                  # [n_features]
        silent = sae.encode(resid[0, -1:])[0] == 0
        scores = torch.where(silent & (scores > 0), scores, torch.full_like(scores, float("-inf")))
        vals, ids = torch.topk(scores, min(top_m, scores.numel()))
        keep = torch.isfinite(vals)
    return ids[keep], vals[keep]


def run_gradient_descent_edit(model, sae, clean_ctx, target_token_id, prompt, layer, hook_name, top_n=200,
                              positions="all", steps=100, lr=0.1, lam_kl=3.0, lam_size=0.005, margin_target=0.3,
                              on_step=None, additive=False, add_top_m=200, add_cap=None, add_cap_factor=1.0, lam_add=0.005, add_lr=0.3,
                              kl_always=False):
    """Tune one multiplier per candidate feature for this prompt (and, with additive=True, an added amount for silent
    features). Returns a dict with the multipliers, the additions, the per-step history, and the result re-checked through
    the real hook path."""
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
    groups = [{"params": [z], "lr": lr}]
    add_ids = add_wd = s = None
    cap = None
    if additive:
        add_ids, add_scores = silent_candidates(model, sae, resid, tokens, layer, target_token_id, add_top_m)
        act_real = acts[1:] if acts.shape[0] > 1 else acts                      # the loudest feature on the real prompt positions: the
        loudest = float(act_real.max().item())                                  # start-of-text token has huge activations and would inflate the cap
        cap = float(add_cap) if add_cap else float(add_cap_factor) * loudest
        if add_ids.numel() > 0:
            add_wd = sae.W_dec[add_ids].detach()                                # [M, d_model]
            s = torch.full((add_ids.numel(),), ADD_S0, device=device).requires_grad_(True)
            s0 = torch.full_like(s, ADD_S0).detach()                          # same dtype / op as s, so c is bit-exactly 0 at the start
            groups.append({"params": [s], "lr": add_lr})
    opt = torch.optim.Adam(groups)

    best_key, best_a, best_c, best_step = float("inf"), None, None, 0
    history = []
    for step in range(steps + 1):
        a = _to_a(z)
        delta = torch.einsum("pk,k,kd->pd", acts, a, wd).unsqueeze(0)
        c = None
        if s is not None:
            c = cap * (torch.sigmoid(s) - torch.sigmoid(s0)) * (s >= s0).to(s.dtype)      # >= keeps the gradient alive at the start
            add_full = torch.zeros_like(delta)
            add_full[0, -1] = c @ add_wd
            delta = delta + add_full
        last = _last_logits(model, resid + delta, tokens, layer)
        rank, prob, kl, margin = _score(last, clean_last, target_token_id, blocker_id)
        lead = 1.0 if kl_always else float(margin.item() >= margin_target)
        loss = F.relu(margin_target - margin) + lam_kl * kl * lead + lam_size * (a.abs() * w).sum()
        if c is not None:
            loss = loss + lam_add * c.sum() / cap
        key = (1e6 if rank > 1 else 0.0) + float(loss.item())
        history.append({"step": step, "rank": rank, "prob": float(prob.item()), "kl": float(kl.item()), "loss": float(loss.item())})
        if key < best_key:
            best_key, best_a, best_step = key, a.detach().clone(), step
            best_c = c.detach().clone() if c is not None else None
        if on_step is not None:
            on_step(step, steps, rank)
        if step == steps:
            break
        # autograd.grad w.r.t. the free numbers only: .backward() would also fill weight gradients for every GPT-2 parameter
        params = [z] + ([s] if s is not None else [])
        grads = torch.autograd.grad(loss, params)
        for p_, g_ in zip(params, grads):
            p_.grad = g_
        opt.step()

    # Re-check the chosen edit through the real hook path (full model, same delta-patch hook the sweep uses).
    scale_map = {fid: float(1.0 + ak) for fid, ak in zip(fids, best_a.tolist())}
    add_map = {}
    if best_c is not None:                      # ALL candidates' amounts go through the real hook (no threshold: many small ones add up)
        add_map = {int(i): float(v) for i, v in zip(add_ids.tolist(), best_c.tolist())}
    model.reset_hooks()
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_scale_and_add_hook(scale_map, add_map, sae))])
    model.reset_hooks()
    real_last = logits[0, -1]
    real_rank, real_prob, real_kl, _ = _score(real_last, clean_last, target_token_id, blocker_id)
    real_top1 = int(torch.argmax(real_last).item())

    with torch.no_grad():                       # edit size at the last position: ||change|| / ||residual|| (multiplier part + additive part)
        d_last = (best_a * acts[-1]) @ wd
        if best_c is not None:
            d_last = d_last + best_c @ add_wd
        edit_size = float(d_last.norm().item() / resid[0, -1].norm().item())

    out = {
        "fids": fids, "a": best_a.tolist(), "activation_max": amax.tolist(), "best_step": best_step, "history": history,
        "blocker_id": blocker_id, "n_candidates": K,
        "train_path": {"rank": history[best_step]["rank"], "prob": history[best_step]["prob"], "kl": history[best_step]["kl"]},
        "real_path": {"rank": real_rank, "prob": float(real_prob.item()), "kl": float(real_kl.item()), "top1_id": real_top1},
        "settings": {"top_n": top_n, "positions": positions, "steps": steps, "lr": lr, "lam_kl": lam_kl,
                     "lam_size": lam_size, "margin": margin_target, "additive": additive, "kl_always": kl_always},
        "added": add_map, "add_cap": cap, "edit_size_frac_norm": edit_size,
    }
    if additive:
        out["settings"].update({"add_top_m": add_top_m, "add_cap": cap, "add_cap_factor": add_cap_factor, "lam_add": lam_add, "add_lr": add_lr})
        out["add_candidates"] = int(add_ids.numel())
        amounts = list(add_map.values())
        out["add_summary"] = {                  # how concentrated the addition is (a few strong features, or many weak ones)
            "n_ge_1pct_cap": sum(1 for v in amounts if v >= ADD_REPORT_FRAC * cap),
            "n_ge_10pct_cap": sum(1 for v in amounts if v >= 0.10 * cap),
            "total_over_cap": sum(amounts) / cap if cap else 0.0,
            "max_over_cap": (max(amounts) / cap) if amounts and cap else 0.0,
        }
    return out


# ------------------------------------------------------------------------------------------------ generation (Prototype lab)
def _generation_hook(scale_map, add_map, sae, prompt_len, keep_on):
    """The tuned edit as a hook for a growing sequence. Multipliers scale their features at every position (keep_on=True) or
    only at the prompt's positions (keep_on=False). The additive amounts are added ONCE, at the prompt's last position
    (index prompt_len - 1: the position they were tuned for), on every forward pass, so the sequence is recomputed
    consistently without a key-value cache."""
    scale_hook = make_scale_map_hook(scale_map, sae)

    def hook_fn(resid, hook):
        out = scale_hook(resid, hook)
        if not keep_on and resid.shape[1] > prompt_len:
            out = torch.cat([out[:, :prompt_len], resid[:, prompt_len:]], dim=1)
        if add_map:
            out = out.clone()
            ids = torch.as_tensor(list(add_map.keys()), device=resid.device)
            amounts = torch.as_tensor(list(add_map.values()), device=resid.device, dtype=resid.dtype)
            out[:, prompt_len - 1, :] = out[:, prompt_len - 1, :] + amounts @ sae.W_dec[ids].to(resid.dtype)
        return out

    return hook_fn


def generate_greedy(model, sae, hook_name, prompt, n_tokens, scale_map=None, add_map=None, keep_on=True, target_id=None):
    """Greedy decoding (always the most likely next token) for `n_tokens` tokens, with or without the tuned edit.
    No key-value cache: each step recomputes the whole sequence, so the edit is applied exactly as described above.
    Returns (steps, continuation text); each step has the chosen token, its probability, the target's probability and rank."""
    toks = model.to_tokens(prompt)
    prompt_len = toks.shape[1]
    edited = bool(scale_map) or bool(add_map)
    steps = []
    for i in range(n_tokens):
        model.reset_hooks()
        with torch.no_grad():
            if edited:
                logits = model.run_with_hooks(toks, fwd_hooks=[(hook_name, _generation_hook(scale_map or {}, add_map or {}, sae, prompt_len, keep_on))])
            else:
                logits = model(toks)
        model.reset_hooks()
        probs = F.softmax(logits[0, -1], dim=-1)
        nxt = int(torch.argmax(probs).item())
        step = {"step": i + 1, "token_id": nxt, "token": model.to_string([nxt]), "prob": float(probs[nxt].item())}
        if target_id is not None:
            step["target_prob"] = float(probs[target_id].item())
            step["target_rank"] = int((probs > probs[target_id]).sum().item()) + 1
        steps.append(step)
        toks = torch.cat([toks, torch.tensor([[nxt]], device=toks.device)], dim=1)
        if nxt == model.tokenizer.eos_token_id:
            break
    return steps, model.to_string(toks[0, prompt_len:])
