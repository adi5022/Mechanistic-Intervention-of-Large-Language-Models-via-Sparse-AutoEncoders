"""
Headless diagnostics for two questions about the SAE edit:

  (1) Do later layers "repair" the edit?  -> persistence trace, logit lens, and a "held edit" control.
  (2) Is the SAE the limiting factor?     -> compare the SAE edit with an SAE-free, norm-matched optimal
                                             residual edit at the same layer (an upper bound for that size of push).

Everything here works on one prompt/target and returns a JSON-serialisable dict, so it can be run from a
script or from the Streamlit tab.
"""
import time

import torch
import torch.nn.functional as F

from src.editing import get_target_token_id
from src.hooks import build_scale_map, build_scale_map_graded

N_LAYERS = 12
OPT_FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.4]   # push size as a fraction of the residual norm


def _rank_prob(logits_last, target_id):
    probs = F.softmax(logits_last, dim=-1)
    rank = int((probs > probs[target_id]).sum().item()) + 1
    return rank, float(probs[target_id].item())


def _lens(model, resid_last, target_id):
    """Logit lens at one residual vector: target rank, target prob, margin over the best other token."""
    with torch.no_grad():
        x = model.ln_final(resid_last.reshape(1, 1, -1))[0, 0]
        logits = x @ model.W_U + model.b_U
    rank, prob = _rank_prob(logits, target_id)
    other = logits.clone()
    other[target_id] = -1e9
    return rank, prob, float(logits[target_id].item() - other.max().item())


def _clean_cache(model, tokens):
    store = {}
    def keep(name):
        return name.endswith("hook_resid_pre") or name == f"blocks.{N_LAYERS - 1}.hook_resid_post"
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=keep)
    for name in cache.keys():
        store[name] = cache[name].detach()
    return logits[0, -1], store


def sae_delta(sae, resid, scale_map):
    """Residual-space change made by the SAE delta patch (exactly what the hook adds)."""
    with torch.no_grad():
        acts = sae.encode(resid)
        base = sae.decode(acts)
        mod = acts.clone()
        if scale_map:
            idx = torch.as_tensor(list(scale_map.keys()), device=resid.device)
            sc = torch.as_tensor(list(scale_map.values()), device=resid.device, dtype=mod.dtype)
            mod[..., idx] = mod[..., idx] * sc
        return sae.decode(mod) - base, base


def scale_map_from_record(rec):
    """Scale map of the best edit of a hybrid run (falls back to everything applied)."""
    best = rec.get("best_result") or {}
    if best.get("step", 0) > 0 and (best.get("mute_strengths") or best.get("boost_strengths")):
        return build_scale_map_graded(best.get("mute_strengths", {}), best.get("boost_strengths", {}))
    return build_scale_map(rec.get("applied_mutes_final", []), rec["mute_strength"],
                           rec.get("applied_boosts_final", []), rec["boost_strength"])


def _run_with_delta(model, hook_name, delta, extra_hooks=None):
    def add(resid, hook):
        return resid + delta
    hooks = [(hook_name, add)] + (extra_hooks or [])
    return hooks


def trace_edit(model, tokens, layer, delta, target_id, clean_logits, clean_store):
    """Follow one residual delta injected at `layer` through the remaining blocks."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, lambda r, hook: r + delta)]):
            logits, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n.endswith("hook_resid_pre") or n == f"blocks.{N_LAYERS - 1}.hook_resid_post")
    d_flat = delta.reshape(-1)
    dd = float((d_flat * d_flat).sum().item()) or 1e-12
    dn = dd ** 0.5
    rows = []
    points = [(l, f"blocks.{l}.hook_resid_pre", f"L{l}") for l in range(N_LAYERS)]
    points.append((N_LAYERS, f"blocks.{N_LAYERS - 1}.hook_resid_post", "final"))
    for l, name, label in points:
        clean = clean_store[name]
        edited = clean + delta if (l == layer) else (cache[name] if l > layer else clean)
        diff = (edited - clean).reshape(-1)
        proj = float((diff * d_flat).sum().item()) / dd
        cos = float((diff * d_flat).sum().item()) / ((float((diff * diff).sum().item()) ** 0.5 * dn) or 1e-12)
        rk_c, pr_c, mg_c = _lens(model, clean[0, -1], target_id)
        rk_e, pr_e, mg_e = _lens(model, edited[0, -1], target_id)
        rows.append({"point": label, "layer": l, "downstream": l >= layer,
                     "delta_norm": float(diff.norm().item()), "persistence": proj if l >= layer else None,
                     "cosine": cos if (l >= layer and diff.norm() > 0) else None,
                     "lens_rank_clean": rk_c, "lens_rank_edit": rk_e,
                     "lens_prob_clean": pr_c, "lens_prob_edit": pr_e,
                     "margin_clean": mg_c, "margin_edit": mg_e})
    rank, prob = _rank_prob(logits[0, -1], target_id)
    return {"rows": rows, "rank": rank, "prob": prob}


def held_edit(model, tokens, layer, delta, target_id, clean_store):
    """
    Control for repair: at every later block, add back whatever fraction of the injected direction has been
    lost, so the edit stays at full strength all the way to the output. If this beats the plain edit, later
    layers were undoing part of it.
    """
    d_flat = delta.reshape(-1)
    dd = float((d_flat * d_flat).sum().item()) or 1e-12

    def make_hook(l):
        clean = clean_store[f"blocks.{l}.hook_resid_pre"]
        def hook_fn(resid, hook):
            diff = (resid - clean).reshape(-1)
            lost = 1.0 - float((diff * d_flat).sum().item()) / dd
            return resid + max(lost, 0.0) * delta
        return hook_fn

    hooks = [(f"blocks.{layer}.hook_resid_pre", lambda r, hook: r + delta)]
    hooks += [(f"blocks.{l}.hook_resid_pre", make_hook(l)) for l in range(layer + 1, N_LAYERS)]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    rank, prob = _rank_prob(logits[0, -1], target_id)
    return {"rank": rank, "prob": prob}


def optimal_delta(model, tokens, layer, target_id, budget, positions="last", steps=60):
    """
    SAE-free upper bound: gradient-optimise a residual push of Frobenius norm <= budget that raises the
    target, at `layer`, on the last token only or on every non-BOS position.
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    n_pos = tokens.shape[1]
    d_model = model.cfg.d_model
    mask = torch.zeros(1, n_pos, 1, device=tokens.device)
    if positions == "last":
        mask[:, -1] = 1.0
    else:
        mask[:, 1:] = 1.0
    delta = torch.zeros(1, n_pos, d_model, device=tokens.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=max(budget * 0.15, 1e-4))
    best = None
    for _ in range(steps):
        opt.zero_grad()
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, lambda r, hook: r + delta * mask)])
        logp = F.log_softmax(logits[0, -1], dim=-1)
        rank = int((logp > logp[target_id]).sum().item()) + 1
        prob = float(logp[target_id].exp().item())
        if best is None or (rank, -prob) < (best[0], -best[1]):
            best = (rank, prob)
        (-logp[target_id]).backward()
        opt.step()
        with torch.no_grad():
            delta.mul_(mask)
            n = delta.norm()
            if n > budget:
                delta.mul_(budget / n)
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, lambda r, hook: r + delta * mask)])
    rank, prob = _rank_prob(logits[0, -1], target_id)
    if (rank, -prob) < (best[0], -best[1]):
        best = (rank, prob)
    return {"rank": best[0], "prob": best[1]}


def diagnose_layer(model, sae, layer, prompt, target, hybrid_cfg, target_id, tokens, clean_logits, clean_store,
                   run_hybrid, device, opt_steps=60):
    """One prompt at one layer: SAE sweep, trace, held control and the SAE-free bound."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    t0 = time.perf_counter()
    rec = run_hybrid(model, sae, hook_name, layer, device, prompt, target, hybrid_cfg, "all")
    scale_map = scale_map_from_record(rec)
    resid = clean_store[hook_name]
    delta, recon = sae_delta(sae, resid, scale_map)
    x_norm_last = float(resid[0, -1].norm().item())
    x_norm_all = float(resid[0, 1:].norm().item())
    err_rel = float(((resid - recon)[0, -1].norm() / resid[0, -1].norm()).item())

    trace = trace_edit(model, tokens, layer, delta, target_id, clean_logits, clean_store)
    held = held_edit(model, tokens, layer, delta, target_id, clean_store)
    d_norm = float(delta.norm().item())
    d_norm_last = float(delta[0, -1].norm().item())

    opt = {"matched": {}, "scan": {}}
    for pos in ("last", "all"):
        opt["matched"][pos] = optimal_delta(model, tokens, layer, target_id, max(d_norm, 1e-3), pos, opt_steps)
        opt["scan"][pos] = {}
        base = x_norm_last if pos == "last" else x_norm_all
        for fr in OPT_FRACTIONS:
            opt["scan"][pos][str(fr)] = optimal_delta(model, tokens, layer, target_id, fr * base, pos, opt_steps)

    best = rec.get("best_result") or {}
    return {
        "layer": layer,
        "sae": {
            "rank_before": rec["baseline_rank"], "rank_after": best.get("rank", rec["baseline_rank"]),
            "prob_before": rec["baseline_target_prob_pct"] / 100, "prob_after": rec["final_target_prob_pct"] / 100,
            "success": bool(rec["success"]), "features_used": rec.get("features_in_best", 0),
            "stop_reason": rec.get("stop_reason"), "final_top1": rec.get("final_top1"),
            "delta_norm": d_norm, "delta_norm_last": d_norm_last, "resid_norm_last": x_norm_last,
            "relative_push": d_norm_last / x_norm_last, "recon_error_rel": err_rel,
        },
        "trace": trace, "held": held, "optimal": opt,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def diagnose_prompt(model, get_sae, prompt, target, layers, hybrid_cfg, run_hybrid, device,
                    opt_steps=60, progress_cb=None):
    """Full diagnostic of one prompt across `layers`. `get_sae(layer)` returns the SAE for a layer."""
    model.reset_hooks()
    target_id = get_target_token_id(model, " " + target.strip())
    tokens = model.to_tokens(prompt.strip())
    clean_logits, clean_store = _clean_cache(model, tokens)
    rank0, prob0 = _rank_prob(clean_logits, target_id)
    top1 = model.to_string([int(clean_logits.argmax().item())])
    # where does the fact show up in the clean model? (logit lens, every layer)
    clean_lens = []
    for l in range(N_LAYERS):
        rk, pr, mg = _lens(model, clean_store[f"blocks.{l}.hook_resid_pre"][0, -1], target_id)
        clean_lens.append({"layer": l, "rank": rk, "prob": pr, "margin": mg})
    out = {"prompt": prompt, "target": target, "target_token": model.to_string([target_id]),
           "baseline_rank": rank0, "baseline_prob": prob0, "baseline_top1": top1,
           "clean_lens": clean_lens, "layers": []}
    for i, layer in enumerate(layers):
        if progress_cb:
            progress_cb(i, len(layers), f"layer {layer}")
        sae = get_sae(layer)
        out["layers"].append(diagnose_layer(model, sae, layer, prompt, target, hybrid_cfg, target_id, tokens,
                                            clean_logits, clean_store, run_hybrid, device, opt_steps))
    model.reset_hooks()
    return out


def verdict(result):
    """Plain-language reading of one prompt's diagnostic (used by the tab and the script)."""
    lines = []
    ls = result["layers"]
    if not ls:
        return lines
    ok_sae = [l["layer"] for l in ls if l["sae"]["success"]]
    ok_opt = [l["layer"] for l in ls if l["optimal"]["matched"]["all"]["rank"] == 1 or l["optimal"]["matched"]["last"]["rank"] == 1]
    lines.append(f"SAE edit reaches rank 1 at layers: {ok_sae or 'none'}.")
    lines.append(f"A norm-matched SAE-free edit reaches rank 1 at layers: {ok_opt or 'none'}.")
    persist = []
    for l in ls:
        rows = [r for r in l["trace"]["rows"] if r["downstream"] and r["persistence"] is not None]
        if rows:
            persist.append((l["layer"], rows[-1]["persistence"]))
    if persist:
        lines.append("Fraction of the injected direction still present at the last block: " +
                     ", ".join(f"L{a}: {p:.2f}" for a, p in persist) + ".")
    return lines
