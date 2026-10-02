"""
Per-prompt cache for the learned-strength study (docs/Research_Journal/23.md, Phase 2).

For ONE prompt this runs GPT-2 small and the layer-8 SAE once and records everything training will need, so that
training never has to repeat the slow work:

  A. baseline pass:   target rank / probability, the blocker (current top-1), the layer-8 residual at every token
  B. candidates:      the Top-N active SAE features over all prompt positions (same definition as the sweep:
                      non-BOS positions, scored by maximum activation), and their activation at every position
  C. removal pass:    each candidate switched fully off, one at a time (batched): the target's and the blocker's
                      probability afterwards. These reproduce the sweep's mute ranking (blocker drops most) and
                      boost ranking (target drops most). They do NOT depend on any mute/boost strength.
  D. descriptors:     per candidate (activation, positions, removal effects, direct-logit alignment) and per prompt
                      (the 12 summary numbers the PromptNet reads)

Everything is stored as one small .pt file named after the CounterFact case_id, so a run can be resumed by skipping
files that already exist, and several machines can write into the same folder (or have their folders copied together).
"""
import math
import os
import time

import torch

from src.editing import build_clean_context, get_top_active_features
from src.batched_eval import batched_ablation_probs
from src.hooks import make_ablation_hook

SUMMARY_NAMES = [
    "log10_target_rank", "log10_target_prob", "blocker_prob", "logit_gap_blocker_minus_target", "n_tokens",
    "n_active_last_token", "n_active_all_positions", "sum_candidate_activation",
    "best_removal_effect_on_blocker", "sum_top5_removal_effect_on_target", "resid_norm_last_token",
    "sae_recon_error_ratio_last_token",
]


def cache_path(out_dir, case_id):
    return os.path.join(out_dir, f"{case_id}.pt")


def build_one(model, sae, rec, hook_name, top_n=200):
    """Compute the cache entry for one record of data/counterfact_hard_set.json. Returns a dict of CPU tensors/numbers."""
    t0 = time.perf_counter()
    dev = next(model.parameters()).device
    prompt, target_id = rec["prompt"], int(rec["target_token_id"])

    # --- A. baseline pass (the repo's own function, so ranks match the sweep) ---
    model.reset_hooks()
    ctx = build_clean_context(model, sae, prompt, target_id)
    tokens, probs = ctx.tokens, ctx.clean_probs
    blocker_id = int(torch.argmax(probs).item())
    p_t, p_b = float(probs[target_id]), float(probs[blocker_id])
    resid = ctx.resid_all[0].detach()                                   # [P, 768]
    P = resid.shape[0]

    # --- B. candidates over all positions ---
    cands = get_top_active_features(model, sae, prompt, top_n=top_n, clean_ctx=ctx, positions="all")
    fids = [int(f) for f, _ in cands]
    K = len(fids)
    with torch.no_grad():
        acts_all = sae.encode(resid)                                    # [P, n_features]
        recon_last = sae.decode(acts_all[-1:])
        recon_err = float((resid[-1:] - recon_last).norm() / resid[-1:].norm().clamp_min(1e-9))
        n_active_last = int((acts_all[-1] > 0).sum())
        n_active_all = int((acts_all[1:].max(dim=0).values > 0).sum()) if P > 1 else n_active_last
    if K:
        idx = torch.as_tensor(fids, device=dev)
        cand_acts = acts_all[:, idx].detach()                           # [P, K], includes the BOS row (the edit hook scales every position)
        body = cand_acts[1:] if P > 1 else cand_acts
        act_max, pos_max = body.max(dim=0)
        act_last = cand_acts[-1]
        n_pos = (body > 0).sum(dim=0)
        # --- C. removal pass: every candidate fully off, target and blocker probability afterwards ---
        # NOTE: in the batched API `scale` is a MULTIPLIER on the feature (0.0 = fully removed, 1.0 = untouched)
        rm = batched_ablation_probs(model, sae, tokens, fids, 0.0, [target_id, blocker_id], hook_name).detach()   # [K, 2]
        rm_t, rm_b = rm[:, 0], rm[:, 1]
        dt, db = rm_t - p_t, rm_b - p_b
        # --- D. descriptors ---
        W_dec = sae.W_dec[idx].detach()                                 # [K, 768]
        align_t = W_dec @ model.W_U[:, target_id].detach()
        align_b = W_dec @ model.W_U[:, blocker_id].detach()
        dec_norm = W_dec.norm(dim=1)
        top5_t = torch.sort(dt)[0][:5].sum().item()
        best_b = float(db.min())
        summed_act = float(act_max.sum())
    else:
        z = torch.zeros(0, device=dev)
        cand_acts = torch.zeros(P, 0, device=dev)
        act_max = act_last = n_pos = pos_max = rm_t = rm_b = dt = db = align_t = align_b = dec_norm = z
        top5_t = best_b = summed_act = 0.0

    logit_gap = math.log(max(p_b, 1e-30)) - math.log(max(p_t, 1e-30))     # blocker logit minus target logit
    summary = {
        "log10_target_rank": math.log10(ctx.clean_rank), "log10_target_prob": math.log10(max(p_t, 1e-30)),
        "blocker_prob": p_b, "logit_gap_blocker_minus_target": logit_gap, "n_tokens": float(P),
        "n_active_last_token": float(n_active_last), "n_active_all_positions": float(n_active_all),
        "sum_candidate_activation": summed_act, "best_removal_effect_on_blocker": best_b,
        "sum_top5_removal_effect_on_target": top5_t, "resid_norm_last_token": float(resid[-1].norm()),
        "sae_recon_error_ratio_last_token": recon_err,
    }
    cpu = lambda x: x.detach().float().cpu() if torch.is_tensor(x) and x.is_floating_point() else (x.detach().cpu() if torch.is_tensor(x) else x)
    return {
        "case_id": rec["case_id"], "relation_id": rec["relation_id"], "subject": rec["subject"], "prompt": prompt,
        "target": rec["target"], "target_id": target_id, "blocker_id": blocker_id,
        "blocker_token": model.to_string([blocker_id]),
        "tokens": tokens[0].cpu(), "resid8": cpu(resid),
        "baseline_rank": int(ctx.clean_rank), "baseline_prob": p_t, "blocker_prob": p_b, "logit_gap": logit_gap,
        "cand_ids": torch.as_tensor(fids, dtype=torch.long), "cand_acts": cpu(cand_acts),
        "cand_act_max": cpu(act_max), "cand_act_last": cpu(act_last), "cand_n_pos": cpu(n_pos),
        "cand_pos_max": cpu(pos_max), "rm_target_prob": cpu(rm_t), "rm_blocker_prob": cpu(rm_b),
        "dt": cpu(dt), "db": cpu(db), "align_target": cpu(align_t), "align_blocker": cpu(align_b), "dec_norm": cpu(dec_norm),
        "summary": summary, "layer": 8, "top_n": top_n, "device": str(dev), "seconds": round(time.perf_counter() - t0, 3),
    }


def sanity_check(model, sae, entry, hook_name, tol=2e-4):
    """Recompute the removal effect of the strongest candidate ONE feature at a time and compare with the batched value."""
    if entry["cand_ids"].numel() == 0:
        return True, 0.0
    fid = int(entry["cand_ids"][0])
    tokens = model.to_tokens(entry["prompt"])
    model.reset_hooks()
    with torch.no_grad():
        model.add_hook(hook_name, make_ablation_hook(fid, sae, 1.0))
        p = torch.softmax(model(tokens)[0, -1], dim=-1)
    model.reset_hooks()
    single = float(p[entry["target_id"]])
    batched = float(entry["rm_target_prob"][0])
    return abs(single - batched) <= tol, abs(single - batched)
