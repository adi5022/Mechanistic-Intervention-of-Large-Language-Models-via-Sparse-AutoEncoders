"""
Headless (no-Streamlit) runner for the Hybrid Mute & Boost sweep.

This is a faithful port of the run logic in `experiment_app.py` (Tab "Hybrid mute and boost") so the same
experiment can be executed over many prompts and both candidate sources without touching the UI.
It returns one JSON-serialisable record per run that contains everything the tab shows on screen (baseline
card, per-token feature counts, pool tables, overlap fixes, re-tested features, every sweep step with its
top-5 table and combination-safety check, best-so-far events, per-round timeline, run summary, event log,
rank progression, stopwatch, ledger, rejections) plus the per-candidate scores behind the pools.

The record keeps the same top-level keys as the Session History records, so a batch result can be uploaded
into the Session History tab as well.
"""
import contextlib
import io
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from src.editing import (
    get_target_token_id,
    get_top_competitor_features,
    get_top_target_features,
    check_target_safe,
    check_boost_safe,
    check_target_safe_batch,
    check_boost_safe_batch,
    check_combination_safe,
    build_clean_context,
    build_steered_context,
)
from src.hooks import make_mute_and_boost_hook, build_scale_map_graded, make_scale_map_hook
from src.batched_eval import MAX_EVAL_BATCH, batched_ablation_probs_and_ranks, batched_probs_and_ranks_per_row_scales
from src.format_utils import fmt_prob

SOURCE_LABELS = {"all": "All prompt positions", "last": "Last token only"}

DEFAULT_CFG = {
    "mute_strength": 0.6,
    "boost_strength": 0.5,
    "top_n": 120,
    "cumulative_sweep": True,
    "pool_refill": True,
    "max_refill_rounds": 0,
    "mute_sizes": [1, 3, 5],
    "boost_sizes": [1, 3, 5],
    "safety_filter": True,
    "stop_on_rank1": True,
    "use_batched": True,
    # --- dynamic safety filtering ---
    "safety_mode": "strict",        # "strict" (original) | "tolerance" (accept small harm) | "graded" (per-feature strength)
    "tolerance": 0.01,              # harm budget per feature = max(1e-6, tolerance * current target probability)
    "rank_slack": 0,                # a candidate may worsen the target's rank by up to this many places
    "alpha_min": 0.05,              # graded: give up on a feature below this fraction of the configured strength
    "max_backoff": 4,               # graded: at most this many halvings after the first computed strength
    "rescued_order": "after",       # "after": budget-accepted features follow the strictly-safe ones; "interleaved": keep rank order
    "max_steps": 0,                 # 0 = unlimited sweep steps
    "collateral": True,             # measure damage on unrelated prompts for the best result
    "combination_check": True,      # per-step new-blocker check (2 extra forward passes); False = same ranks, ~3x faster sweeps
    "record_detail": "full",        # "full" | "compact": compact drops per-step feature lists / top-5 tables / later-round candidate tables (~20x smaller)
    "multi_token": "last_piece",    # "last_piece" (legacy behaviour) | "first_piece" (score the first token of the word)
}

NEUTRAL_PROMPTS = [
    "The weather today is", "I went to the store to buy some", "She smiled and said", "In the beginning of the",
    "My favorite food is", "The meeting will start at", "He picked up the phone and", "According to the report,",
    "Once upon a time, there was a", "The best way to learn is to", "After the game, everyone went", "The new law will",
    "Yesterday I saw a", "The company announced that", "It was a dark and", "You should always remember to",
    "The children were playing in the", "Scientists have discovered that", "The old man walked slowly to the", "We need to talk about",
]


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _fmt_rank_change(new, old):
    d = old - new
    return f"{d:+d} ranks ({'better' if d > 0 else 'worse' if d < 0 else 'no change'})"


def _collateral(model, sae, hook_name, scale_map, prompts):
    """
    Collateral-damage guardrail: apply the FINAL edit (the best result's per-feature scale map) to unrelated
    prompts and measure how far the next-token distribution moves: KL(clean || edited) in nats and whether the
    top-1 word changes. Features that are not active on a prompt cannot change it, so this measures real
    interference only where the edited features actually fire.
    """
    kls, flips = [], 0
    for pr in prompts:
        toks = model.to_tokens(pr)
        model.reset_hooks()
        with torch.no_grad():
            clean = torch.log_softmax(model(toks)[0, -1], dim=-1)
            model.add_hook(hook_name, make_scale_map_hook(scale_map, sae))
            edited = torch.log_softmax(model(toks)[0, -1], dim=-1)
        model.reset_hooks()
        kls.append(float(torch.sum(clean.exp() * (clean - edited)).item()))
        flips += int(clean.argmax().item() != edited.argmax().item())
    n = max(1, len(prompts))
    return {"n_prompts": len(prompts), "mean_kl_nats": sum(kls) / n, "max_kl_nats": max(kls) if kls else 0.0,
            "top1_flip_rate": flips / n, "per_prompt_kl": [round(k, 8) for k in kls]}


_STEP_KEEP = ("round", "step", "mute_batch_size", "boost_batch_size", "mute_strength", "boost_strength", "new_top1", "target_prob",
              "target_rank", "target_prob_pct", "is_new_best", "note", "carried_over", "added_this_round",
              "rank_change_vs_round_baseline", "rank_change_vs_clean_prompt")


def _compact_steps(steps):
    return [{k: s_[k] for k in _STEP_KEEP if k in s_} for s_ in steps]


def _top_tokens(model, probs, k=5, target_id=None):
    vals, idx = torch.topk(probs, k=k)
    rows = []
    for r, (p, i) in enumerate(zip(vals, idx), start=1):
        rows.append({
            "Rank": r, "Token": model.to_string([i.item()]), "Probability": f"{fmt_prob(p.item())}",
            "Is Target": "Yes (TARGET)" if i.item() == target_id else "No",
        })
    return rows


def run_hybrid_sweep(model, sae, hook_name, layer, device, prompt, target, cfg=None, candidate_source="all"):
    """Run one Hybrid Mute & Boost experiment. `candidate_source` is "all" or "last"."""
    c = dict(DEFAULT_CFG)
    c.update(cfg or {})
    if candidate_source not in SOURCE_LABELS:
        raise ValueError(f"candidate_source must be 'all' or 'last', got {candidate_source!r}")
    cand_positions = candidate_source
    source_label = SOURCE_LABELS[candidate_source]

    sm, sb = float(c["mute_strength"]), float(c["boost_strength"])
    top_n = int(c["top_n"])
    cumulative_sweep = bool(c["cumulative_sweep"])
    refill_on = bool(cumulative_sweep and c["pool_refill"])
    max_rounds = int(c["max_refill_rounds"])
    use_safety = bool(c["safety_filter"])
    use_batched = bool(c["use_batched"])
    stop_on_rank1 = bool(c["stop_on_rank1"])
    mute_sizes = [int(x) for x in c["mute_sizes"]]
    boost_sizes = [int(x) for x in c["boost_sizes"]]
    safety_mode = c["safety_mode"]
    if safety_mode not in ("strict", "tolerance", "graded"):
        raise ValueError(f"safety_mode must be strict/tolerance/graded, got {safety_mode!r}")
    if safety_mode != "strict" and not use_batched:
        raise ValueError("tolerance/graded safety modes need use_batched=True")
    tau = float(c["tolerance"])
    rank_slack = int(c["rank_slack"])
    alpha_min = float(c["alpha_min"])
    max_backoff = int(c["max_backoff"])
    rescued_after = c["rescued_order"] == "after"
    max_steps = int(c["max_steps"])
    graded = safety_mode == "graded"
    combination_check = bool(c["combination_check"])
    compact = c["record_detail"] == "compact"

    prompt = prompt.strip()
    target = target.strip()
    target_str = " " + target
    _tgt_ids = model.to_tokens(target_str, prepend_bos=False).squeeze(0).reshape(-1)
    n_target_tokens = int(_tgt_ids.numel())
    if c["multi_token"] == "first_piece" and n_target_tokens > 1:
        target_token_id = int(_tgt_ids[0].item())
    else:
        target_token_id = get_target_token_id(model, target_str)
    target_token_str = model.to_string([target_token_id])

    screen = []                       # ordered list of the messages the tab would print

    def say(msg):
        screen.append(msg)

    _sync(device)
    t_start = time.perf_counter()

    model.reset_hooks()
    clean_ctx = build_clean_context(model, sae, prompt, target_token_id)
    tokens = clean_ctx.tokens
    probs = clean_ctx.clean_probs
    current_top1_id = torch.argmax(probs).item()
    current_top1_str = model.to_string([current_top1_id])
    baseline_target_prob = clean_ctx.clean_target_prob
    t_baseline_done = time.perf_counter()
    baseline_top5 = _top_tokens(model, probs, 5, target_token_id)
    say(f"Baseline Top-1: `{current_top1_str}` | Target '{target_str}' Prob: `{fmt_prob(baseline_target_prob)}` | rank #{clean_ctx.clean_rank}")

    def _chunks(n):
        return (-(-n // MAX_EVAL_BATCH) if use_batched else n) if n else 0

    def assess_side(side, fids, ctx, base_map):
        """
        Safety-test every candidate of one side against the current baseline `ctx`.

        strict:    accept if rank does not worsen and target prob drops by <= 1e-6 (the original rule).
        tolerance: also accept if the drop is within a budget = max(1e-6, tolerance * current target prob)
                   and the rank slips by at most `rank_slack`.
        graded:    like tolerance, but a candidate that fails at full strength is retried at a smaller
                   strength (first a proportional guess budget/harm, then repeated halving) and accepted at
                   the largest fraction `alpha` of the configured strength that fits the budget.
        Returns ({fid: assessment dict}, number_of_batched_passes).
        """
        strength = sm if side == "mute" else sb
        sign = -1.0 if side == "mute" else 1.0
        full_scale = 1.0 + sign * strength
        clean_p, clean_r = ctx.clean_target_prob, ctx.clean_rank
        thr = 1e-6 if safety_mode == "strict" else max(1e-6, tau * clean_p)
        info = {}
        if not fids:
            return info, 0
        probs_t, ranks_t = batched_ablation_probs_and_ranks(
            model, sae, ctx.tokens, fids, full_scale, target_token_id, hook_name, base_scale_map=base_map)
        passes = _chunks(len(fids))
        for fid, pr, rk in zip(fids, probs_t.tolist(), ranks_t.tolist()):
            delta = pr - clean_p
            harm_fail = delta < -1e-6
            rank_fail = rk > clean_r
            strict_ok = (not harm_fail) and (not rank_fail)
            reason = "ok" if strict_ok else ("both" if harm_fail and rank_fail else "harm" if harm_fail else "rank")
            accepted = strict_ok or (safety_mode != "strict" and delta >= -thr and rk <= clean_r + rank_slack)
            info[fid] = {"delta_full": delta, "rank_full": rk, "strict_ok": strict_ok, "reason": reason,
                         "accepted": accepted, "alpha": 1.0 if accepted else None, "delta": delta, "rank": rk,
                         "threshold": thr}
        if graded:
            pending = {}
            for fid, a in info.items():
                if a["accepted"]:
                    continue
                d = a["delta_full"]
                if d < -thr:                                   # harm ~ proportional to strength: first guess budget/harm
                    guess = 0.95 * thr / abs(d)
                    if guess >= alpha_min:
                        pending[fid] = guess
                else:                                          # harm fine, only the rank slipped: try half strength
                    pending[fid] = 0.5
            for _ in range(max_backoff + 1):
                if not pending:
                    break
                ids = list(pending)
                alphas = [pending[f] for f in ids]
                scales = [1.0 + sign * strength * al for al in alphas]
                pt, rt = batched_probs_and_ranks_per_row_scales(
                    model, sae, ctx.tokens, ids, scales, target_token_id, hook_name, base_scale_map=base_map)
                passes += _chunks(len(ids))
                nxt = {}
                for fid, al, pr, rk in zip(ids, alphas, pt.tolist(), rt.tolist()):
                    delta = pr - clean_p
                    if delta >= -thr and rk <= clean_r + rank_slack:
                        info[fid].update(accepted=True, alpha=al, delta=delta, rank=rk)
                    elif al * 0.5 >= alpha_min:
                        nxt[fid] = al * 0.5
                pending = nxt
        return info, passes

    def build_pools(ctx, base_map, exclude_ids):
        blocker_id = int(torch.argmax(ctx.clean_probs).item())
        with torch.no_grad():
            acts_all = sae.encode(ctx.resid_all[0])
        tok_strs = model.to_str_tokens(prompt)
        per_token = [(tok_strs[i], int((acts_all[i] > 0).sum().item())) for i in range(1, acts_all.shape[0])]
        if cand_positions == "all":
            active_mask = (acts_all[1:] > 0).any(dim=0)
        else:
            active_mask = acts_all[-1] > 0
        active_ids = set(torch.nonzero(active_mask).flatten().tolist())
        available = len(active_ids - set(exclude_ids))
        eff_top_n = min(top_n, available)

        comp_feats = get_top_competitor_features(
            model, sae, prompt, blocker_id, top_n=eff_top_n, clean_ctx=ctx,
            use_batched=use_batched, exclude_ids=exclude_ids, base_scale_map=base_map, positions=cand_positions
        ) if eff_top_n > 0 else []
        tgt_feats = get_top_target_features(
            model, sae, prompt, target_token_id, top_n=eff_top_n, clean_ctx=ctx,
            use_batched=use_batched, exclude_ids=exclude_ids, base_scale_map=base_map, positions=cand_positions
        ) if eff_top_n > 0 else []
        comp_fids = [f for f, _ in comp_feats]
        tgt_fids = [f for f, _ in tgt_feats]
        comp_effect = dict(comp_feats)
        tgt_effect = dict(tgt_feats)
        n_passes = _chunks(len(comp_fids)) + _chunks(len(tgt_fids))

        comp_ok, comp_rej, comp_delta = [], [], {}
        tgt_ok, tgt_rej, tgt_delta = [], [], {}
        comp_alpha = {f: 1.0 for f in comp_fids}
        tgt_alpha = {f: 1.0 for f in tgt_fids}
        assessments = {"mute": {}, "boost": {}}
        if use_safety:
            if use_batched:
                for side, fids in (("mute", comp_fids), ("boost", tgt_fids)):
                    info, side_passes = assess_side(side, fids, ctx, base_map)
                    assessments[side] = info
                    n_passes += side_passes
                    strict_list, rescued_list, rej_list, dmap, amap = [], [], [], {}, {}
                    for fid in fids:
                        a = info[fid]
                        if a["accepted"]:
                            (strict_list if a["strict_ok"] else rescued_list).append(fid)
                            dmap[fid] = a["delta"]
                            amap[fid] = a["alpha"]
                        else:
                            rej_list.append((fid, a["delta_full"]))
                            dmap[fid] = a["delta_full"]
                    if rescued_after:
                        ok_list = strict_list + rescued_list
                    else:
                        ok_list = [f for f in fids if info[f]["accepted"]]
                    if side == "mute":
                        comp_ok, comp_rej, comp_delta, comp_alpha = ok_list, rej_list, dmap, amap
                    else:
                        tgt_ok, tgt_rej, tgt_delta, tgt_alpha = ok_list, rej_list, dmap, amap
            else:
                for fid in comp_fids:
                    ok, d = check_target_safe(model, sae, prompt, fid, target_token_id, strength=sm,
                                              clean_target_prob=ctx.clean_target_prob, clean_rank=ctx.clean_rank,
                                              base_scale_map=base_map)
                    (comp_ok if ok else comp_rej).append(fid if ok else (fid, d))
                    comp_delta[fid] = d
                for fid in tgt_fids:
                    ok, d, _r = check_boost_safe(model, sae, prompt, fid, target_token_id, strength=sb,
                                                 clean_target_prob=ctx.clean_target_prob, clean_rank=ctx.clean_rank,
                                                 base_scale_map=base_map)
                    (tgt_ok if ok else tgt_rej).append(fid if ok else (fid, d))
                    tgt_delta[fid] = d
                n_passes += _chunks(len(comp_fids)) + _chunks(len(tgt_fids))
        else:
            comp_ok, tgt_ok = list(comp_fids), list(tgt_fids)

        overlap = []
        both = [f for f in comp_ok if f in set(tgt_ok)]
        for fid in both:
            if use_safety:
                md, bd = comp_delta[fid], tgt_delta[fid]
                keep = "boost" if bd > md else "mute"
                detail = {"Target Δprob if muted (%)": md * 100, "Target Δprob if boosted (%)": bd * 100}
            else:
                keep = "boost" if tgt_fids.index(fid) < comp_fids.index(fid) else "mute"
                detail = {"Position in mute list": comp_fids.index(fid) + 1, "Position in boost list": tgt_fids.index(fid) + 1}
            overlap.append({"Feature": fid, **detail, "Kept on": keep})
        drop_from_boost = {o["Feature"] for o in overlap if o["Kept on"] == "mute"}
        drop_from_mute = {o["Feature"] for o in overlap if o["Kept on"] == "boost"}
        comp_ok = [f for f in comp_ok if f not in drop_from_mute]
        tgt_ok = [f for f in tgt_ok if f not in drop_from_boost]

        return {
            "overlap": overlap, "blocker_id": blocker_id, "available": available, "n_active": len(active_ids),
            "eff_top_n": eff_top_n, "per_token": per_token,
            "comp_fids": comp_fids, "tgt_fids": tgt_fids,
            "comp_ok": comp_ok, "tgt_ok": tgt_ok, "comp_rej": comp_rej, "tgt_rej": tgt_rej,
            "comp_delta": comp_delta, "tgt_delta": tgt_delta,
            "comp_effect": comp_effect, "tgt_effect": tgt_effect, "passes": n_passes,
            "comp_alpha": comp_alpha, "tgt_alpha": tgt_alpha, "assessments": assessments,
        }

    hybrid_details = []
    best_so_far = {
        "rank": clean_ctx.clean_rank, "prob": baseline_target_prob, "top1": current_top1_str,
        "mute_size": 0, "boost_size": 0, "mute_features": [], "boost_features": [], "step": 0,
    }
    rank_progression = [{
        "Step": 0, "Label": "Baseline", "Round": 0,
        "Target Rank": clean_ctx.clean_rank, "Target Prob (%)": baseline_target_prob * 100, "Target Prob": fmt_prob(baseline_target_prob),
    }]
    refill_markers = []
    is_any_success = False
    step_counter = 0
    stop_sweep = False
    hit_step_cap = False
    stop_reason = None
    applied_mutes, applied_boosts = [], []
    rescued_set = {"mute": set(), "boost": set()}     # features accepted ONLY because of tolerance / graded strength
    applied_mute_str, applied_boost_str = {}, {}       # fid -> actual strength applied (configured strength x alpha)
    ledger = []
    rejected_history = {"mute": {}, "boost": {}}
    all_rejections = []
    round_records = []
    round_screens = []                 # per-round "what the tab shows" details
    event_log = []
    total_passes = {"baseline": 1, "ranking+safety": 0, "steered baseline": 0, "sweep": 0}
    filter_time = 0.0

    ctx = clean_ctx
    prev_ctx = clean_ctx
    prev_start_counts = (0, 0)
    round_idx = 0

    while True:
        round_t0 = time.perf_counter()
        base_map = build_scale_map_graded(applied_mute_str, applied_boost_str)
        round_top1_id = int(torch.argmax(ctx.clean_probs).item())
        round_top1_str = model.to_string([round_top1_id])
        round_top1_prob = ctx.clean_probs[round_top1_id].item()

        if round_idx == 0 and round_top1_id == target_token_id:
            stop_reason = "Target is already Rank #1 in the clean baseline — nothing to steer."
            is_any_success = True
            say(stop_reason)
            break

        round_start_counts = (len(applied_mutes), len(applied_boosts))
        round_screen = {
            "round": round_idx,
            "start_state": {
                "target_rank": ctx.clean_rank, "target_prob_pct": ctx.clean_target_prob * 100,
                "blocker": round_top1_str, "blocker_prob_pct": round_top1_prob * 100,
                "applied_mutes": len(applied_mutes), "applied_boosts": len(applied_boosts),
            },
            "steps": [],
        }
        if round_idx > 0:
            round_screen["previous_round"] = {
                "start_rank": prev_ctx.clean_rank, "end_rank": ctx.clean_rank,
                "start_prob_pct": prev_ctx.clean_target_prob * 100, "end_prob_pct": ctx.clean_target_prob * 100,
                "applied_before": list(prev_start_counts), "applied_after": [len(applied_mutes), len(applied_boosts)],
                "improved": prev_ctx.clean_rank != ctx.clean_rank,
            }
            if prev_ctx.clean_rank == ctx.clean_rank:
                say(f"Round {round_idx - 1} did not improve the target's rank (#{prev_ctx.clean_rank} → #{ctx.clean_rank}).")
        say(f"Round {round_idx}: target rank #{ctx.clean_rank} ({fmt_prob(ctx.clean_target_prob)}), blocker `{round_top1_str}` at {fmt_prob(round_top1_prob)}")

        tf0 = time.perf_counter()
        pools = build_pools(ctx, base_map, set(applied_mutes) | set(applied_boosts))
        filter_time += time.perf_counter() - tf0
        total_passes["ranking+safety"] += pools["passes"]
        comp_ids, target_ids = pools["comp_ok"], pools["tgt_ok"]
        for _side in ("mute", "boost"):
            for _fid, _a in pools["assessments"][_side].items():
                if _a["accepted"] and not _a["strict_ok"]:
                    rescued_set[_side].add(_fid)

        for fid, d in pools["comp_rej"]:
            rejected_history["mute"].setdefault(fid, []).append(round_idx)
            all_rejections.append((round_idx, "mute", fid, d))
        for fid, d in pools["tgt_rej"]:
            rejected_history["boost"].setdefault(fid, []).append(round_idx)
            all_rejections.append((round_idx, "boost", fid, d))

        round_screen["blocking_token"] = {"token": round_top1_str, "prob_pct": round_top1_prob * 100}
        round_screen["active_features_per_token"] = [{"token": t, "n_active": n} for t, n in pools["per_token"]]
        round_screen["n_active_distinct"] = pools["n_active"]
        round_screen["available_unapplied"] = pools["available"]
        round_screen["top_n_requested"] = top_n
        round_screen["top_n_effective"] = pools["eff_top_n"]
        round_screen["top_n_capped"] = top_n > pools["available"]
        round_screen["pool_table"] = [
            {"Pool": "Mute (competitor features)", "Requested (Top N)": top_n, "Available": pools["available"],
             "Candidates evaluated": len(pools["comp_fids"]),
             "Safe (after overlap fix)": len(pools["comp_ok"]) if use_safety else "filter off",
             "Rejected": len(pools["comp_rej"]),
             "Moved to other side (overlap)": sum(1 for o in pools["overlap"] if o["Kept on"] == "boost")},
            {"Pool": "Boost (target features)", "Requested (Top N)": top_n, "Available": pools["available"],
             "Candidates evaluated": len(pools["tgt_fids"]),
             "Safe (after overlap fix)": len(pools["tgt_ok"]) if use_safety else "filter off",
             "Rejected": len(pools["tgt_rej"]),
             "Moved to other side (overlap)": sum(1 for o in pools["overlap"] if o["Kept on"] == "mute")},
        ]
        round_screen["overlap"] = pools["overlap"]
        round_screen["mute_candidates"] = [
            {"feature": f, "blocker_prob_delta_if_ablated": pools["comp_effect"].get(f),
             "target_prob_delta_if_muted": pools["comp_delta"].get(f),
             "verdict": ("safe" if f in pools["comp_ok"] else
                         "moved to boost side" if f in {o["Feature"] for o in pools["overlap"] if o["Kept on"] == "boost"} else "rejected")}
            for f in pools["comp_fids"]]
        round_screen["boost_candidates"] = [
            {"feature": f, "target_prob_delta_if_ablated": pools["tgt_effect"].get(f),
             "target_prob_delta_if_boosted": pools["tgt_delta"].get(f),
             "verdict": ("safe" if f in pools["tgt_ok"] else
                         "moved to mute side" if f in {o["Feature"] for o in pools["overlap"] if o["Kept on"] == "mute"} else "rejected")}
            for f in pools["tgt_fids"]]

        _mv = {c_["feature"]: c_["verdict"] for c_ in round_screen["mute_candidates"]}
        _bv = {c_["feature"]: c_["verdict"] for c_ in round_screen["boost_candidates"]}
        _allf = set(_mv) | set(_bv)
        round_screen["candidate_features_total"] = len(_allf)
        round_screen["unusable_on_both_sides"] = sum(1 for f_ in _allf if _mv.get(f_, "rejected") == "rejected" and _bv.get(f_, "rejected") == "rejected")

        retested = []
        for side, fids, rej, ok in (("mute", pools["comp_fids"], pools["comp_rej"], pools["comp_ok"]),
                                   ("boost", pools["tgt_fids"], pools["tgt_rej"], pools["tgt_ok"])):
            rej_now = {f for f, _ in rej}
            for fid in fids:
                earlier = [r for r in rejected_history[side].get(fid, []) if r < round_idx]
                if earlier:
                    retested.append({
                        "Side": side, "Feature": fid, "Rejected in round(s)": ", ".join(map(str, earlier)),
                        f"Round {round_idx} verdict": "rejected again" if fid in rej_now else "now safe",
                    })
        round_screen["retested_features"] = retested

        _sa_rows, _sa_counts = {}, {}
        for _side in ("mute", "boost"):
            _rows = []
            for _fid, _a in pools["assessments"][_side].items():
                _rows.append({
                    "feature": _fid, "reason_strict": _a["reason"], "delta_full_pp": _a["delta_full"] * 100, "rank_full": _a["rank_full"],
                    "accepted": _a["accepted"], "rescued": bool(_a["accepted"] and not _a["strict_ok"]), "alpha": _a["alpha"],
                    "delta_at_alpha_pp": _a["delta"] * 100, "rank_at_alpha": _a["rank"],
                })
            _sa_rows[_side] = _rows
            _sa_counts[_side] = {
                "candidates": len(_rows),
                "strict_ok": sum(1 for r_ in _rows if r_["reason_strict"] == "ok"),
                "rescued": sum(1 for r_ in _rows if r_["rescued"]),
                "rejected": sum(1 for r_ in _rows if not r_["accepted"]),
                "strict_fail_harm_only": sum(1 for r_ in _rows if r_["reason_strict"] == "harm"),
                "strict_fail_rank_only": sum(1 for r_ in _rows if r_["reason_strict"] == "rank"),
                "strict_fail_both": sum(1 for r_ in _rows if r_["reason_strict"] == "both"),
                "mean_alpha_rescued": (sum(r_["alpha"] for r_ in _rows if r_["rescued"]) / max(1, sum(1 for r_ in _rows if r_["rescued"]))),
            }
        _any = next((v for v in pools["assessments"]["mute"].values()), None) or next((v for v in pools["assessments"]["boost"].values()), None)
        round_screen["safety_assessment"] = {
            "mode": safety_mode, "tolerance": tau if safety_mode != "strict" else None, "rank_slack": rank_slack,
            "harm_budget_abs": _any["threshold"] if _any else None, "counts": _sa_counts, "candidates": _sa_rows,
        }
        pool_extra = {
            "rescued_mutes": _sa_counts["mute"]["rescued"], "rescued_boosts": _sa_counts["boost"]["rescued"],
            "mute_alpha_partial": {int(f): a for f, a in pools["comp_alpha"].items() if f in pools["comp_ok"] and a < 1.0},
            "boost_alpha_partial": {int(f): a for f, a in pools["tgt_alpha"].items() if f in pools["tgt_ok"] and a < 1.0},
        }

        M, B = len(comp_ids), len(target_ids)
        round_screen["safe_mute_ids"] = list(comp_ids)
        round_screen["safe_boost_ids"] = list(target_ids)
        round_screen["rejected_mute_ids"] = [f for f, _ in pools["comp_rej"]]
        round_screen["rejected_boost_ids"] = [f for f, _ in pools["tgt_rej"]]
        event_log.append(
            f"Round {round_idx}: baseline rank #{ctx.clean_rank} (top-1 `{round_top1_str}`); pools built — "
            f"{M} safe mute / {B} safe boost (rejected {len(pools['comp_rej'])} / {len(pools['tgt_rej'])}; {len(pools['overlap'])} overlapping feature(s) assigned to one side); "
            f"{pools['passes']} forward passes for ranking + safety."
        )

        pool_ids = {
            **pool_extra,
            "safe_mute_ids": list(comp_ids), "safe_boost_ids": list(target_ids),
            "rejected_mute_ids": [f for f, _ in pools["comp_rej"]], "rejected_boost_ids": [f for f, _ in pools["tgt_rej"]],
            "overlap": pools["overlap"],
        }

        if M == 0 and B == 0:
            stop_reason = ("No unapplied active features remain at this position." if pools["available"] == 0
                           else f"Dead end: Round {round_idx} produced no safe mute or boost candidates.")
            say(stop_reason)
            round_records.append({
                "round": round_idx, "baseline_rank": ctx.clean_rank, "baseline_prob": ctx.clean_target_prob,
                "blocker": round_top1_str, "safe_mutes": 0, "safe_boosts": 0, "overlap_resolved": len(pools["overlap"]),
                "rejected_mutes": len(pools["comp_rej"]), "rejected_boosts": len(pools["tgt_rej"]),
                **pool_ids, "steps": 0, "end_rank": ctx.clean_rank, "best_rank": ctx.clean_rank,
                "time_s": round(time.perf_counter() - round_t0, 3),
            })
            round_screens.append(round_screen)
            break

        if cumulative_sweep:
            combo_pairs = [(min(k, M), min(k, B)) for k in range(1, max(M, B) + 1)]
        else:
            combo_pairs = [(m_n, b_n) for m_n in mute_sizes for b_n in boost_sizes]

        round_start_step = step_counter
        round_best_rank = ctx.clean_rank
        round_end_rank = ctx.clean_rank
        round_screen["planned_steps"] = len(combo_pairs)

        for m_n, b_n in combo_pairs:
            if stop_sweep:
                break
            mute_batch = comp_ids[:m_n]
            boost_batch = target_ids[:b_n]
            full_mutes = applied_mutes + mute_batch
            full_boosts = applied_boosts + boost_batch
            m_str = dict(applied_mute_str)
            m_str.update({f: sm * pools["comp_alpha"].get(f, 1.0) for f in mute_batch})
            b_str = dict(applied_boost_str)
            b_str.update({f: sb * pools["tgt_alpha"].get(f, 1.0) for f in boost_batch})
            step_scale_map = build_scale_map_graded(m_str, b_str) if graded else None

            model.reset_hooks()
            if graded:
                combined_fn = make_scale_map_hook(step_scale_map, sae)
            else:
                combined_fn = make_mute_and_boost_hook(full_mutes, sm, full_boosts, sb, sae)
            model.add_hook(hook_name, combined_fn)
            with torch.no_grad():
                logits = model(tokens)
            total_passes["sweep"] += 1
            step_probs = F.softmax(logits[0, -1, :], dim=-1)
            top5_probs, top5_indices = torch.topk(step_probs, k=5)
            new_top1_id = top5_indices[0].item()
            new_top1_str = model.to_string([new_top1_id])
            target_prob = step_probs[target_token_id].item()

            if new_top1_id == target_token_id:
                is_any_success = True

            table_data = []
            for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                table_data.append({
                    "Rank": rank_idx, "Token": model.to_string([idx.item()]),
                    "Probability": f"{fmt_prob(p.item())}",
                    "Is Target": "Yes (TARGET)" if idx.item() == target_token_id else "No",
                })

            if combination_check:
                with contextlib.redirect_stdout(io.StringIO()):        # check_combination_safe prints debug tables
                    safety_res = check_combination_safe(
                        model, sae, prompt,
                        mute_feature_ids=full_mutes, mute_strength=sm,
                        boost_feature_ids=full_boosts, boost_strength=sb,
                        target_token_id=target_token_id, top_k=10, scale_map=step_scale_map
                    )
                total_passes["sweep"] += 2
            else:
                # Same steered forward pass we already have: the rank is computed directly, no extra passes.
                _rank_now = int((step_probs > step_probs[target_token_id]).sum().item()) + 1
                safety_res = {"is_safe": None, "target_clean_rank": clean_ctx.clean_rank, "target_new_rank": _rank_now,
                              "target_clean_prob": baseline_target_prob, "target_new_prob": target_prob, "new_blockers": [],
                              "note": "combination_check disabled (rank computed from the step's own forward pass)"}

            step_counter += 1
            combo_rank = safety_res["target_new_rank"]
            round_end_rank = combo_rank
            round_best_rank = min(round_best_rank, combo_rank)
            rank_progression.append({
                "Step": step_counter, "Label": f"R{round_idx} M{len(full_mutes)}/B{len(full_boosts)}",
                "Round": round_idx, "Target Rank": combo_rank, "Target Prob (%)": target_prob * 100, "Target Prob": fmt_prob(target_prob),
            })
            is_new_best = (
                combo_rank < best_so_far["rank"]
                or (combo_rank == best_so_far["rank"] and target_prob > best_so_far["prob"])
            )
            note = None
            if is_new_best:
                best_so_far = {
                    "rank": combo_rank, "prob": target_prob, "top1": new_top1_str,
                    "mute_size": len(full_mutes), "boost_size": len(full_boosts),
                    "mute_features": list(full_mutes), "boost_features": list(full_boosts),
                    "mute_strengths": {int(f): v for f, v in m_str.items()}, "boost_strengths": {int(f): v for f, v in b_str.items()},
                    "step": step_counter,
                }
                note = f"New best so far — target rank {combo_rank}"
            elif combo_rank > best_so_far["rank"]:
                note = (f"Worse than the best so far (rank {best_so_far['rank']}, found at Mute {best_so_far['mute_size']}/"
                        f"Boost {best_so_far['boost_size']}) — that best is still kept as the reported result.")

            step_rec = {
                "round": round_idx,
                "mute_batch_size": len(full_mutes), "mute_features": list(full_mutes), "mute_strength": sm,
                "boost_batch_size": len(full_boosts), "boost_features": list(full_boosts), "boost_strength": sb,
                "new_top1": new_top1_str, "target_prob": f"{fmt_prob(target_prob)}", "top5": table_data,
                "combination_safety_check": safety_res,
            }
            hybrid_details.append(step_rec)
            round_screen["steps"].append({
                **step_rec,
                "step": step_counter,
                "carried_over": {"mutes": len(applied_mutes), "boosts": len(applied_boosts)},
                "added_this_round": {"mutes": len(mute_batch), "boosts": len(boost_batch)},
                "target_rank": combo_rank, "target_prob_pct": target_prob * 100,
                "rank_change_vs_round_baseline": _fmt_rank_change(combo_rank, ctx.clean_rank),
                "rank_change_vs_clean_prompt": _fmt_rank_change(combo_rank, clean_ctx.clean_rank),
                "is_new_best": is_new_best, "note": note,
            })

            if max_steps and step_counter >= max_steps:
                stop_sweep = True
                hit_step_cap = True
            if stop_on_rank1 and new_top1_id == target_token_id:
                say(f"Target token '{target_str}' reached Rank #1 in Round {round_idx} with {len(full_mutes)} mutes & {len(full_boosts)} boosts! Stopping.")
                stop_sweep = True
                break

        model.reset_hooks()
        round_records.append({
            "round": round_idx, "baseline_rank": ctx.clean_rank, "baseline_prob": ctx.clean_target_prob,
            "blocker": round_top1_str, "safe_mutes": M, "safe_boosts": B, "overlap_resolved": len(pools["overlap"]),
            "rejected_mutes": len(pools["comp_rej"]), "rejected_boosts": len(pools["tgt_rej"]),
            **pool_ids,
            "steps": step_counter - round_start_step, "end_rank": round_end_rank, "best_rank": round_best_rank,
            "time_s": round(time.perf_counter() - round_t0, 3),
        })
        round_screen["end_state"] = {"end_rank": round_end_rank, "best_rank": round_best_rank}
        round_screens.append(round_screen)
        regressed = round_end_rank > ctx.clean_rank

        if stop_sweep:
            stop_reason = (f"Reached the max_steps limit ({max_steps})." if hit_step_cap and not is_any_success
                           else f"Target reached Rank #1 in Round {round_idx}.")
            break
        if not refill_on:
            stop_reason = ("Candidate pool exhausted (pool refill is off)." if cumulative_sweep
                           else "All requested batch-size combinations were run.")
            break

        for fid in comp_ids:
            ledger.append({"Feature": fid, "Side": "mute", "Round joined": round_idx,
                           "Target Δprob when tested (%)": (pools["comp_delta"][fid] * 100) if fid in pools["comp_delta"] else None,
                           "Effect on blocker if fully ablated (%)": pools["comp_effect"].get(fid, 0.0) * 100})
        for fid in target_ids:
            ledger.append({"Feature": fid, "Side": "boost", "Round joined": round_idx,
                           "Target Δprob when tested (%)": (pools["tgt_delta"][fid] * 100) if fid in pools["tgt_delta"] else None,
                           "Effect on target if fully ablated (%)": pools["tgt_effect"].get(fid, 0.0) * 100})
        applied_mutes = applied_mutes + comp_ids
        applied_boosts = applied_boosts + target_ids
        applied_mute_str.update({f: sm * pools["comp_alpha"].get(f, 1.0) for f in comp_ids})
        applied_boost_str.update({f: sb * pools["tgt_alpha"].get(f, 1.0) for f in target_ids})

        if max_rounds and round_idx >= max_rounds:
            stop_reason = f"Reached the Max Refill Rounds limit ({max_rounds})."
            break

        tf0 = time.perf_counter()
        new_ctx = build_steered_context(model, sae, clean_ctx, build_scale_map_graded(applied_mute_str, applied_boost_str), target_token_id)
        filter_time += time.perf_counter() - tf0
        total_passes["steered baseline"] += 1

        event_log.append(
            f"Round {round_idx} pool exhausted at {len(applied_mutes)} mutes / {len(applied_boosts)} boosts applied "
            f"(target rank #{round_end_rank}" + (", REGRESSED vs round baseline — kept applied (track-only)" if regressed else "") +
            f"). Refill: fresh steered forward pass → new baseline rank #{new_ctx.clean_rank}."
        )
        refill_markers.append(step_counter)
        if int(torch.argmax(new_ctx.clean_probs).item()) == target_token_id:
            stop_reason = "Target is already Rank #1 in the steered model (stop-on-rank-1 was off)."
            break
        prev_start_counts = round_start_counts
        prev_ctx, ctx = ctx, new_ctx
        round_idx += 1

    model.reset_hooks()
    say(f"Run finished: {stop_reason}")

    _sync(device)
    t_end = time.perf_counter()
    t_wall_end = time.perf_counter()

    best_final_top1 = best_so_far["top1"]
    best_final_target_prob = best_so_far["prob"]

    collateral = None
    if c["collateral"] and best_so_far["step"] > 0:
        collateral = _collateral(model, sae, hook_name,
                                 build_scale_map_graded(best_so_far.get("mute_strengths", {}), best_so_far.get("boost_strengths", {})),
                                 NEUTRAL_PROMPTS)
    n_rescued_in_best = (sum(1 for f in best_so_far["mute_features"] if f in rescued_set["mute"])
                         + sum(1 for f in best_so_far["boost_features"] if f in rescued_set["boost"]))

    if compact:
        hybrid_details = _compact_steps(hybrid_details)
        for _rs in round_screens:
            _rs["steps"] = _compact_steps(_rs["steps"])
            _rs.pop("mute_candidates", None)                       # redundant with safety_assessment / the counts kept above
            _rs.pop("boost_candidates", None)
            if _rs["round"] > 0 and _rs.get("safety_assessment"):  # per-candidate rows are only kept for round 0
                _rs["safety_assessment"].pop("candidates", None)

    return {
        "run_id": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "Hybrid Mute & Boost",
        "layer": layer,
        "prompt": prompt,
        "target": target,
        "target_token": target_token_str,
        "target_token_count": n_target_tokens,
        "target_is_single_token": n_target_tokens == 1,
        "mute_strength": sm,
        "boost_strength": sb,
        "cumulative_sweep": cumulative_sweep,
        "pool_refill": refill_on,
        "max_refill_rounds": max_rounds if refill_on else None,
        "rounds": round_records,
        "stop_reason": stop_reason,
        "forward_passes": total_passes,
        "mute_sizes": (f"cumulative 1..{hybrid_details[-1]['mute_batch_size']} across {len(round_records)} round(s)"
                       if cumulative_sweep and hybrid_details else ", ".join(map(str, mute_sizes))),
        "boost_sizes": (f"cumulative 1..{hybrid_details[-1]['boost_batch_size']} across {len(round_records)} round(s)"
                        if cumulative_sweep and hybrid_details else ", ".join(map(str, boost_sizes))),
        "top_n": top_n,
        "candidate_source": source_label,
        "use_batched": use_batched,
        "total_time_s": round(t_end - t_start, 3),
        "baseline_top1": current_top1_str,
        "baseline_target_prob": f"{fmt_prob(baseline_target_prob)}",
        "baseline_target_prob_pct": baseline_target_prob * 100,
        "final_top1": best_final_top1,
        "final_target_prob": f"{fmt_prob(best_final_target_prob)}",
        "final_target_prob_pct": best_final_target_prob * 100,
        "success": is_any_success,
        "hybrid_details": hybrid_details,
        "settings": {
            "safety_filter": use_safety, "stop_on_rank1": stop_on_rank1,
            "mute_strength": sm, "boost_strength": sb, "top_n": top_n, "candidate_source": source_label,
            "cumulative_sweep": cumulative_sweep, "pool_refill": refill_on, "max_refill_rounds": max_rounds,
            "gpu_batched": use_batched, "sae_layer": layer, "hook_name": hook_name, "device": device,
            "safety_mode": safety_mode, "tolerance": tau, "rank_slack": rank_slack, "alpha_min": alpha_min,
            "max_backoff": max_backoff, "rescued_order": c["rescued_order"], "max_steps": max_steps,
            "multi_token": c["multi_token"], "combination_check": combination_check, "record_detail": c["record_detail"],
        },
        "safety_mode": safety_mode,
        "collateral": collateral,
        "features_in_best": len(best_so_far["mute_features"]) + len(best_so_far["boost_features"]),
        "rescued_features_in_best": n_rescued_in_best,
        "target_scored_token": target_token_str,
        "baseline_rank": clean_ctx.clean_rank,
        "baseline_top5": baseline_top5,
        "best_result": best_so_far,
        "rank_progression": rank_progression,
        "refill_steps": refill_markers,
        "event_log": event_log,
        "applied_features_ledger": ledger,
        "rejections": [{"round": r, "side": sd, "feature": f, "target_prob_delta": d} for r, sd, f, d in all_rejections],
        "applied_mutes_final": applied_mutes,
        "applied_boosts_final": applied_boosts,
        "timing": {
            "model_compute_s": round(t_end - t_start, 3),
            "filter_s": round(filter_time, 3),
            "sweep_s": round(t_end - t_baseline_done - filter_time, 3),
            "wall_clock_s": round(t_wall_end - t_start, 3),
        },
        "screen_log": screen,
        "round_details": round_screens,
        "run_summary": {
            "rounds_run": len(round_records),
            "features_applied_at_end": {"mute": len(applied_mutes), "boost": len(applied_boosts)},
            "sweep_steps": step_counter,
            "forward_passes_total": sum(total_passes.values()),
            "stop_reason": stop_reason,
            "best_target_rank": best_so_far["rank"],
            "best_target_prob_pct": best_so_far["prob"] * 100,
            "rank_gain_vs_baseline": clean_ctx.clean_rank - best_so_far["rank"],
            "found_at": {"mute": best_so_far["mute_size"], "boost": best_so_far["boost_size"], "step": best_so_far["step"]},
        },
        "xai_best_result_explanation": None,
        "xai_mechanistic_explanation": None,
    }
