"""
Batch driver for the candidate-source benchmark: runs every prompt in a JSON spec through the
Hybrid Mute & Boost sweep with "All prompt positions" and/or "Last token only" candidates,
collects one full record per run, and builds a paired comparison.

Spec format (a bare list of {"prompt","target"} objects is also accepted):

{
  "name": "candidate source study",
  "settings": {"mute_strength": 0.6, "boost_strength": 0.5, "top_n": 120},   # any key of DEFAULT_CFG
  "modes": ["all", "last"],
  "repeats": 1,
  "prompts": [
    {"prompt": "This is Sophia, she is a", "target": "woman"},
    {"prompt": "...", "target": "...", "settings": {"mute_strength": 0.8}}    # optional per-prompt overrides
  ]
}
"""
import json
import os
import time
import traceback
from datetime import datetime

from src.hybrid_runner import run_hybrid_sweep, DEFAULT_CFG, SOURCE_LABELS


def parse_spec(obj):
    """Validate a spec (dict or list) and return a normalised dict. Raises ValueError with a readable message."""
    if isinstance(obj, list):
        obj = {"prompts": obj}
    if not isinstance(obj, dict):
        raise ValueError("Spec must be a JSON object or a list of prompt objects.")
    prompts = obj.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Spec needs a non-empty 'prompts' list.")
    norm_prompts = []
    for i, p in enumerate(prompts, start=1):
        if isinstance(p, (list, tuple)) and len(p) == 2:
            p = {"prompt": p[0], "target": p[1]}
        if not isinstance(p, dict) or not str(p.get("prompt", "")).strip() or not str(p.get("target", "")).strip():
            raise ValueError(f"Prompt #{i} needs non-empty 'prompt' and 'target' fields.")
        overrides = p.get("settings") or {}
        bad = [k for k in overrides if k not in DEFAULT_CFG]
        if bad:
            raise ValueError(f"Prompt #{i}: unknown setting(s) {bad}. Allowed: {sorted(DEFAULT_CFG)}")
        norm_prompts.append({"prompt": str(p["prompt"]).strip(), "target": str(p["target"]).strip().lstrip(),
                             "settings": overrides})
    settings = obj.get("settings") or {}
    bad = [k for k in settings if k not in DEFAULT_CFG]
    if bad:
        raise ValueError(f"Unknown setting(s) {bad}. Allowed: {sorted(DEFAULT_CFG)}")
    modes = obj.get("modes") or ["all", "last"]
    modes = [m.strip().lower() for m in modes]
    bad = [m for m in modes if m not in SOURCE_LABELS]
    if bad:
        raise ValueError(f"Unknown mode(s) {bad}. Use 'all' and/or 'last'.")
    repeats = int(obj.get("repeats", 1))
    if repeats < 1 or repeats > 20:
        raise ValueError("'repeats' must be between 1 and 20.")
    return {"name": obj.get("name") or "candidate source batch", "settings": settings, "modes": modes,
            "repeats": repeats, "prompts": norm_prompts}


def _safe_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _run_view(rec):
    """Compact per-run numbers used by the paired table."""
    if rec is None:
        return None
    if rec.get("error"):
        return {"error": rec["error"]}
    r0 = (rec.get("rounds") or [{}])[0]
    best = rec.get("best_result") or {}
    return {
        "final_rank": best.get("rank"),
        "final_prob_pct": (best.get("prob") or 0) * 100,
        "final_top1": rec.get("final_top1"),
        "success": bool(rec.get("success")),
        "steps": len(rec.get("hybrid_details") or []),
        "rounds": len(rec.get("rounds") or []),
        "features_at_best": (best.get("mute_size") or 0) + (best.get("boost_size") or 0),
        "round0_safe_mute": r0.get("safe_mutes"),
        "round0_safe_boost": r0.get("safe_boosts"),
        "round0_candidates_available": ((rec.get("round_details") or [{}])[0].get("available_unapplied")),
        "stop_reason": rec.get("stop_reason"),
        "time_s": rec.get("total_time_s"),
    }


def _winner(a, l, baseline_rank, single_token=True):
    if single_token is False:
        return "n/a (multi-token target)"
    if baseline_rank == 1:
        return "n/a (already rank 1)"
    if a is None or l is None or a.get("error") or l.get("error"):
        return "n/a (error / missing mode)"
    ka = (0 if a["success"] else 1, a["final_rank"], -a["final_prob_pct"])
    kl = (0 if l["success"] else 1, l["final_rank"], -l["final_prob_pct"])
    if (a["success"], a["final_rank"]) == (l["success"], l["final_rank"]):
        return "tie"
    return "all" if ka < kl else "last"


def build_paired_summary(runs):
    """Group runs by (prompt, target, repeat) and compare the two modes."""
    groups = {}
    for r in runs:
        key = (r["prompt"], r["target"], r["repeat"])
        groups.setdefault(key, {})[r["candidate_source_key"]] = r
    rows = []
    for (prompt, target, repeat), g in groups.items():
        a, l = _run_view(g.get("all")), _run_view(g.get("last"))
        base = next((x for x in (g.get("all"), g.get("last")) if x and not x.get("error")), None)
        baseline_rank = base.get("baseline_rank") if base else None
        rows.append({
            "prompt": prompt, "target": target, "repeat": repeat, "baseline_rank": baseline_rank,
            "baseline_prob_pct": (float(base["baseline_target_prob"].rstrip("%")) if base else None),
            "target_single_token": base.get("target_is_single_token") if base else None,
            "all": a, "last": l, "winner": _winner(a, l, baseline_rank, base.get("target_is_single_token") if base else True),
        })
    steerable = [x for x in rows if x["baseline_rank"] and x["baseline_rank"] > 1 and x["all"] and x["last"]
                 and not x["all"].get("error") and not x["last"].get("error") and x["target_single_token"] is not False]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    agg = {
        "prompt_pairs_total": len(rows),
        "steerable_pairs (baseline rank > 1, single-token target, both modes ran)": len(steerable),
        "already_rank1_pairs": sum(1 for x in rows if x["baseline_rank"] == 1),
        "excluded_multi_token_target_pairs": [x["prompt"] + " -> " + x["target"] for x in rows if x["target_single_token"] is False],
        "winner_counts": {k: sum(1 for x in rows if x["winner"] == k) for k in ("all", "last", "tie")},
        "success_rate_all": mean([1.0 if x["all"]["success"] else 0.0 for x in steerable]),
        "success_rate_last": mean([1.0 if x["last"]["success"] else 0.0 for x in steerable]),
        "mean_rank_gain_all": mean([x["baseline_rank"] - x["all"]["final_rank"] for x in steerable]),
        "mean_rank_gain_last": mean([x["baseline_rank"] - x["last"]["final_rank"] for x in steerable]),
        "mean_features_at_best_all": mean([x["all"]["features_at_best"] for x in steerable]),
        "mean_features_at_best_last": mean([x["last"]["features_at_best"] for x in steerable]),
        "mean_round0_candidates_all": mean([x["all"]["round0_candidates_available"] for x in steerable]),
        "mean_round0_candidates_last": mean([x["last"]["round0_candidates_available"] for x in steerable]),
    }
    return rows, agg


def run_batch(model, sae, hook_name, layer, device, spec, progress_cb=None, save_path=None, start_id=1):
    """
    Execute a normalised spec. `progress_cb(done, total, record)` is called after every run.
    If `save_path` is set the full result is rewritten after every run so nothing is lost on a crash.
    """
    spec = parse_spec(spec)
    jobs = []
    for rep in range(1, spec["repeats"] + 1):
        for p in spec["prompts"]:
            for mode in spec["modes"]:
                jobs.append((rep, p, mode))
    total = len(jobs)
    meta = {"name": spec["name"], "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "layer": layer,
            "hook_name": hook_name, "device": device, "total_runs": total, "finished": None}
    result = {"meta": meta, "spec": spec, "runs": [], "paired_summary": [], "aggregate": {}}
    t0 = time.perf_counter()

    for i, (rep, p, mode) in enumerate(jobs, start=1):
        cfg = dict(spec["settings"])
        cfg.update(p["settings"])
        try:
            rec = run_hybrid_sweep(model, sae, hook_name, layer, device, p["prompt"], p["target"], cfg, candidate_source=mode)
        except Exception as e:                                   # keep the batch going; record the failure
            model.reset_hooks()
            rec = {"prompt": p["prompt"], "target": p["target"], "mode": "Hybrid Mute & Boost", "layer": layer,
                   "candidate_source": SOURCE_LABELS[mode], "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(), "success": False,
                   "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        rec["run_id"] = start_id + i - 1
        rec["batch_index"] = i
        rec["repeat"] = rep
        rec["candidate_source_key"] = mode
        result["runs"].append(rec)
        if save_path:
            meta["elapsed_s"] = round(time.perf_counter() - t0, 1)
            _safe_dump(result, save_path)
        if progress_cb:
            progress_cb(i, total, rec)

    rows, agg = build_paired_summary(result["runs"])
    result["paired_summary"], result["aggregate"] = rows, agg
    meta["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["elapsed_s"] = round(time.perf_counter() - t0, 1)
    meta["errors"] = sum(1 for r in result["runs"] if r.get("error"))
    if save_path:
        _safe_dump(result, save_path)
    return result
