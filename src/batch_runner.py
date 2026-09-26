"""
Batch driver for prompt studies: runs every prompt in a JSON spec through the Hybrid Mute & Boost sweep,
for each candidate-source mode ("all" / "last") and each "arm" (a named set of setting overrides, e.g. a
safety-filter variant), collects one full record per run and builds paired comparisons.

Spec format (a bare list of {"prompt","target"} objects is also accepted):

{
  "name": "safety filter study",
  "settings": {"mute_strength": 0.6, "boost_strength": 0.5, "top_n": 120},   # defaults for every run (keys of DEFAULT_CFG)
  "modes": ["all"],                                                          # "all" and/or "last"
  "repeats": 1,
  "arms": [                                                                  # optional; default is one arm called "default"
    {"label": "strict",     "settings": {"safety_mode": "strict"}},
    {"label": "graded_5pct", "settings": {"safety_mode": "graded", "tolerance": 0.05}}
  ],
  "prompts": [
    {"prompt": "This is Sophia, she is a", "target": "woman"},
    {"prompt": "...", "target": "...", "settings": {"mute_strength": 0.8}}    # optional per-prompt overrides
  ]
}
"""
import json
import math
import os
import time
import traceback
from datetime import datetime

from src.hybrid_runner import run_hybrid_sweep, DEFAULT_CFG, SOURCE_LABELS


def _check_settings(d, where):
    bad = [k for k in (d or {}) if k not in DEFAULT_CFG]
    if bad:
        raise ValueError(f"{where}: unknown setting(s) {bad}. Allowed: {sorted(DEFAULT_CFG)}")


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
        _check_settings(p.get("settings"), f"Prompt #{i}")
        norm_prompts.append({"prompt": str(p["prompt"]).strip(), "target": str(p["target"]).strip(),
                             "settings": p.get("settings") or {},
                             **({"meta": p["meta"]} if isinstance(p.get("meta"), dict) else {})})
    _check_settings(obj.get("settings"), "settings")
    modes = [m.strip().lower() for m in (obj.get("modes") or ["all", "last"])]
    bad = [m for m in modes if m not in SOURCE_LABELS]
    if bad:
        raise ValueError(f"Unknown mode(s) {bad}. Use 'all' and/or 'last'.")
    repeats = int(obj.get("repeats", 1))
    if repeats < 1 or repeats > 20:
        raise ValueError("'repeats' must be between 1 and 20.")
    arms = obj.get("arms") or [{"label": "default", "settings": {}}]
    norm_arms, seen = [], set()
    for i, a in enumerate(arms, start=1):
        label = str(a.get("label") or f"arm{i}")
        if label in seen:
            raise ValueError(f"Duplicate arm label {label!r}.")
        seen.add(label)
        _check_settings(a.get("settings"), f"Arm {label!r}")
        norm_arms.append({"label": label, "settings": a.get("settings") or {}})
    return {"name": obj.get("name") or "batch", "settings": obj.get("settings") or {}, "modes": modes,
            "repeats": repeats, "arms": norm_arms, "prompts": norm_prompts}


def _safe_dump(obj, path):
    """Write atomically; on Windows another process (antivirus, a viewer) can briefly lock the target, so retry."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    with open(path, "w", encoding="utf-8") as f:          # last resort: overwrite in place
        json.dump(obj, f, indent=1, default=str)


def write_progress(path, payload):
    """Tiny status file the Batch tab polls while a background run is going (kept separate from the big result file)."""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str)
        for _ in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.1)
    except Exception:
        pass


def _run_view(rec):
    """Compact per-run numbers used by the comparison tables."""
    if rec is None:
        return None
    if rec.get("error"):
        return {"error": rec["error"]}
    r0 = (rec.get("rounds") or [{}])[0]
    best = rec.get("best_result") or {}
    col = rec.get("collateral") or {}
    return {
        "final_rank": best.get("rank"),
        "final_prob_pct": (best.get("prob") or 0) * 100,
        "final_top1": rec.get("final_top1"),
        "success": bool(rec.get("success")),
        "steps": len(rec.get("hybrid_details") or []),
        "rounds": len(rec.get("rounds") or []),
        "features_at_best": (best.get("mute_size") or 0) + (best.get("boost_size") or 0),
        "rescued_in_best": rec.get("rescued_features_in_best"),
        "kl_nats": col.get("mean_kl_nats"),
        "top1_flip_rate": col.get("top1_flip_rate"),
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


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def build_paired_summary(runs):
    """Compare the two candidate-source modes per (prompt, target, repeat) within ONE arm."""
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
            "baseline_prob_pct": (base.get("baseline_target_prob_pct") if base else None),
            "target_single_token": base.get("target_is_single_token") if base else None,
            "all": a, "last": l,
            "winner": _winner(a, l, baseline_rank, base.get("target_is_single_token") if base else True),
        })
    steerable = [x for x in rows if x["baseline_rank"] and x["baseline_rank"] > 1 and x["all"] and x["last"]
                 and not x["all"].get("error") and not x["last"].get("error") and x["target_single_token"] is not False]
    agg = {
        "prompt_pairs_total": len(rows),
        "steerable_pairs (baseline rank > 1, single-token target, both modes ran)": len(steerable),
        "already_rank1_pairs": sum(1 for x in rows if x["baseline_rank"] == 1),
        "excluded_multi_token_target_pairs": [x["prompt"] + " -> " + x["target"] for x in rows if x["target_single_token"] is False],
        "winner_counts": {k: sum(1 for x in rows if x["winner"] == k) for k in ("all", "last", "tie")},
        "success_rate_all": _mean([1.0 if x["all"]["success"] else 0.0 for x in steerable]),
        "success_rate_last": _mean([1.0 if x["last"]["success"] else 0.0 for x in steerable]),
        "mean_rank_gain_all": _mean([x["baseline_rank"] - x["all"]["final_rank"] for x in steerable]),
        "mean_rank_gain_last": _mean([x["baseline_rank"] - x["last"]["final_rank"] for x in steerable]),
        "mean_features_at_best_all": _mean([x["all"]["features_at_best"] for x in steerable]),
        "mean_features_at_best_last": _mean([x["last"]["features_at_best"] for x in steerable]),
        "mean_round0_candidates_all": _mean([x["all"]["round0_candidates_available"] for x in steerable]),
        "mean_round0_candidates_last": _mean([x["last"]["round0_candidates_available"] for x in steerable]),
    }
    return rows, agg


def _binom_two_sided(k, n):
    """Exact two-sided sign/McNemar p-value for k of n discordant pairs going one way (p0 = 0.5)."""
    if n == 0:
        return None
    k = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def build_arm_comparison(runs, reference=None):
    """
    Compare arms (safety-filter variants) on identical (prompt, target, mode, repeat) cells.
    Only cells with baseline rank > 1 and a single-token target are scored. Returns (cells, per_arm_mode, vs_reference).
    """
    cells = {}
    arms_seen = []
    for r in runs:
        if r.get("error"):
            continue
        arm = r.get("arm", "default")
        if arm not in arms_seen:
            arms_seen.append(arm)
        cells.setdefault((r["prompt"], r["target"], r["candidate_source_key"], r["repeat"]), {})[arm] = r
    reference = reference or (arms_seen[0] if arms_seen else None)
    scored = {k: v for k, v in cells.items()
              if all(a in v for a in arms_seen)
              and next(iter(v.values()))["baseline_rank"] > 1 and next(iter(v.values())).get("target_is_single_token", True)}

    table = []
    for (prompt, target, mode, repeat), g in scored.items():
        row = {"prompt": prompt, "target": target, "mode": mode, "repeat": repeat,
               "baseline_rank": next(iter(g.values()))["baseline_rank"]}
        for arm in arms_seen:
            row[arm] = _run_view(g[arm])
        table.append(row)

    per_arm_mode = {}
    for mode in sorted({t["mode"] for t in table}):
        sub = [t for t in table if t["mode"] == mode]
        for arm in arms_seen:
            v = [t[arm] for t in sub]
            per_arm_mode[f"{arm} | {mode}"] = {
                "arm": arm, "mode": mode, "n_prompts": len(v),
                "reached_rank1": sum(1 for x in v if x["success"]),
                "success_rate": _mean([1.0 if x["success"] else 0.0 for x in v]),
                "mean_final_rank": _mean([x["final_rank"] for x in v]),
                "median_final_rank": sorted(x["final_rank"] for x in v)[len(v) // 2] if v else None,
                "mean_rank_gain": _mean([t["baseline_rank"] - t[arm]["final_rank"] for t in sub]),
                "mean_final_prob_pct": _mean([x["final_prob_pct"] for x in v]),
                "mean_features_at_best": _mean([x["features_at_best"] for x in v]),
                "mean_rescued_in_best": _mean([x["rescued_in_best"] for x in v]),
                "mean_steps": _mean([x["steps"] for x in v]),
                "mean_time_s": _mean([x["time_s"] for x in v]),
                "mean_kl_nats": _mean([x["kl_nats"] for x in v]),
                "mean_top1_flip_rate": _mean([x["top1_flip_rate"] for x in v]),
            }

    vs_ref = {}
    for mode in sorted({t["mode"] for t in table}):
        sub = [t for t in table if t["mode"] == mode]
        for arm in arms_seen:
            if arm == reference:
                continue
            better = sum(1 for t in sub if (t[arm]["success"], -t[arm]["final_rank"]) > (t[reference]["success"], -t[reference]["final_rank"]))
            worse = sum(1 for t in sub if (t[arm]["success"], -t[arm]["final_rank"]) < (t[reference]["success"], -t[reference]["final_rank"]))
            gained = sum(1 for t in sub if t[arm]["success"] and not t[reference]["success"])
            lost = sum(1 for t in sub if t[reference]["success"] and not t[arm]["success"])
            vs_ref[f"{arm} vs {reference} | {mode}"] = {
                "n": len(sub), "arm_better_rank_or_success": better, "arm_worse_rank_or_success": worse,
                "same": len(sub) - better - worse,
                "prompts_newly_reaching_rank1": gained, "prompts_losing_rank1": lost,
                "sign_test_p_rank_outcome": _binom_two_sided(better, better + worse),
                "mcnemar_p_rank1": _binom_two_sided(gained, gained + lost),
            }
    return table, per_arm_mode, vs_ref


def _read_jsonl(path):
    runs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:            # a half-written last line after a crash
                    pass
    return runs


def merge_results(paths, out_path):
    """Combine several result files (shards of one study) into one finished result file."""
    runs, spec, meta0 = [], None, None
    for p in paths:
        r = None
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        jl = _read_jsonl(p + ".jsonl")
        got = r["runs"] if r else []
        have = {_run_key(x) for x in got}
        got = got + [x for x in jl if _run_key(x) not in have]
        if r and spec is None:
            spec, meta0 = r.get("spec"), r.get("meta")
        runs += got
    runs.sort(key=lambda x: x.get("batch_index", 0))
    for i, r in enumerate(runs, start=1):
        r["run_id"] = i
    meta = dict(meta0 or {})
    meta["total_runs"] = len(runs)
    meta["merged_from"] = [os.path.basename(p) for p in paths]
    meta["elapsed_s"] = max([(json.load(open(p, encoding="utf-8")).get("meta", {}).get("elapsed_s") or 0) for p in paths if os.path.exists(p)] or [0])
    result = {"meta": meta, "spec": spec, "runs": runs, "paired_summary": [], "aggregate": {}}
    finalize_result(result)
    _safe_dump(result, out_path)
    return result


def _run_key(r):
    return (r.get("arm", "default"), r["prompt"], r["target"], r["candidate_source_key"], r["repeat"])


def run_batch(model, sae, hook_name, layer, device, spec, progress_cb=None, save_path=None, start_id=1, resume=False,
              progress_path=None, shard=None, checkpoint_every=50):
    """
    Execute a normalised spec. `progress_cb(done, total, record)` is called after every run.
    If `save_path` is set the full result is rewritten after every run so nothing is lost on a crash;
    with resume=True an existing file at `save_path` is loaded and its finished runs are skipped.
    """
    spec = parse_spec(spec)
    jobs = []
    for rep in range(1, spec["repeats"] + 1):
        for p in spec["prompts"]:
            for arm in spec["arms"]:
                for mode in spec["modes"]:
                    jobs.append((rep, p, arm, mode))
    jobs = list(enumerate(jobs))                                 # keep each job's global position
    if shard:                                                    # (index, count): shuffled round-robin so workers get a similar mix
        import random
        order = list(range(len(jobs)))
        random.Random(12345).shuffle(order)
        mine = {order[k] for k in range(len(order)) if k % shard[1] == shard[0]}
        jobs = [j for j in jobs if j[0] in mine]
    total = len(jobs)
    meta = {"name": spec["name"], "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "layer": layer,
            "hook_name": hook_name, "device": device, "total_runs": total, "finished": None}
    result = {"meta": meta, "spec": spec, "runs": [], "paired_summary": [], "aggregate": {}}
    done_keys = set()
    prior_elapsed = 0.0
    jsonl_path = (save_path + ".jsonl") if save_path else None
    if resume and save_path:
        old_runs, old_meta = [], {}
        if jsonl_path and os.path.exists(jsonl_path):
            old_runs = _read_jsonl(jsonl_path)
        elif os.path.exists(save_path):
            with open(save_path, encoding="utf-8") as f:
                _old = json.load(f)
            old_runs, old_meta = _old.get("runs", []), _old.get("meta", {})
        if os.path.exists(save_path) and not old_meta:
            try:
                with open(save_path, encoding="utf-8") as f:
                    old_meta = json.load(f).get("meta", {})
            except Exception:
                old_meta = {}
        result["runs"] = [r for r in old_runs if not r.get("error")]
        done_keys = {_run_key(r) for r in result["runs"]}
        prior_elapsed = float(old_meta.get("elapsed_s") or 0.0)
        meta["started"] = old_meta.get("started", meta["started"])
        meta["resumed"] = True
    elif jsonl_path and os.path.exists(jsonl_path):
        os.remove(jsonl_path)                                   # a fresh (non-resumed) run starts a fresh log
    if jsonl_path and resume and done_keys:                     # rewrite the log without any failed runs we are about to retry
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in result["runs"]:
                f.write(json.dumps(r, default=str) + "\n")
    t0 = time.perf_counter()
    session_done = 0
    write_progress(progress_path, {"name": spec["name"], "total": total, "done": len(result["runs"]), "elapsed_s": prior_elapsed,
                                   "eta_s": None, "finished": False, "error": None, "out": save_path,
                                   "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "last": None})

    for i, (gidx, (rep, p, arm, mode)) in enumerate(jobs, start=1):
        key = (arm["label"], p["prompt"], p["target"], mode, rep)
        if key in done_keys:
            continue
        cfg = dict(spec["settings"])
        cfg.update(arm["settings"])
        cfg.update(p["settings"])
        try:
            rec = run_hybrid_sweep(model, sae, hook_name, layer, device, p["prompt"], p["target"], cfg, candidate_source=mode)
        except Exception as e:                                   # keep the batch going; record the failure
            model.reset_hooks()
            rec = {"prompt": p["prompt"], "target": p["target"], "mode": "Hybrid Mute & Boost", "layer": layer,
                   "candidate_source": SOURCE_LABELS[mode], "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(), "success": False,
                   "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        rec["run_id"] = start_id + len(result["runs"])
        rec["batch_index"] = gidx + 1
        rec["repeat"] = rep
        rec["candidate_source_key"] = mode
        rec["arm"] = arm["label"]
        if p.get("meta"):
            rec["prompt_meta"] = p["meta"]
        result["runs"].append(rec)
        if jsonl_path:                                          # O(1) append per run instead of rewriting the whole result
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if save_path and checkpoint_every and len(result["runs"]) % checkpoint_every == 0:
            meta["elapsed_s"] = round(prior_elapsed + time.perf_counter() - t0, 1)
            _safe_dump(result, save_path)
        session_done += 1
        elapsed_session = time.perf_counter() - t0
        done_now = len(result["runs"])
        eta = (elapsed_session / session_done) * max(0, total - done_now)
        best = (rec.get("best_result") or {}).get("rank")
        write_progress(progress_path, {
            "name": spec["name"], "total": total, "done": done_now, "elapsed_s": round(prior_elapsed + elapsed_session, 1),
            "eta_s": round(eta, 0), "finished": False, "error": rec.get("error"), "out": save_path,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last": {"arm": arm["label"], "mode": mode, "prompt": p["prompt"], "target": p["target"],
                     "baseline_rank": rec.get("baseline_rank"), "best_rank": best, "success": rec.get("success"),
                     "steps": len(rec.get("hybrid_details") or []), "time_s": rec.get("total_time_s")}})
        if progress_cb:
            progress_cb(done_now, total, rec)

    finalize_result(result)
    meta["elapsed_s"] = round(prior_elapsed + time.perf_counter() - t0, 1)
    if save_path:
        _safe_dump(result, save_path)
    write_progress(progress_path, {"name": spec["name"], "total": total, "done": len(result["runs"]), "elapsed_s": meta["elapsed_s"],
                                   "eta_s": 0, "finished": True, "error": None, "out": save_path,
                                   "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "last": None})
    return result


def finalize_result(result):
    """Fill paired_summary / aggregate / arm comparison from result['runs'] (also usable on a saved file)."""
    runs = result["runs"]
    arms = []
    for r in runs:
        a = r.get("arm", "default")
        if a not in arms:
            arms.append(a)
    per_arm = {}
    for a in arms:
        rows, agg = build_paired_summary([r for r in runs if r.get("arm", "default") == a])
        per_arm[a] = {"paired_summary": rows, "aggregate": agg}
    first = arms[0] if arms else "default"
    result["paired_summary"] = per_arm.get(first, {}).get("paired_summary", [])
    result["aggregate"] = per_arm.get(first, {}).get("aggregate", {})
    result["per_arm"] = per_arm
    if len(arms) > 1:
        table, per_arm_mode, vs_ref = build_arm_comparison(runs)
        result["arm_comparison"] = {"reference": arms[0], "arms": arms, "cells": table,
                                    "per_arm_mode": per_arm_mode, "vs_reference": vs_ref}
    meta = result["meta"]
    meta["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["errors"] = sum(1 for r in runs if r.get("error"))
    return result
