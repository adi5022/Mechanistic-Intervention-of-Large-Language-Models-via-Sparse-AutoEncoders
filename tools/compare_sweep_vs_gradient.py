"""
Equal-budget comparison: the fixed mute / boost sweep vs gradient descent with one multiplier per feature, and (Entry 30)
gradient descent with an ADDITIVE edit on silent features.
(docs/Research_Journal/28.md section 8, 29.md, 30.md.)

Held-out CounterFact prompts, stratified by starting-rank band. Every arm runs through the REAL model on the SAME prompt with
the SAME candidate set (Top N 200, all prompt positions):

  sweep_ref     the real sweep (src/hybrid_runner.py), mute 0.6 / boost 0.5 (the original settings; reach 0.4x to 1.5x)
  sweep_wide    the real sweep, mute 1.0 / boost 2.0 (reach 0x to 3x = the SAME allowed range as gradient descent)
  gd            gradient descent, one multiplier per feature in 0..3 (src/gradient_editing.py), the app's default settings
  gd_klall      gd with the side-effect (KL) penalty charged on every step, not only once the target leads
  gd_add        gd + the additive edit: silent features switched on at the last position (see gradient_editing.py)
  gd_add_klall  gd_add with the always-on KL penalty

An arm name may carry overrides: `gd_add@cap=0.25:lam=0.02` (keys: cap = cap factor, lam = sparsity weight, m = silent
candidates, addlr, lamkl). Arms are separated by commas.

Every arm's final edit is turned into a per-feature scale map plus (for additive arms) an add map and measured by ONE function
(`measure`), so the numbers are comparable: rank / probability / top-1 through the real hook; KL on the same prompt (all tokens
except target and original top-1, renormalised, last position) and the full-vocabulary KL; number of features changed, added;
edit size at the last position as a share of the residual norm (multiplier part + additive part); KL and top-1 flips of the
same edit on 20 unrelated prompts. Time is each method's own model-compute time.

Prompt sets:  --split test (test_seen + test_unseen, 300) or val (150);  --targets true (the true answer; the hard set) or new
(the CounterFact counterfactual target: very low probability, the "Hong Kong" case; start rank computed here).
--baseline-from DIR reuses the arms already computed for the same prompts (e.g. the Entry 29 pack) so only new arms run.
Results are appended one prompt at a time (jsonl), so a run can be stopped and resumed (--resume --out DIR).

    .venv\\Scripts\\python.exe tools/compare_sweep_vs_gradient.py --n-per-band 2                         # smoke test
    .venv\\Scripts\\python.exe tools/compare_sweep_vs_gradient.py --n-per-band 0                          # Entry 29: all 300
    .venv\\Scripts\\python.exe tools/compare_sweep_vs_gradient.py --split val --arms gd,gd_add@cap=0.25  # tuning on validation
    .venv\\Scripts\\python.exe tools/compare_sweep_vs_gradient.py --summarise-only DIR                   # re-print a summary
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

BANDS = [("2-5", 2, 5), ("6-20", 6, 20), ("21-100", 21, 100), ("101-1000", 101, 1000), (">1000", 1001, 10 ** 9)]
KNOWN_ARMS = ["sweep_ref", "sweep_wide", "gd", "gd_klall", "gd_add", "gd_add_klall"]
SWEEP_CFG = {"sweep_ref": {"mute_strength": 0.6, "boost_strength": 0.5},
             "sweep_wide": {"mute_strength": 1.0, "boost_strength": 2.0}}
OVERRIDE_KEYS = {"cap", "lam", "m", "addlr", "lamkl"}
ADD_REPORT_FRAC = 0.01


def parse_arm(name):
    base, _, ov = name.partition("@")
    if base not in KNOWN_ARMS:
        raise ValueError(f"unknown arm {base!r}; known: {KNOWN_ARMS}")
    cfg = {}
    for kv in filter(None, ov.split(":")):
        k, v = kv.split("=")
        if k not in OVERRIDE_KEYS:
            raise ValueError(f"unknown override {k!r}; allowed: {sorted(OVERRIDE_KEYS)}")
        cfg[k] = float(v)
    return base, cfg


def band_of(rank):
    for name, lo, hi in BANDS:
        if lo <= rank <= hi:
            return name
    return None


def load_cases(model, n_per_band, seed, split_name, targets):
    from src.editing import get_target_token_id
    cf = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "datasets", "counterfact.json"), encoding="utf-8"))}
    ranks = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "outputs", "counterfact_all_ranks.json"), encoding="utf-8"))}
    split = json.load(open(os.path.join(ROOT, "data", "counterfact_split.json"), encoding="utf-8"))
    parts = ("test_seen", "test_unseen") if split_name == "test" else ("val",)
    pool = {name: [] for name, _, _ in BANDS}
    skipped_multi = skipped_top = 0
    for part in parts:
        for cid in split[part]:
            rec = cf[cid]["requested_rewrite"]
            tgt = rec["target_true" if targets == "true" else "target_new"]["str"]
            if int(model.to_tokens(" " + tgt, prepend_bos=False).numel()) != 1:
                skipped_multi += 1
                continue
            prompt = rec["prompt"].format(rec["subject"])
            if targets == "true":
                rank = ranks[cid]["rank"]
            else:                                                                  # start rank of the counterfactual target
                tid = get_target_token_id(model, " " + tgt)
                with torch.no_grad():
                    lp = F.log_softmax(model(model.to_tokens(prompt))[0, -1], dim=-1)
                rank = int((lp > lp[tid]).sum().item()) + 1
            if rank == 1:
                skipped_top += 1
                continue
            b = band_of(rank)
            if b:
                pool[b].append({"case_id": cid, "part": part, "band": b, "start_rank_cached": rank, "prompt": prompt, "target": tgt,
                                "relation_id": rec["relation_id"], "targets": targets})
    rng = random.Random(seed)
    cases = []
    for name, _, _ in BANDS:
        rng.shuffle(pool[name])
        cases += pool[name] if n_per_band <= 0 else pool[name][:n_per_band]
    return cases, skipped_multi, skipped_top


def make_measure(model, sae, hook_name, NEUTRAL_PROMPTS, _score):
    from src.hooks import make_scale_and_add_hook

    def run_hooked(tokens, scale_map, add_map):
        model.reset_hooks()
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_scale_and_add_hook(scale_map, add_map, sae))])
        model.reset_hooks()
        return logits[0, -1]

    def collateral(scale_map, add_map):
        kls, flips = [], 0
        for pr in NEUTRAL_PROMPTS:
            toks = model.to_tokens(pr)
            model.reset_hooks()
            with torch.no_grad():
                clean = torch.log_softmax(model(toks)[0, -1], dim=-1)
                model.add_hook(hook_name, make_scale_and_add_hook(scale_map, add_map, sae))
                edited = torch.log_softmax(model(toks)[0, -1], dim=-1)
            model.reset_hooks()
            kls.append(float(torch.sum(clean.exp() * (clean - edited)).item()))
            flips += int(clean.argmax().item() != edited.argmax().item())
        n = max(1, len(NEUTRAL_PROMPTS))
        return {"mean_kl": sum(kls) / n, "max_kl": max(kls), "top1_flip_rate": flips / n}

    def measure(ctx, target_id, blocker_id, clean_last, scale_map, add_map=None, cap=None):
        """One yardstick for every arm: the final edit as {feature id: multiplier} and {feature id: added amount}."""
        add_map = add_map or {}
        last = run_hooked(ctx.tokens, scale_map, add_map) if (scale_map or add_map) else clean_last
        rank, prob, kl, margin = _score(last, clean_last, target_id, blocker_id)
        lp, lc = F.log_softmax(last, dim=-1), F.log_softmax(clean_last, dim=-1)
        kl_full = float((lc.exp() * (lc - lp)).sum().item())
        ids = list(scale_map)
        n_changed = n_muted = n_boosted = 0
        with torch.no_grad():
            d_last = torch.zeros(ctx.resid_all.shape[-1], device=last.device)
            if ids:
                s = torch.tensor([scale_map[i] for i in ids], device=last.device, dtype=torch.float32)
                acts_last = sae.encode(ctx.resid_all[0, -1:])[0, ids]
                d_last = d_last + ((s - 1.0) * acts_last) @ sae.W_dec[ids]
                n_changed, n_muted, n_boosted = int(((s - 1.0).abs() > 0.05).sum()), int((s < 0.95).sum()), int((s > 1.05).sum())
            if add_map:
                aid = list(add_map)
                amt = torch.tensor([add_map[i] for i in aid], device=last.device, dtype=torch.float32)
                d_last = d_last + amt @ sae.W_dec[aid]
            size = float(d_last.norm().item() / ctx.resid_all[0, -1].norm().item())
        n_added = sum(1 for v in add_map.values() if cap and v >= ADD_REPORT_FRAC * cap)
        col = collateral(scale_map, add_map) if (scale_map or add_map) else {"mean_kl": 0.0, "max_kl": 0.0, "top1_flip_rate": 0.0}
        out = {"rank": rank, "prob": float(prob.item()), "kl": float(kl.item()), "kl_full": kl_full, "top1_id": int(last.argmax().item()),
               "n_changed": n_changed, "n_muted": n_muted, "n_boosted": n_boosted, "n_added": n_added,
               "edit_size_frac_norm": size, "collateral": col}
        if add_map:
            out["added"] = {str(k): round(v, 4) for k, v in add_map.items() if v >= 1e-4 * (cap or 1.0)}
            out["add_total_over_cap"] = sum(add_map.values()) / cap if cap else None
        return out
    return measure


def run(a):
    from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
    from src.device_utils import sync_device
    from src.editing import build_clean_context, get_target_token_id
    from src.hybrid_runner import run_hybrid_sweep, NEUTRAL_PROMPTS
    from src.hooks import build_scale_map_graded
    from src.gradient_editing import run_gradient_descent_edit, _score

    device = get_default_device()
    model = load_base_model()
    layer = a.layer
    sae = load_sae_for_layer(layer=layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
    cases, skipped_multi, skipped_top = load_cases(model, a.n_per_band, a.seed, a.split, a.targets)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    parsed = {x: parse_arm(x) for x in arms}
    measure = make_measure(model, sae, hook_name, NEUTRAL_PROMPTS, _score)

    baseline = {}
    if a.baseline_from:
        for l in open(os.path.join(a.baseline_from, "runs.jsonl"), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                baseline[r["case_id"]] = r

    out_dir = a.out or os.path.join(ROOT, "outputs", "sweep_vs_gradient", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, "runs.jsonl")
    done = set()
    if a.resume and os.path.exists(jl):
        done = {json.loads(l)["case_id"] for l in open(jl, encoding="utf-8") if l.strip()}
    json.dump({"started": datetime.now().isoformat(), "args": vars(a), "n_cases": len(cases), "skipped_multi_token": skipped_multi,
               "skipped_already_rank1": skipped_top, "layer": layer, "device": device, "sweep_cfg": SWEEP_CFG},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"{len(cases)} prompts [{a.split}, {a.targets} targets] ({skipped_multi} multi-token, {skipped_top} already rank 1 skipped), "
          f"arms {arms}, baseline from {a.baseline_from}, output {out_dir}", flush=True)

    t0 = time.perf_counter()
    for n, case in enumerate(cases, start=1):
        if case["case_id"] in done:
            continue
        row = dict(case)
        try:
            tid = get_target_token_id(model, " " + case["target"])
            model.reset_hooks()
            ctx = build_clean_context(model, sae, case["prompt"], tid)
            blocker = int(torch.argmax(ctx.clean_probs).item())
            with torch.no_grad():
                clean_last = torch.log(ctx.clean_probs.clamp_min(1e-30))
            row["start_rank"] = ctx.clean_rank
            row["blocker"] = model.to_string([blocker])
            b = baseline.get(case["case_id"])
            if b and b.get("prompt") == case["prompt"] and b.get("target") == case["target"]:
                for k, v in b.items():
                    if isinstance(v, dict) and "rank" in v and k not in parsed:
                        row[k] = v
                row["baseline_reused"] = True
            for arm in arms:
                base, cfg = parsed[arm]
                add_map, cap, extra = {}, None, {}
                if base.startswith("gd"):
                    kw = {"top_n": a.top_n, "positions": "all", "steps": a.gd_steps, "kl_always": base.endswith("klall")}
                    if "lamkl" in cfg:
                        kw["lam_kl"] = cfg["lamkl"]
                    if "_add" in base:
                        kw.update(additive=True, add_top_m=int(cfg.get("m", a.add_m)), add_cap_factor=cfg.get("cap", a.add_cap_factor),
                                  lam_add=cfg.get("lam", a.lam_add), add_lr=cfg.get("addlr", a.add_lr))
                    sync_device(device)
                    t1 = time.perf_counter()
                    g = run_gradient_descent_edit(model, sae, ctx, tid, case["prompt"], layer, hook_name, **kw)
                    sync_device(device)
                    secs = time.perf_counter() - t1
                    scale_map = {f: 1.0 + x for f, x in zip(g["fids"], g["a"])}
                    add_map, cap = g["added"], g.get("add_cap") if "_add" in base else None
                    extra = {"n_candidates": g["n_candidates"], "best_step": g["best_step"]}
                    if "add_summary" in g:
                        extra.update({"add_summary": g["add_summary"], "add_cap": g["add_cap"], "add_candidates": g["add_candidates"]})
                else:
                    scfg = {"top_n": a.top_n, "collateral": False, "record_detail": "compact", **SWEEP_CFG[base]}
                    rec = run_hybrid_sweep(model, sae, hook_name, layer, device, case["prompt"], case["target"], scfg, candidate_source="all")
                    secs = rec["timing"]["model_compute_s"]
                    best = rec["best_result"]
                    scale_map = (build_scale_map_graded(best.get("mute_strengths", {}), best.get("boost_strengths", {}))
                                 if best["step"] > 0 else {})
                    extra = {"sweep_reported_rank": best["rank"], "sweep_steps": rec["run_summary"]["sweep_steps"],
                             "sweep_rounds": rec["run_summary"]["rounds_run"], "stop_reason": rec["run_summary"]["stop_reason"]}
                m = measure(ctx, tid, blocker, clean_last, scale_map, add_map, cap)
                m.update(extra)
                m["time_s"] = round(secs, 3)
                m["top1"] = model.to_string([m.pop("top1_id")])
                row[arm] = m
        except Exception as e:                                                   # keep going; record the failure
            import traceback
            model.reset_hooks()
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        el = time.perf_counter() - t0
        msg = row.get("error") or " | ".join(f"{arm} {row['start_rank']}->{row[arm]['rank']} KL {row[arm]['kl']:.2f} sz {row[arm]['edit_size_frac_norm']:.2f}" for arm in arms)
        print(f"[{n}/{len(cases)}] {row['band']:<8} {case['prompt'][:38]!r:<42} {msg}  ({el / 60:.1f} min)", flush=True)
    summarise(out_dir)


# ---------------------------------------------------------------------------------------------------- summary
def _mcnemar(a_ok, b_ok):
    from scipy.stats import binomtest
    only_a = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    only_b = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def _med(x):
    return statistics.median(x) if x else float("nan")


def _mean(x):
    return sum(x) / len(x) if x else float("nan")


def _detect_arms(rows):
    seen = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, dict) and "rank" in v and "edit_size_frac_norm" in v and k not in seen:
                seen.append(k)
    order = {k: i for i, k in enumerate(KNOWN_ARMS)}
    return sorted(seen, key=lambda k: (order.get(k.partition("@")[0], 99), k))


def summarise(out_dir, arms=None):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "runs.jsonl"), encoding="utf-8") if l.strip()]
    errs = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    arms = arms or _detect_arms(rows)
    rows = [r for r in rows if all(a in r for a in arms)]                       # prompts with every arm present
    S = {"n": len(rows), "errors": len(errs), "arms": {}, "paired": {}, "per_band": {}}
    lines = [f"Comparison: {len(rows)} prompts ({len(errs)} errors)  [{out_dir}]", ""]
    w = max(12, max((len(a) for a in arms), default=12) + 1)
    hdr = f"{'arm':<{w}}{'rank-1':>12}{'median KL':>11}{'mean KL':>9}{'KL (succ.)':>12}{'features':>10}{'added':>7}{'edit size':>11}{'(median)':>9}{'neutral KL':>12}{'flips':>7}{'time s':>8}"
    lines += [hdr, "-" * len(hdr)]
    for arm in arms:
        R = [r[arm] for r in rows]
        succ = [x for x in R if x["rank"] == 1]
        S["arms"][arm] = {
            "rank1": len(succ), "n": len(R), "rank1_rate": len(succ) / max(len(R), 1),
            "median_rank_end": _med([x["rank"] for x in R]),
            "median_kl_all": _med([x["kl"] for x in R]), "mean_kl_all": _mean([x["kl"] for x in R]),
            "mean_kl_successes": _mean([x["kl"] for x in succ]), "median_kl_successes": _med([x["kl"] for x in succ]),
            "mean_features_changed": _mean([x["n_changed"] for x in R]), "mean_added": _mean([x.get("n_added", 0) for x in R]),
            "mean_edit_size_frac_norm": _mean([x["edit_size_frac_norm"] for x in R]), "median_edit_size_frac_norm": _med([x["edit_size_frac_norm"] for x in R]),
            "mean_neutral_kl": _mean([x["collateral"]["mean_kl"] for x in R]),
            "mean_neutral_flip_rate": _mean([x["collateral"]["top1_flip_rate"] for x in R]),
            "mean_time_s": _mean([x["time_s"] for x in R]), "median_time_s": _med([x["time_s"] for x in R]),
            "mean_target_prob_successes": _mean([x["prob"] for x in succ])}
        s = S["arms"][arm]
        lines.append(f"{arm:<{w}}{s['rank1']:>6}/{s['n']:<4}{s['median_kl_all']:>11.3f}{s['mean_kl_all']:>9.3f}{s['mean_kl_successes']:>12.3f}"
                     f"{s['mean_features_changed']:>10.1f}{s['mean_added']:>7.1f}{s['mean_edit_size_frac_norm']:>11.3f}{s['median_edit_size_frac_norm']:>9.3f}"
                     f"{s['mean_neutral_kl']:>12.4f}{s['mean_neutral_flip_rate']:>7.2f}{s['mean_time_s']:>8.1f}")
    lines += ["", "median KL = same-prompt KL over all prompts (failed ones included); KL (succ.) = mean over prompts that reached rank 1;",
              "features = multipliers moved more than 0.05 from 1; added = silent features given at least 1% of the cap;",
              "edit size = ||change|| / ||residual|| at the last position (multiplier part + additive part); neutral KL / flips = the same edit on 20 unrelated prompts."]
    pairs = []
    if "gd" in arms:
        pairs += [(a_, "gd") for a_ in arms if a_ != "gd"]
    if "sweep_ref" in arms and "sweep_wide" in arms:
        pairs.append(("sweep_ref", "sweep_wide"))
    lines += ["", "PAIRED (exact McNemar on reaching rank 1; KL on prompts where BOTH arms reached rank 1):"]
    for A, B in pairs:
        ok_a = [r[A]["rank"] == 1 for r in rows]
        ok_b = [r[B]["rank"] == 1 for r in rows]
        oa, ob, p = _mcnemar(ok_a, ok_b)
        both = [r for r in rows if r[A]["rank"] == 1 and r[B]["rank"] == 1]
        d = [r[A]["kl"] - r[B]["kl"] for r in both]
        ds = [r[A]["edit_size_frac_norm"] - r[B]["edit_size_frac_norm"] for r in both]
        wil = None
        if len(d) >= 6 and any(x != 0 for x in d):
            from scipy.stats import wilcoxon
            wil = float(wilcoxon(d).pvalue)
        S["paired"][f"{A}_vs_{B}"] = {"only_" + A: oa, "only_" + B: ob, "mcnemar_p": p, "both_rank1": len(both),
                                      "median_kl_diff_A_minus_B": _med(d), f"{A}_lower_kl": sum(1 for x in d if x < 0), "wilcoxon_p": wil,
                                      "median_edit_size_diff_A_minus_B": _med(ds)}
        lines.append(f"  {A} vs {B}: only {A} {oa}, only {B} {ob} (p = {p:.4g}); both reached rank 1 on {len(both)}: median KL diff ({A} - {B}) "
                     f"{_med(d):+.3f}, {A} lower on {sum(1 for x in d if x < 0)}" + (f" (Wilcoxon p = {wil:.4g})" if wil is not None else "")
                     + f"; median edit-size diff {_med(ds):+.3f}")
    lines += ["", "PER STARTING-RANK BAND (rank-1 count / n):"]
    for name, _, _ in BANDS:
        sub = [r for r in rows if r["band"] == name]
        if not sub:
            continue
        S["per_band"][name] = {arm: [sum(1 for r in sub if r[arm]["rank"] == 1), len(sub)] for arm in arms}
        lines.append(f"  {name:<9}" + "  ".join(f"{arm} {S['per_band'][name][arm][0]}/{len(sub)}" for arm in arms))
    mism = [r for r in rows for arm in arms if r[arm].get("sweep_reported_rank") not in (None, r[arm]["rank"])]
    lines += ["", f"Consistency: sweep-reported best rank differs from the re-measured rank on {len(mism)} prompt/arm pairs "
                  "(0 expected; a difference means the final edit was not reproduced exactly)."]
    S["consistency_mismatches"] = len(mism)
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    text = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write(text)
    print("\n" + text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-band", type=int, default=15, help="prompts per starting-rank band (0 = all of the chosen split)")
    ap.add_argument("--arms", default="sweep_ref,sweep_wide,gd", help="comma-separated arms to RUN (see the header)")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--targets", choices=["true", "new"], default="true", help="true answer (hard set) or the counterfactual target")
    ap.add_argument("--baseline-from", default=None, help="folder with a runs.jsonl for the same prompts: its arms are reused, not rerun")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--gd-steps", type=int, default=100)
    ap.add_argument("--add-m", type=int, default=200, help="silent candidates for additive arms")
    ap.add_argument("--add-cap-factor", type=float, default=1.0, help="cap = this x the loudest active feature on the prompt")
    ap.add_argument("--lam-add", type=float, default=0.005, help="sparsity weight of the additive amounts")
    ap.add_argument("--add-lr", type=float, default=0.3)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="output folder (default outputs/sweep_vs_gradient/<time>)")
    ap.add_argument("--resume", action="store_true", help="with --out: skip prompts already in runs.jsonl")
    ap.add_argument("--summarise-only", default=None, help="folder of a finished run: only re-print the summary")
    a = ap.parse_args()
    if a.summarise_only:
        summarise(a.summarise_only)
        return
    run(a)


if __name__ == "__main__":
    main()
