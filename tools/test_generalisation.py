"""
Reworded-prompt test (docs/Research_Journal/30.md section 10): does an edit tuned on ONE prompt carry over?

For every test prompt the edit is tuned on the ORIGINAL prompt only (gradient descent, optionally with the additive edit).
That exact edit (the same per-feature multipliers, and for additive arms the same added amounts) is then applied, unchanged,
to
  * the fact's REWORDED prompts (CounterFact `paraphrase_prompts`): does the target go up there too?
  * NEARBY facts (`neighborhood_prompts`, other subjects with the same relation): does the output stay the same?
  * 20 unrelated neutral prompts (the runner's NEUTRAL_PROMPTS): same question.
The edit is applied by feature id (multipliers scale a feature wherever it fires; an added amount is added at the last position
of whatever prompt it is applied to), through the real hook, exactly as on the original prompt.

--targets true    the true answer (the hard set, as in Entry 29)
--targets random  the SAME prompts but a random unrelated word as the target (the control: an edit that "transfers" to the
                  reworded prompts of a RANDOM word carries no information about the fact)

Arms (same names as tools/compare_sweep_vs_gradient.py): gd, gd_add, gd_klall, gd_add_klall; overrides like gd_add@cap=0.25.

Metrics per arm. Original prompt: rank 1 rate. Reworded prompts: share where the target's rank improved, share that reached
rank 1, median rank before and after, mean log10 gain in rank. Nearby facts / neutral prompts: KL(clean || edited) over the whole
vocabulary, share where the top-1 word changed, share where the TARGET became the top-1 word (a leak).

    .venv\\Scripts\\python.exe tools/test_generalisation.py --n-per-band 2                     # smoke test
    .venv\\Scripts\\python.exe tools/test_generalisation.py --n-per-band 10                    # 40 prompts, about 20 minutes
    .venv\\Scripts\\python.exe tools/test_generalisation.py --n-per-band 10 --targets random   # the control
    .venv\\Scripts\\python.exe tools/test_generalisation.py --summarise-only DIR
"""
import argparse
import json
import math
import os
import random
import statistics as st
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch
import torch.nn.functional as F

import compare_sweep_vs_gradient as cmp

N_NEIGH_MAX = 10


def eval_prompt(model, sae, hook_name, make_hook, text, target_id, scale_map, add_map):
    """Rank / probability of the target and the whole-vocabulary KL, clean vs edited, at the last position of `text`."""
    toks = model.to_tokens(text)
    model.reset_hooks()
    with torch.no_grad():
        clean = F.log_softmax(model(toks)[0, -1], dim=-1)
        edited = F.log_softmax(model.run_with_hooks(toks, fwd_hooks=[(hook_name, make_hook(scale_map, add_map, sae))])[0, -1], dim=-1)
    model.reset_hooks()
    out = {"rank_clean": int((clean > clean[target_id]).sum().item()) + 1, "rank_edit": int((edited > edited[target_id]).sum().item()) + 1,
           "prob_clean": float(clean[target_id].exp().item()), "prob_edit": float(edited[target_id].exp().item()),
           "kl": float((clean.exp() * (clean - edited)).sum().item()),
           "top1_flip": int(clean.argmax().item() != edited.argmax().item()), "target_top1": int(edited.argmax().item() == target_id)}
    return out


def _texts(x):
    return [(t if isinstance(t, str) else t.get("prompt", "")) for t in (x or [])]


def run(a):
    from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
    from src.editing import build_clean_context, get_target_token_id
    from src.hybrid_runner import NEUTRAL_PROMPTS
    from src.hooks import make_scale_and_add_hook
    from src.gradient_editing import run_gradient_descent_edit

    device = get_default_device()
    model = load_base_model()
    layer = a.layer
    sae = load_sae_for_layer(layer=layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
    cases, skipped_multi, _ = cmp.load_cases(model, a.n_per_band, a.seed, a.split, "true")
    cf = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "datasets", "counterfact.json"), encoding="utf-8"))}
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    parsed = {x: cmp.parse_arm(x) for x in arms}

    if a.targets == "random":                                                    # same prompts, a random unrelated word as the target
        rng = random.Random(a.seed + 1)
        words = [i for i in range(model.cfg.d_vocab) if (s := model.to_string([i])).startswith(" ") and s[1:].isalpha() and s[1:].islower() and len(s) > 4]
        kept = []
        for c in cases:
            c = dict(c)
            c["target"] = model.to_string([rng.choice(words)])[1:]
            kept.append(c)
        cases = kept

    out_dir = a.out or os.path.join(ROOT, "outputs", "generalisation", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, "runs.jsonl")
    done = set()
    if a.resume and os.path.exists(jl):
        done = {json.loads(l)["case_id"] for l in open(jl, encoding="utf-8") if l.strip()}
    json.dump({"started": datetime.now().isoformat(), "args": vars(a), "n_cases": len(cases), "layer": layer, "device": device},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"{len(cases)} prompts [{a.split}, {a.targets} targets], arms {arms}, output {out_dir}", flush=True)

    t0 = time.perf_counter()
    for n, case in enumerate(cases, start=1):
        if case["case_id"] in done:
            continue
        row = dict(case)
        try:
            rec = cf[case["case_id"]]
            paras = [t for t in _texts(rec.get("paraphrase_prompts")) if t.strip()]
            neighs = [t for t in _texts(rec.get("neighborhood_prompts")) if t.strip()][:N_NEIGH_MAX]
            tid = get_target_token_id(model, " " + case["target"])
            model.reset_hooks()
            ctx = build_clean_context(model, sae, case["prompt"], tid)
            row["start_rank"] = ctx.clean_rank
            row["n_para"], row["n_neigh"] = len(paras), len(neighs)
            for arm in arms:
                base, cfg = parsed[arm]
                kw = {"top_n": a.top_n, "positions": "all", "steps": a.gd_steps, "kl_always": base.endswith("klall")}
                if "lamkl" in cfg:
                    kw["lam_kl"] = cfg["lamkl"]
                if "_add" in base:
                    kw.update(additive=True, add_top_m=int(cfg.get("m", a.add_m)), add_cap_factor=cfg.get("cap", a.add_cap_factor),
                              lam_add=cfg.get("lam", a.lam_add), add_lr=cfg.get("addlr", a.add_lr))
                g = run_gradient_descent_edit(model, sae, ctx, tid, case["prompt"], layer, hook_name, **kw)
                scale_map = {f: 1.0 + x for f, x in zip(g["fids"], g["a"])}
                add_map = g["added"]
                main = eval_prompt(model, sae, hook_name, make_scale_and_add_hook, case["prompt"], tid, scale_map, add_map)
                row[arm] = {
                    "main": {"rank": main["rank_edit"], "tuning_rank": g["real_path"]["rank"], "kl": main["kl"], "prob": main["prob_edit"]},
                    "edit_size_frac_norm": g["edit_size_frac_norm"], "n_added": sum(1 for v in add_map.values() if g.get("add_cap") and v >= 0.01 * g["add_cap"]),
                    "para": [eval_prompt(model, sae, hook_name, make_scale_and_add_hook, t, tid, scale_map, add_map) for t in paras],
                    "neigh": [eval_prompt(model, sae, hook_name, make_scale_and_add_hook, t, tid, scale_map, add_map) for t in neighs],
                    "neutral": [eval_prompt(model, sae, hook_name, make_scale_and_add_hook, t, tid, scale_map, add_map) for t in NEUTRAL_PROMPTS],
                }
        except Exception as e:
            import traceback
            model.reset_hooks()
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        msg = row.get("error") or " | ".join(
            f"{arm}: main #{row[arm]['main']['rank']}, reworded {sum(1 for p in row[arm]['para'] if p['rank_edit'] < p['rank_clean'])}/{len(row[arm]['para'])} improved "
            f"({sum(1 for p in row[arm]['para'] if p['rank_edit'] == 1)} rank-1), neighbours KL {st.mean([p['kl'] for p in row[arm]['neigh']]) if row[arm]['neigh'] else float('nan'):.2f}"
            for arm in arms)
        print(f"[{n}/{len(cases)}] {row['band']:<8} {case['prompt'][:34]!r:<38} -> {case['target']!r:<12} {msg}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    summarise(out_dir)


# ---------------------------------------------------------------------------------------------------- summary
def _med(x):
    return st.median(x) if x else float("nan")


def _mean(x):
    return sum(x) / len(x) if x else float("nan")


def summarise(out_dir):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "runs.jsonl"), encoding="utf-8") if l.strip()]
    errs = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    arms = [k for k in rows[0] if isinstance(rows[0][k], dict) and "main" in rows[0][k]] if rows else []
    targets = json.load(open(os.path.join(out_dir, "meta.json")))["args"].get("targets")
    S = {"n": len(rows), "errors": len(errs), "targets": targets, "arms": {}}
    lines = [f"Reworded-prompt test: {len(rows)} prompts ({len(errs)} errors), targets = {targets}  [{out_dir}]", ""]
    h1 = f"{'arm':<26}{'orig. rank-1':>13}{'edit size':>10} | {'reworded: improved':>19}{'rank-1':>8}{'med rank':>16}{'log10 gain':>11} | {'nearby: KL':>11}{'flips':>7}{'target top-1':>13} | {'neutral KL':>11}{'flips':>7}"
    lines += [h1, "-" * len(h1)]
    for arm in arms:
        R = [r[arm] for r in rows]
        P = [p for r in R for p in r["para"]]
        N = [p for r in R for p in r["neigh"]]
        U = [p for r in R for p in r["neutral"]]
        s = {
            "orig_rank1": sum(1 for r in R if r["main"]["rank"] == 1), "n": len(R), "median_edit_size": _med([r["edit_size_frac_norm"] for r in R]),
            "para_n": len(P), "para_improved": _mean([1.0 if p["rank_edit"] < p["rank_clean"] else 0.0 for p in P]),
            "para_rank1": _mean([1.0 if p["rank_edit"] == 1 else 0.0 for p in P]),
            "para_median_rank_clean": _med([p["rank_clean"] for p in P]), "para_median_rank_edit": _med([p["rank_edit"] for p in P]),
            "para_mean_log10_gain": _mean([math.log10(p["rank_clean"] / p["rank_edit"]) for p in P]),
            "neigh_n": len(N), "neigh_mean_kl": _mean([p["kl"] for p in N]), "neigh_median_kl": _med([p["kl"] for p in N]),
            "neigh_flip": _mean([p["top1_flip"] for p in N]), "neigh_target_top1": _mean([p["target_top1"] for p in N]),
            "neutral_mean_kl": _mean([p["kl"] for p in U]), "neutral_flip": _mean([p["top1_flip"] for p in U]),
            "main_mismatch": sum(1 for r in R if r["main"]["rank"] != r["main"]["tuning_rank"]),
        }
        # transfer given the original prompt was solved
        solved = [r for r in R if r["main"]["rank"] == 1]
        Ps = [p for r in solved for p in r["para"]]
        s["solved_para_improved"] = _mean([1.0 if p["rank_edit"] < p["rank_clean"] else 0.0 for p in Ps])
        s["solved_para_rank1"] = _mean([1.0 if p["rank_edit"] == 1 else 0.0 for p in Ps])
        S["arms"][arm] = s
        lines.append(f"{arm:<26}{s['orig_rank1']:>8}/{s['n']:<4}{100 * s['median_edit_size']:>9.0f}% | {100 * s['para_improved']:>18.0f}%{100 * s['para_rank1']:>7.0f}%"
                     f"{s['para_median_rank_clean']:>9.0f} ->{s['para_median_rank_edit']:>4.0f}{s['para_mean_log10_gain']:>11.2f} | {s['neigh_mean_kl']:>11.3f}{100 * s['neigh_flip']:>6.0f}%{100 * s['neigh_target_top1']:>12.0f}% | "
                     f"{s['neutral_mean_kl']:>11.3f}{100 * s['neutral_flip']:>6.0f}%")
    lines += ["", "reworded = CounterFact paraphrase prompts of the same fact, edit tuned on the ORIGINAL prompt only; 'improved' = target rank better than unedited;",
              "nearby = other subjects with the same relation (should not change); 'target top-1' = share of nearby prompts where the TARGET became the top word (a leak);",
              "KL = KL(clean || edited) over the whole vocabulary at the last position; neutral = 20 unrelated prompts."]
    lines += ["", "Given the ORIGINAL prompt reached rank 1 (does the fix carry over to reworded prompts?):"]
    for arm in arms:
        s = S["arms"][arm]
        lines.append(f"  {arm:<26} reworded improved {100 * s['solved_para_improved']:.0f}%, reached rank 1 {100 * s['solved_para_rank1']:.0f}%")
    if "gd" in arms and len(arms) > 1:
        from scipy.stats import wilcoxon
        lines += ["", "PAIRED vs gd (per prompt; Wilcoxon signed-rank):"]
        for arm in arms:
            if arm == "gd":
                continue
            d_gain = [_mean([math.log10(p["rank_clean"] / p["rank_edit"]) for p in r[arm]["para"]]) - _mean([math.log10(p["rank_clean"] / p["rank_edit"]) for p in r["gd"]["para"]]) for r in rows if r[arm]["para"] and r["gd"]["para"]]
            d_kl = [_mean([p["kl"] for p in r[arm]["neigh"]]) - _mean([p["kl"] for p in r["gd"]["neigh"]]) for r in rows if r[arm]["neigh"] and r["gd"]["neigh"]]
            pg = float(wilcoxon(d_gain).pvalue) if len(d_gain) >= 6 and any(x != 0 for x in d_gain) else float("nan")
            pk = float(wilcoxon(d_kl).pvalue) if len(d_kl) >= 6 and any(x != 0 for x in d_kl) else float("nan")
            lines.append(f"  {arm:<26} reworded log10 gain, median difference {_med(d_gain):+.2f} (p = {pg:.3g}); nearby KL, median difference {_med(d_kl):+.3f} (p = {pk:.3g})")
    mism = sum(S["arms"][a_]["main_mismatch"] for a_ in arms)
    lines += ["", f"Consistency: the original prompt's rank through the hook differs from the tuning-pass rank on {mism} prompt/arm pairs (0 expected)."]
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    text = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write(text)
    print("\n" + text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-band", type=int, default=10)
    ap.add_argument("--arms", default="gd,gd_add@cap=1.0,gd_add@cap=0.25", help="comma-separated arms (see the header)")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--targets", choices=["true", "random"], default="true")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--gd-steps", type=int, default=100)
    ap.add_argument("--add-m", type=int, default=200)
    ap.add_argument("--add-cap-factor", type=float, default=1.0)
    ap.add_argument("--lam-add", type=float, default=0.005)
    ap.add_argument("--add-lr", type=float, default=0.3)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarise-only", default=None)
    a = ap.parse_args()
    if a.summarise_only:
        summarise(a.summarise_only)
        return
    run(a)


if __name__ == "__main__":
    main()
