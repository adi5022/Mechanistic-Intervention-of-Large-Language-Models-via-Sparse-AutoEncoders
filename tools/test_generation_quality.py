"""
Tests for the two risks raised in an outside critique (docs/Research_Journal/30.md section 14):

 A. SEQUENTIAL COLLAPSE: after the edit, does the model still produce coherent text, or does it repeat / degenerate?
    For every prompt the edit is tuned on the original prompt (as always), then the model continues for --gen-tokens tokens,
    greedily, in three ways: no edit, edit with the multipliers kept on while generating, edit applied to the prompt only
    (the additive part is always added once, at the prompt's last position). Measured on the generated token ids:
      distinct-2        share of distinct bigrams (1.0 = no repetition)
      loop              a 3-gram repeated 3 or more times (degeneration)
      target repeats    how many times the target word appears AFTER its first appearance
      tail perplexity   perplexity, under the UNEDITED model, of the continuation after its first token, given the prompt and
                        that first token (a coherence score the edit cannot game: lower = the text still reads as likely English)
 B. CONTEXT SENSITIVITY: does the edit force the target even when the context says not to say it? The same edit is applied
    unchanged to the prompt preceded by an explicit instruction ("Do not say the word X.") or a statement ("X is the wrong
    answer."). GPT-2 small is a base model and mostly ignores such instructions even without an edit, and mentioning a word
    raises its probability (copying), so the fair comparison is edited vs unedited on the SAME prefixed prompt.

--targets true    the true answer;  --targets random   a random unrelated word (control: an edit that forces a meaningless word
                  tells us about the strength of the edit, not about the fact)

    .venv\\Scripts\\python.exe tools/test_generation_quality.py --n-per-band 1                     # smoke test
    .venv\\Scripts\\python.exe tools/test_generation_quality.py --n-per-band 10                    # 40 prompts
    .venv\\Scripts\\python.exe tools/test_generation_quality.py --n-per-band 10 --targets random
    .venv\\Scripts\\python.exe tools/test_generation_quality.py --summarise-only DIR
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

TEMPLATES = {
    "plain": "{prompt}",
    "do_not_say": "Do not say the word{target}. {prompt}",
    "wrong_answer": "{target0} is the wrong answer. {prompt}",
}


def text_metrics(ids, target_id):
    n = len(ids)
    bigrams = list(zip(ids, ids[1:]))
    trigrams = list(zip(ids, ids[1:], ids[2:]))
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else float("nan")
    loop = int(any(trigrams.count(t) >= 3 for t in set(trigrams))) if trigrams else 0
    first = ids.index(target_id) if target_id in ids else None
    repeats = (ids[first + 1:].count(target_id)) if first is not None else 0
    return {"n": n, "distinct2": d2, "loop": loop, "target_in_text": int(first is not None), "target_repeats": repeats}


def tail_ppl(model, prompt, ids):
    """Perplexity under the UNEDITED model of ids[1:] given the prompt and ids[0] (teacher forced)."""
    if len(ids) < 2:
        return float("nan")
    toks = model.to_tokens(prompt)
    seq = torch.cat([toks, torch.tensor([ids], device=toks.device)], dim=1)
    model.reset_hooks()
    with torch.no_grad():
        lp = F.log_softmax(model(seq)[0], dim=-1)
    P = toks.shape[1]
    # logits at position P-1+j predict ids[j]; the tail is j = 1..len-1
    nll = [-float(lp[P - 1 + j, ids[j]]) for j in range(1, len(ids))]
    return math.exp(sum(nll) / len(nll))


def eval_ctx(model, sae, hook_name, make_hook, text, target_id, scale_map, add_map):
    toks = model.to_tokens(text)
    model.reset_hooks()
    with torch.no_grad():
        clean = F.log_softmax(model(toks)[0, -1], dim=-1)
        edited = F.log_softmax(model.run_with_hooks(toks, fwd_hooks=[(hook_name, make_hook(scale_map, add_map, sae))])[0, -1], dim=-1)
    model.reset_hooks()
    return {"rank_clean": int((clean > clean[target_id]).sum()) + 1, "rank_edit": int((edited > edited[target_id]).sum()) + 1,
            "prob_clean": float(clean[target_id].exp()), "prob_edit": float(edited[target_id].exp()),
            "top1_clean": int(clean.argmax() == target_id), "top1_edit": int(edited.argmax() == target_id)}


def run(a):
    from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
    from src.editing import build_clean_context, get_target_token_id
    from src.hooks import make_scale_and_add_hook
    from src.gradient_editing import run_gradient_descent_edit, generate_greedy

    device = get_default_device()
    model = load_base_model()
    layer = a.layer
    sae = load_sae_for_layer(layer=layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
    cases, _, _ = cmp.load_cases(model, a.n_per_band, a.seed, a.split, "true")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    parsed = {x: cmp.parse_arm(x) for x in arms}
    if a.targets == "random":
        rng = random.Random(a.seed + 1)
        words = [i for i in range(model.cfg.d_vocab) if (s := model.to_string([i])).startswith(" ") and s[1:].isalpha() and s[1:].islower() and len(s) > 4]
        cases = [dict(c, target=model.to_string([rng.choice(words)])[1:]) for c in cases]

    out_dir = a.out or os.path.join(ROOT, "outputs", "generation_quality", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, "runs.jsonl")
    done = set()
    if a.resume and os.path.exists(jl):
        done = {json.loads(l)["case_id"] for l in open(jl, encoding="utf-8") if l.strip()}
    json.dump({"started": datetime.now().isoformat(), "args": vars(a), "n_cases": len(cases), "templates": TEMPLATES}, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"{len(cases)} prompts [{a.split}, {a.targets} targets], arms {arms}, {a.gen_tokens} tokens, output {out_dir}", flush=True)

    t0 = time.perf_counter()
    for n, case in enumerate(cases, start=1):
        if case["case_id"] in done:
            continue
        row = dict(case)
        try:
            tid = get_target_token_id(model, " " + case["target"])
            model.reset_hooks()
            ctx = build_clean_context(model, sae, case["prompt"], tid)
            row["start_rank"] = ctx.clean_rank
            base_steps, base_text = generate_greedy(model, sae, hook_name, case["prompt"], a.gen_tokens, target_id=tid)
            base_ids = [s["token_id"] for s in base_steps]
            row["baseline"] = {**text_metrics(base_ids, tid), "tail_ppl": tail_ppl(model, case["prompt"], base_ids), "text": base_text}
            for arm in arms:
                base, cfg = parsed[arm]
                kw = {"top_n": a.top_n, "positions": "all", "steps": a.gd_steps, "kl_always": base.endswith("klall")}
                if "_add" in base:
                    kw.update(additive=True, add_top_m=int(cfg.get("m", a.add_m)), add_cap_factor=cfg.get("cap", a.add_cap_factor),
                              lam_add=cfg.get("lam", a.lam_add), add_lr=cfg.get("addlr", a.add_lr))
                g = run_gradient_descent_edit(model, sae, ctx, tid, case["prompt"], layer, hook_name, **kw)
                smap = {f: 1.0 + x for f, x in zip(g["fids"], g["a"])}
                amap = g["added"]
                rec = {"main_rank": g["real_path"]["rank"], "edit_size_frac_norm": g["edit_size_frac_norm"], "gen": {}, "ctx": {}}
                for keep in (True, False):
                    steps, text = generate_greedy(model, sae, hook_name, case["prompt"], a.gen_tokens, smap, amap, keep_on=keep, target_id=tid)
                    ids = [s["token_id"] for s in steps]
                    rec["gen"]["keep_on" if keep else "prompt_only"] = {**text_metrics(ids, tid), "tail_ppl": tail_ppl(model, case["prompt"], ids), "text": text,
                                                                         "mean_chosen_prob": st.mean(s["prob"] for s in steps)}
                word = case["target"].strip()
                for name, tpl in TEMPLATES.items():
                    text = tpl.format(prompt=case["prompt"], target=" " + word, target0=word)
                    rec["ctx"][name] = eval_ctx(model, sae, hook_name, make_scale_and_add_hook, text, tid, smap, amap)
                row[arm] = rec
        except Exception as e:
            import traceback
            model.reset_hooks()
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        msg = row.get("error") or " | ".join(f"{arm}: #{row[arm]['main_rank']} -> {row[arm]['gen']['keep_on']['text'][:34]!r}" for arm in arms)
        print(f"[{n}/{len(cases)}] {row['band']:<8} {case['prompt'][:30]!r:<34} -> {case['target']!r:<11} base {row['baseline']['text'][:26]!r} || {msg}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    summarise(out_dir)


def _med(x):
    x = [v for v in x if v == v]
    return st.median(x) if x else float("nan")


def _mean(x):
    x = [v for v in x if v == v]
    return sum(x) / len(x) if x else float("nan")


def summarise(out_dir):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "runs.jsonl"), encoding="utf-8") if l.strip()]
    errs = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    arms = [k for k in rows[0] if isinstance(rows[0][k], dict) and "gen" in rows[0][k]] if rows else []
    targets = json.load(open(os.path.join(out_dir, "meta.json")))["args"].get("targets")
    S = {"n": len(rows), "errors": len(errs), "targets": targets, "A": {}, "B": {}}
    L = [f"Generation-quality and context-sensitivity test: {len(rows)} prompts ({len(errs)} errors), targets = {targets}  [{out_dir}]", ""]
    L.append("A. SEQUENTIAL COLLAPSE (continuations of the same length; tail perplexity under the UNEDITED model, lower = more fluent)")
    h = f"{'run':<34}{'distinct-2':>11}{'loop':>7}{'target in text':>16}{'target repeats':>16}{'tail ppl (median)':>19}"
    L += [h, "-" * len(h)]

    def line(label, ms):
        d2, lp, tt, tr, pp = _mean([m["distinct2"] for m in ms]), _mean([m["loop"] for m in ms]), _mean([m["target_in_text"] for m in ms]), \
            _mean([m["target_repeats"] for m in ms]), _med([m["tail_ppl"] for m in ms])
        L.append(f"{label:<34}{d2:>11.2f}{100 * lp:>6.0f}%{100 * tt:>15.0f}%{tr:>16.2f}{pp:>19.1f}")
        return {"distinct2": d2, "loop_rate": lp, "target_in_text": tt, "target_repeats": tr, "tail_ppl_median": pp}

    S["A"]["baseline"] = line("no edit", [r["baseline"] for r in rows])
    for arm in arms:
        for mode in ("keep_on", "prompt_only"):
            S["A"][f"{arm}/{mode}"] = line(f"{arm} ({'edit kept on' if mode == 'keep_on' else 'prompt only'})", [r[arm]["gen"][mode] for r in rows])
    L += ["", "B. CONTEXT SENSITIVITY (the SAME edit applied to the prompt with a prefix; target = the word the edit was tuned for)"]
    h = f"{'arm / context':<34}{'target top-1: unedited':>24}{'edited':>9}{'median rank: unedited':>24}{'edited':>9}{'median prob % unedited':>24}{'edited':>9}"
    L += [h, "-" * len(h)]
    for arm in arms:
        for name in TEMPLATES:
            c = [r[arm]["ctx"][name] for r in rows]
            S["B"][f"{arm}/{name}"] = {"top1_clean": _mean([x["top1_clean"] for x in c]), "top1_edit": _mean([x["top1_edit"] for x in c]),
                                       "rank_clean": _med([x["rank_clean"] for x in c]), "rank_edit": _med([x["rank_edit"] for x in c]),
                                       "prob_clean": _med([x["prob_clean"] for x in c]), "prob_edit": _med([x["prob_edit"] for x in c])}
            s_ = S["B"][f"{arm}/{name}"]
            L.append(f"{arm + ' / ' + name:<34}{100 * s_['top1_clean']:>23.0f}%{100 * s_['top1_edit']:>8.0f}%{s_['rank_clean']:>24.0f}{s_['rank_edit']:>9.0f}"
                     f"{100 * s_['prob_clean']:>24.2f}{100 * s_['prob_edit']:>9.2f}")
    L += ["", "plain = the original prompt; do_not_say = 'Do not say the word X. <prompt>'; wrong_answer = 'X is the wrong answer. <prompt>'."]
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    text = "\n".join(L)
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write(text)
    print("\n" + text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-band", type=int, default=10)
    ap.add_argument("--arms", default="gd,gd_add@cap=1.0,gd_add@cap=0.25")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--targets", choices=["true", "random"], default="true")
    ap.add_argument("--gen-tokens", type=int, default=20)
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
