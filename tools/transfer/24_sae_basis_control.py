"""
Step S1 of docs/cross_model_transfer/PLAN.md: does small's SAE matter for steering medium? D6 (the 200 dials tuned through the translator against medium's output) repeated with a random
basis instead of the SAE decoder directions. The design and the reading rule are in the plan, written before this tool was built.

    .venv\\Scripts\\python.exe tools/transfer/24_sae_basis_control.py --smoke      # rehearsal on 6 test records (numbers mean nothing)
    .venv\\Scripts\\python.exe tools/transfer/24_sae_basis_control.py              # about 45 minutes; tuned recipes are saved as it goes (resumable)

Arms: rb_real (random basis tuned toward the real counterfactual target), rb_wrong (another record's rb_real recipe), rb_rand (random basis tuned toward the rank-matched random word, judged on that
word). Compared with D6's SAE-basis numbers (packs/.../d6_b_aware.json, same records and settings). Saves outputs/transfer/s1_basis_control.json.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.sae_utils import load_sae_for_layer
from src.transfer.basis_control import change_with_basis, random_basis, tune_with_basis
from src.transfer.export_eval import plain_logits, summarise_logits
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to
from src.transfer.swap import NEUTRAL, run_injected

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
Q, DOSE = 16, 2.0
READ_TOP1, READ_LIFT = 0.10, 0.05


def pct(x):
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    print(f"log: {os.path.relpath(tee_to('24_sae_basis_control'), ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = pick_device(a.device)
    t0 = time.time()
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    test = [r for r in bench if r["split"] == "test"]
    if a.smoke:
        test = test[:6]
    W = torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"]["s2m_L8_L16"]["W"].to(dev)
    d6 = json.load(open(os.path.join(OUT_DIR, "d6_b_aware.json"), encoding="utf8"))["summary"]["ok"]
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    for m in (small, medium):
        for p_ in m.parameters():
            p_.requires_grad_(False)
    basis = random_basis(sae, seed=0)
    print(f"{len(test)} test records; random basis: {tuple(basis.shape)} Gaussian rows with the norms of the SAE decoder rows; layer {Q}, dose {DOSE}", flush=True)

    path = os.path.join(OUT_DIR, "s1_recipes" + ("_smoke" if a.smoke else "") + ".pt")
    recipes = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    stage(f"1/3  tuning with the random basis ({len(recipes)} records already done)")
    for n, r in enumerate(test):
        cid = r["case_id"]
        if cid in recipes:
            continue
        tok = small.to_tokens(r["prompt"])
        entry = {"real": tune_with_basis(small, medium, sae, W, edits[cid]["real"]["fids"], tok, r["tid_new"], Q, DOSE, basis)}
        if "random" in edits[cid]:
            entry["rand"] = tune_with_basis(small, medium, sae, W, edits[cid]["random"]["fids"], tok, edits[cid]["random"]["tid"], Q, DOSE, basis)
        recipes[cid] = entry
        if (n + 1) % 10 == 0 or n + 1 == len(test):
            torch.save(recipes, path)
            print(f"   tuned {n + 1}/{len(test)} ({time.time() - t0:.0f} s)", flush=True)
    torch.save(recipes, path)

    stage("2/3  judging: tuned prompt, rewordings, neighbours, unrelated prompts")
    res1, res2, clean_cache = {}, {}, {}
    nt = len(test)

    def rows_for(tok, specs):
        ds = []
        for recipe in specs:
            h, dS = change_with_basis(small, sae, tok, recipe, basis, dev)
            ds.append(DOSE * (dS @ W))
        return run_injected(medium, tok.repeat(len(specs), 1), torch.stack(ds), Q)

    for i, r in enumerate(test):
        cid = r["case_id"]
        tn, tt = r["tid_new"], r["tid_true"]
        tw = edits[cid]["random"]["tid"] if "random" in edits[cid] else None
        specs = [recipes[cid]["real"], recipes[test[(i + 7) % nt]["case_id"]]["real"]] + ([recipes[cid]["rand"]] if "rand" in recipes[cid] else [])
        has_rand = len(specs) == 3

        def judge(lg, clean=None):
            kw = {} if clean is None else {"clean_logits": clean[None].repeat(2, 1)}
            s = summarise_logits(lg[:2], tn, tt, **kw)
            out = {"rb_real": {k: v[0:1] for k, v in s.items()}, "rb_wrong": {k: v[1:2] for k, v in s.items()}}
            if has_rand:
                kw = {} if clean is None else {"clean_logits": clean[None]}
                out["rb_rand"] = summarise_logits(lg[2:3], tw, tt, **kw)
            return out

        tok = small.to_tokens(r["prompt"])
        o1 = {"base": summarise_logits(plain_logits(medium, tok)[None], tn, tt)}
        o1.update(judge(rows_for(tok, specs)))
        res1[cid] = o1
        o2 = {}
        for kind, plist in (("paraphrase", r["paraphrases"]), ("neighbour", r["neighbours"]), ("unrelated", NEUTRAL)):
            rows = []
            for text in plist:
                tk = small.to_tokens(text)
                if text not in clean_cache:
                    clean_cache[text] = plain_logits(medium, tk)
                clean = clean_cache[text]
                row = {"base": summarise_logits(clean[None], tn, tt, clean[None])}
                row.update(judge(rows_for(tk, specs), clean))
                rows.append(row)
            o2[kind] = rows
        res2[cid] = o2
        if (i + 1) % 25 == 0:
            print(f"   {i + 1}/{nt} ({time.time() - t0:.0f} s)", flush=True)

    stage("3/3  result")
    rec_primary = [r["case_id"] for r in test if edits[r["case_id"]]["real"]["rank"] == 1]

    def agg2(ids, arm, kind, col):
        return mean([float(row[arm][col][0]) for c in ids for row in res2[c][kind] if arm in row])

    out = {}
    print("\n" + "=" * 120)
    print("RESULT  step S1 (does small's SAE matter for steering medium?)" + ("  [smoke: numbers mean nothing]" if a.smoke else ""))
    for label, ids in (("PRIMARY (original edit reached rank 1 in small)", rec_primary), ("ALL test records", [r["case_id"] for r in test])):
        if not ids:
            continue
        print(f"\n  ---- {label}: {len(ids)} records ----")
        print(f"    {'arm':<46}{'top-1':>7}{'med rank':>10}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
        o = {}
        for arm, nm, base in (("rb_real", "random basis, real target", "base"), ("rb_wrong", "random basis, another record's recipe", "base"), ("rb_rand", "random basis, random-word target (own word)", "base")):
            pick = [c for c in ids if arm in res1[c]]
            if not pick:
                continue
            t1 = mean([res1[c][arm]["top1"][0].item() for c in pick])
            mr = med([res1[c][arm]["rank_new"][0].item() for c in pick])
            ps, ns = agg2(pick, arm, "paraphrase", "es"), 1 - agg2(pick, arm, "neighbour", "es")
            kl, fl = agg2(pick, arm, "unrelated", "kl"), agg2(pick, arm, "unrelated", "flip")
            o[arm] = {"top1": t1, "median_rank": mr, "PS": ps, "NS": ns, "KL": kl, "flips": fl}
            print(f"    {nm:<46}{pct(t1):>7}{mr:>10.0f}{pct(ps):>7}{pct(ns):>7}{kl:>8.3f}{pct(fl):>8}")
        o["medium_unchanged_PS"] = agg2(ids, "base", "paraphrase", "es")
        o["medium_unchanged_NS"] = 1 - agg2(ids, "base", "neighbour", "es")
        out[label] = o
        print(f"    {'medium unchanged':<46}{'':>7}{'':>10}{pct(o['medium_unchanged_PS']):>7}{pct(o['medium_unchanged_NS']):>7}")
    print(f"\n  D6 WITH THE SAE BASIS (same records, same settings; packs/.../d6_b_aware.json, 88 primary records):")
    print(f"    {'SAE basis, real target':<46}{pct(d6['top1']['b_aware']):>7}{d6['median_rank']['b_aware']:>10.0f}{pct(d6['stage2']['b_aware']['PS']):>7}{pct(d6['stage2']['b_aware']['NS']):>7}{d6['stage2']['b_aware']['KL']:>8.3f}{pct(d6['stage2']['b_aware']['flips']):>8}")
    print(f"    {'SAE basis, random-word target (own word)':<46}{pct(d6['top1']['random_word_b']):>7}{d6['median_rank']['random_word_b']:>10.0f}{pct(d6['stage2']['random_word_b']['PS']):>7}{pct(d6['stage2']['random_word_b']['NS']):>7}")
    prim = out.get("PRIMARY (original edit reached rank 1 in small)")
    if prim and "rb_real" in prim:
        top_diff = d6["top1"]["b_aware"] - prim["rb_real"]["top1"]
        lift_sae = d6["stage2"]["b_aware"]["PS"] - d6["stage2"]["medium unchanged"]["PS"]
        lift_rb = prim["rb_real"]["PS"] - prim["medium_unchanged_PS"]
        lift_diff = lift_sae - lift_rb
        if -top_diff >= READ_TOP1:
            reading = "the random basis is BETTER than the SAE basis (top-1 higher by at least 10 points)"
        elif top_diff >= READ_TOP1 or lift_diff >= READ_LIFT:
            reading = "the SAE basis MATTERS (D6's top-1 is higher by at least 10 points or its rewordings lift by at least 5 points)"
        else:
            reading = "NO EVIDENCE that the SAE basis matters (both differences are smaller than the thresholds)"
        print(f"\n  difference (SAE basis minus random basis): top-1 {100 * top_diff:+.0f} points, rewordings lift {100 * lift_diff:+.0f} points  ->  {reading}")
        print("  reading rule fixed in the plan before the run")
        out["reading"], out["top1_diff"], out["lift_diff"] = reading, top_diff, lift_diff
    path = os.path.join(OUT_DIR, "s1_basis_control" + ("_smoke" if a.smoke else "") + ".json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}")
    print("=" * 120)


if __name__ == "__main__":
    main()
