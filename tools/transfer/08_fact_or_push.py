"""
Phase C of docs/cross_model_transfer/PLAN.md (hypothesis H24), GPT-2 small alone: does the gradient-descent multiplier edit change how the
subject is represented (like a fact edit), or only push the answer word at the read-out?

    .venv\\Scripts\\python.exe tools/transfer/08_fact_or_push.py --quick
    .venv\\Scripts\\python.exe tools/transfer/08_fact_or_push.py          # the 150 test records of the step D benchmark (about 25 minutes)

Needs the output of 05_export_edits.py. Four edit variants per record, each re-applied (same feature multipliers, the prompt's own activations) to the tuned
prompt, its 2 paraphrase prompts and 5 neighbour prompts, all inside GPT-2 small:
  all          the existing edit (every position), from 05_export_edits.py
  subject      a new edit that may only change the SUBJECT's tokens (src/gradient_editing_masked.py)
  last         a new edit that may only change the LAST position
  random word  the existing edit toward a rank-matched random word (judged on that word)
Measures: edit success (target rank 1 on the tuned prompt); paraphrase effect (log-rank gain of the target on the paraphrases, and how often
P(new) > P(true)); neighbour effect (P(true) still > P(new)); carry = paraphrase gain / tuned-prompt gain; and for the existing edit, how much of
its change at the last position points along the target word's own output direction (cosine and logit-lens rank).
Pre-stated reading (written in the plan before this run), judged on test records whose edit reached rank 1:
  association-like  if the real edit's paraphrase gain is above the random-word edit's (paired Wilcoxon p < 0.01) AND the subject-only edit keeps at least
                    half of the all-position edit's paraphrase gain and success rate
  read-out steering if the paraphrase gain is not above the random-word edit's, OR the last-position change is dominated by the target's output direction
                    (median logit-lens rank 10 or better)
  mixed             otherwise
Writes outputs/transfer/c_fact_or_push.json.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from src.editing import build_clean_context
from src.gradient_editing_masked import run_masked_edit, subject_positions
from src.sae_utils import load_sae_for_layer
from src.transfer.export_eval import make_recipe
from src.transfer.models import load_model, pick_device
from src.transfer.stitch import states
from src.transfer.swap import run_injected

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
P_ASSOC = 0.01


def rate(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs):
    xs = sorted(x for x in xs if x == x)
    return xs[len(xs) // 2] if xs else float("nan")


def pct(x):
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


@torch.no_grad()
def change(small, sae, tok, recipe, kind, subject, text):
    h = states(small, tok, 8)[0]
    fids, a = recipe
    acts = sae.encode(h)[:, fids]
    mask = torch.zeros(h.shape[0], 1, device=h.device)
    if kind in ("all", "random"):
        mask[:] = 1
    elif kind == "subject":
        pos = subject_positions(small, text, subject)
        if pos:
            mask[pos] = 1
    elif kind == "last":
        mask[-1] = 1
    return torch.einsum("pk,k,kd->pd", acts * mask, a, sae.W_dec[fids].detach())


def rank_info(small, tok, delta, tid_target, tid_other):
    lg = run_injected(small, tok, delta[None], 8)[0]
    lp = F.log_softmax(lg, dim=-1)
    return int((lg > lg[tid_target]).sum().item()) + 1, float(lp[tid_target] - lp[tid_other])


@torch.no_grad()
def clean_info(small, tok, tid_target, tid_other):
    lg = small(tok)[0, -1].float()
    lp = F.log_softmax(lg, dim=-1)
    return int((lg > lg[tid_target]).sum().item()) + 1, float(lp[tid_target] - lp[tid_other])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    suffix = "_quick" if a.quick else ""
    dev = pick_device(a.device)
    t0 = time.time()
    bench = [r for r in json.load(open(os.path.join(OUT_DIR, f"d_benchmark{suffix}.json"), encoding="utf8")) if r["split"] == "test"]
    edits = torch.load(os.path.join(OUT_DIR, f"d_edits{suffix}.pt"), weights_only=False)
    small, sae = load_model("gpt2", dev), load_sae_for_layer(8, dev)
    out_path = os.path.join(OUT_DIR, f"c_fact_or_push{suffix}.json")
    done = json.load(open(out_path, encoding="utf8")) if os.path.exists(out_path) else {}
    print(f"{len(bench)} test records; {len(done)} already done", flush=True)

    for n, r in enumerate(bench):
        if r["case_id"] in done:
            continue
        e = edits[r["case_id"]]
        tn, tt = r["tid_new"], r["tid_true"]
        tok = small.to_tokens(r["prompt"])
        pos = subject_positions(small, r["prompt"], r["subject"])
        ctx = build_clean_context(small, sae, r["prompt"], tn)
        variants = {"all": make_recipe(e["real"], dev), "random": make_recipe(e["random"], dev) if "random" in e else None}
        masks = {}
        if pos:
            m = torch.zeros(tok.shape[1]); m[pos] = 1; masks["subject"] = (m, "all")
        m = torch.zeros(tok.shape[1]); m[-1] = 1; masks["last"] = (m, "last")
        edit_ok = {}
        for kind, (m, cand) in masks.items():
            res = run_masked_edit(small, sae, ctx, tn, r["prompt"], 8, m, positions=cand, steps=a.steps)
            variants[kind] = (torch.tensor(res["fids"], device=dev), torch.tensor(res["a"], device=dev))
            edit_ok[kind] = {"rank": res["rank"], "kl": res["kl"]}
        rec = {"subject_found": bool(pos), "variants": {}}
        for kind, recipe in variants.items():
            if recipe is None:
                continue
            target = e["random"]["tid"] if kind == "random" else tn
            other = tt
            sets = {"tuned": [r["prompt"]], "paraphrase": r["paraphrases"], "neighbour": r["neighbours"]}
            v = {}
            for sname, texts in sets.items():
                rows = []
                for text in texts:
                    t = small.to_tokens(text)
                    d = change(small, sae, t, recipe, kind, r["subject"], text)
                    b_rank, b_ld = clean_info(small, t, target, other)
                    a_rank, a_ld = rank_info(small, t, d, target, other)
                    rows.append({"rank_before": b_rank, "rank_after": a_rank, "ld_before": b_ld, "ld_after": a_ld,
                                 "gain": math.log(b_rank) - math.log(a_rank)})
                v[sname] = rows
            if kind in ("all", "random"):
                d = change(small, sae, tok, recipe, kind, r["subject"], r["prompt"])[-1]
                wu = small.W_U[:, target]
                lens = d @ small.W_U
                cos_all = F.cosine_similarity(small.W_U.T, d[None], dim=1)
                v["direction"] = {"cos": float(F.cosine_similarity(d, wu, dim=0)), "cos_percentile": float((cos_all < F.cosine_similarity(d, wu, dim=0)).float().mean()),
                                  "lens_rank": int((lens > lens[target]).sum().item()) + 1}
            rec["variants"][kind] = v
        rec["masked_edit_rank_shortcut"] = edit_ok
        done[r["case_id"]] = rec
        if (n + 1) % 5 == 0:
            json.dump(done, open(out_path, "w", encoding="utf8"))
            print(f"   {n + 1}/{len(bench)} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(done, open(out_path, "w", encoding="utf8"))

    # ---------------- analysis ---------------------------------------------------------------------------------------------------
    from scipy.stats import wilcoxon
    ids = [r["case_id"] for r in bench if r["case_id"] in done]

    def ok(c, kind):
        v = done[c]["variants"].get(kind)
        return v is not None and v["tuned"][0]["rank_after"] == 1

    def para_gain(c, kind):
        return rate([x["gain"] for x in done[c]["variants"][kind]["paraphrase"]])

    def tuned_gain(c, kind):
        return done[c]["variants"][kind]["tuned"][0]["gain"]

    print("\n" + "=" * 118)
    print("RESULT  Phase C: fact edit or word push? (GPT-2 small alone)" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    print(f"  {len(ids)} test records. Edit success = the target is rank 1 on the tuned prompt. PS = P(new) > P(true) on the 2 paraphrase prompts; NS = P(true) > P(new) on the 5 neighbour prompts.")
    print(f"  {'variant':<22}{'success':>9}{'PS':>7}{'NS':>7}{'para gain':>11}{'carry':>8}   (para gain = mean log-rank gain on paraphrases, on records where the variant succeeded; carry = para gain / tuned-prompt gain)")
    rows = {}
    for kind, label in (("all", "all positions"), ("subject", "subject tokens only"), ("last", "last position only"), ("random", "random word, all pos.")):
        has = [c for c in ids if kind in done[c]["variants"]]
        good = [c for c in has if ok(c, kind)]
        ps = rate([float(x["ld_after"] > 0) for c in good for x in done[c]["variants"][kind]["paraphrase"]])
        ns = rate([float(x["ld_after"] < 0) for c in good for x in done[c]["variants"][kind]["neighbour"]])
        pg = rate([para_gain(c, kind) for c in good])
        carry = med([para_gain(c, kind) / max(tuned_gain(c, kind), 1e-6) for c in good])
        rows[kind] = {"n": len(has), "n_success": len(good), "success": len(good) / max(len(has), 1), "ps": ps, "ns": ns, "para_gain": pg, "carry": carry}
        print(f"  {label:<22}{pct(len(good) / max(len(has), 1)):>9}{pct(ps):>7}{pct(ns):>7}{pg:>11.2f}{carry:>8.2f}   ({len(good)} of {len(has)} records)")
    base_ps = rate([float(x["ld_before"] > 0) for c in ids for x in done[c]["variants"]["all"]["paraphrase"]])
    base_ns = rate([float(x["ld_before"] < 0) for c in ids for x in done[c]["variants"]["all"]["neighbour"]])
    print(f"  unedited small: PS {pct(base_ps)}, NS {pct(base_ns)}")

    both = [c for c in ids if ok(c, "all") and ok(c, "random")]
    p_assoc = float("nan")
    if len(both) >= 8:
        x, y = [para_gain(c, "all") for c in both], [para_gain(c, "random") for c in both]
        try:
            p_assoc = float(wilcoxon(x, y, alternative="greater").pvalue)
        except ValueError:
            p_assoc = 1.0
        print(f"\n  paraphrase gain, real counterfactual target vs rank-matched random word ({len(both)} records where both edits succeeded): {rate(x):.2f} vs {rate(y):.2f}, paired Wilcoxon (real greater) p = {p_assoc:.1e}")
    dirs = [done[c]["variants"]["all"]["direction"] for c in ids if "direction" in done[c]["variants"]["all"]]
    lens_med = med([d["lens_rank"] for d in dirs])
    print(f"  direction of the existing edit's change at the last position vs the target word's output direction: median cosine {med([d['cos'] for d in dirs]):.3f}, "
          f"median percentile among all words {100 * med([d['cos_percentile'] for d in dirs]):.1f}, median logit-lens rank of the target {lens_med:.0f}")
    dr = [done[c]["variants"]["random"]["direction"] for c in ids if "random" in done[c]["variants"] and "direction" in done[c]["variants"]["random"]]
    if dr:
        print(f"    same for the random-word edit: median cosine {med([d['cos'] for d in dr]):.3f}, median logit-lens rank {med([d['lens_rank'] for d in dr]):.0f}")

    subj_keeps = (rows["subject"]["para_gain"] >= 0.5 * rows["all"]["para_gain"]) and (rows["subject"]["success"] >= 0.5 * rows["all"]["success"]) if rows["subject"]["n"] else False
    higher = (p_assoc == p_assoc) and p_assoc < P_ASSOC
    dominated = lens_med == lens_med and lens_med <= 10
    if higher and subj_keeps and not dominated:
        verdict = "ASSOCIATION-LIKE"
    elif (p_assoc == p_assoc and not higher) or dominated:
        verdict = "READ-OUT STEERING"
    else:
        verdict = "MIXED"
    print(f"\n  pre-stated reading: paraphrase gain above the random word's: {'yes' if higher else 'no'}; subject-only keeps half of the all-position effect: {'yes' if subj_keeps else 'no'}; "
          f"change dominated by the target's output direction (lens rank <= 10): {'yes' if dominated else 'no'}   ->  {verdict}")
    json.dump({"rows": rows, "p_assoc": p_assoc, "verdict": verdict, "baseline": {"ps": base_ps, "ns": base_ns}, "lens_rank_median": lens_med},
              open(os.path.join(OUT_DIR, f"c_fact_or_push_summary{suffix}.json"), "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    sys.exit(main())
