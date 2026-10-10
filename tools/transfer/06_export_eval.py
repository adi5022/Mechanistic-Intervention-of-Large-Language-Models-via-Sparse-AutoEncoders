"""
Steps D3 and D4 of docs/cross_model_transfer/PLAN.md: controlled export of the gradient-descent edit from GPT-2 small to GPT-2 medium.
Needs the output of 05_export_edits.py.

    .venv\\Scripts\\python.exe tools/transfer/06_export_eval.py --quick
    .venv\\Scripts\\python.exe tools/transfer/06_export_eval.py                   # linear translator, maps small 8 -> medium 12 and 16
    .venv\\Scripts\\python.exe tools/transfer/06_export_eval.py --translator mlp  # neural translator (needs 07_fit_mlp_maps.py)

For every record the edit tuned in small ("recipe": feature multipliers) gives a change of small's layer-8 state at every position. That change
is translated, multiplied by a dose and added to medium's residual stream (same prompt). Medium is then asked about the counterfactual target.

Stage 1 (tuned prompt, every record, 7 doses 0.5 to 6): the dose is chosen on the 50 dev records whose edit reached rank 1 in small, by top-1 rate;
  a second, outcome-free dose is also run (norm matched: the injected change is as large relative to medium's residual norm as the edit was
  relative to small's).
Stage 2 (test records, at the chosen dose and the norm-matched dose): the same recipe re-applied to the record's 2 paraphrase prompts, 5 neighbour
  prompts and the 12 unrelated prompts.
Arms: translated | random (same size per position) | wrong_recipe (another record's recipe on this prompt) | word_push (medium's own output direction
  of the target at the last position, same size) | randtarget (the edit toward a rank-matched random word, judged on that word) | with-start-token ablation.
Reference rows: medium unchanged; medium with the fact stated in its prompt; small with the edit itself (the source's own effect).
Measures (CounterFact style): top1 = the counterfactual target is the top answer; rank; ES = P(new) > P(true) on the tuned prompt; PS = same on
paraphrases; NS = P(true) > P(new) on neighbours (higher = fewer leaks); on unrelated prompts KL and top-1 flips against the unchanged model.
Gate 3 (PROPOSED in the plan before this run; evaluated on test records whose edit reached rank 1 in small): (a) top-1 at the chosen dose >= 15%;
(b) it exceeds the best of random / wrong_recipe / randtarget (each at its best dose on test) by >= 10 points; (c) paired Wilcoxon on the gain in
log-rank, translated against that strongest control, p < 0.01. Reported, not gated: paraphrase / neighbour / unrelated effects and the word_push comparison.
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

from src.sae_utils import load_sae_for_layer
from src.transfer.export_eval import (LinearTranslator, MlpTranslator, evaluate, make_arms, make_recipe, norm_match_factors, plain_logits,
                                      small_change, summarise_logits)
from src.transfer.models import load_model, pick_device
from src.transfer.stitch import states
from src.transfer.swap import NEUTRAL, run_injected

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
GATE_RATE, GATE_MARGIN, GATE_P = 0.15, 0.10, 0.01
CONTROLS = ["random", "wrong_recipe", "randtarget"]
ARM_NAMES = ["translated", "random", "wrong_recipe", "word_push", "randtarget"]


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rate(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs):
    xs = sorted(x for x in xs if x == x)
    return xs[len(xs) // 2] if xs else float("nan")


def pct(x):
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


def get_translator(name, key, dev, maps):
    if name == "linear":
        return LinearTranslator(maps[key]["W"].to(dev))
    from src.transfer.mlp_maps import load_mlp_translator
    return MlpTranslator(load_mlp_translator(key, dev))


# ------------------------------------------------------------------------------------------------------------------------------
def stage1_record(env, rec, edit, wrong_edit):
    small, medium, sae, tr, q, dev = env["small"], env["medium"], env["sae"], env["tr"], env["q"], env["dev"]
    tok = small.to_tokens(rec["prompt"])
    tn, tt = rec["tid_new"], rec["tid_true"]
    h, dS = small_change(small, sae, tok, make_recipe(edit["real"], dev))
    Dt = tr(h, dS)
    hw, dSw = small_change(small, sae, tok, make_recipe(wrong_edit["real"], dev))
    gen = torch.Generator().manual_seed(rec["index"])
    arms = make_arms(Dt, tr(hw, dSw), medium.W_U[:, tn], gen)
    h_med = states(medium, tok, q)[0]
    s_nm = norm_match_factors(h, dS, h_med, Dt)
    al = torch.tensor(ALPHAS, device=dev)[:, None, None]
    out = {"base": summarise_logits(plain_logits(medium, tok)[None], tn, tt), "small_base": summarise_logits(plain_logits(small, tok)[None], tn, tt)}
    for arm, D in arms.items():
        out[arm] = evaluate(medium, tok, al * D[None], q, tn, tt)
        out[arm + "_nm"] = evaluate(medium, tok, (s_nm[:, None] * D)[None], q, tn, tt)
    # start-token ablation: keep the start-token row of the change
    _, dS_b = small_change(small, sae, tok, make_recipe(edit["real"], dev), zero_bos=False)
    out["translated_withbos"] = evaluate(medium, tok, al * tr(h, dS_b, keep_bos=True)[None], q, tn, tt)
    # edit toward a rank-matched random word: judged on that word, and on the counterfactual target
    if "random" in edit:
        tw = edit["random"]["tid"]
        hr, dSr = small_change(small, sae, tok, make_recipe(edit["random"], dev))
        Dr = tr(hr, dSr)
        out["randtarget"] = evaluate(medium, tok, al * Dr[None], q, tw, tt)
        out["randtarget_nm"] = evaluate(medium, tok, (norm_match_factors(hr, dSr, h_med, Dr)[:, None] * Dr)[None], q, tw, tt)
        out["randtarget_base"] = summarise_logits(plain_logits(medium, tok)[None], tw, tt)
        out["randtarget_small"] = {"start_rank": edit["random"]["start_rank"], "rank": edit["random"]["rank"]}
    # the source's own effect: the edit applied in small itself
    _, dS_full = small_change(small, sae, tok, make_recipe(edit["real"], dev), zero_bos=False)
    out["small_native"] = summarise_logits(run_injected(small, tok, dS_full[None], 8), tn, tt)
    out["small_edit_rank"] = edit["real"]["rank"]
    return out


def stage2_record(env, rec, edit, wrong_edit, alpha_star, clean_cache):
    """Re-apply the recipe to paraphrase, neighbour and unrelated prompts."""
    small, medium, sae, tr, q, dev = env["small"], env["medium"], env["sae"], env["tr"], env["q"], env["dev"]
    tn, tt = rec["tid_new"], rec["tid_true"]
    rec_real, rec_wrong = make_recipe(edit["real"], dev), make_recipe(wrong_edit["real"], dev)
    sets = {"paraphrase": rec["paraphrases"], "neighbour": rec["neighbours"], "unrelated": NEUTRAL}
    out = {}
    for kind, plist in sets.items():
        rows = []
        for text in plist:
            tok = small.to_tokens(text)
            if text not in clean_cache:
                clean_cache[text] = plain_logits(medium, tok)
            clean = clean_cache[text]
            h, dS = small_change(small, sae, tok, rec_real)
            Dt = tr(h, dS)
            hw, dSw = small_change(small, sae, tok, rec_wrong)
            gen = torch.Generator().manual_seed(rec["index"] + len(text))
            arms = make_arms(Dt, tr(hw, dSw), medium.W_U[:, tn], gen)
            s_nm = norm_match_factors(h, dS, states(medium, tok, q)[0], Dt)
            row = {"base": summarise_logits(clean[None], tn, tt, clean[None])}
            for arm, D in arms.items():
                deltas = torch.stack([alpha_star * D, s_nm[:, None] * D])
                row[arm] = evaluate(medium, tok, deltas, q, tn, tt, clean_logits=clean[None].repeat(2, 1))
            _, dS_full = small_change(small, sae, tok, rec_real, zero_bos=False)
            sc = plain_logits(small, tok)
            row["small_base"] = summarise_logits(sc[None], tn, tt, sc[None])
            row["small_native"] = summarise_logits(run_injected(small, tok, dS_full[None], 8), tn, tt, sc[None])
            if kind != "unrelated":                                     # the fact stated in medium's prompt (plain prompting baseline)
                ctxt = medium.to_tokens(rec["prompt"] + " " + rec["target_new"] + ". " + text)
                row["in_context"] = summarise_logits(plain_logits(medium, ctxt)[None], tn, tt)
            rows.append(row)
        out[kind] = rows
    return out


# ------------------------------------------------------------------------------------------------------------------------------
def pick_alpha(bench, edits, res1, arm="translated"):
    """Dose chosen on the dev records whose edit reached rank 1 in small (all dev records if there are none): top-1 rate, then mean log-rank gain."""
    dev_ids = [r["case_id"] for r in bench if r["split"] == "dev" and edits[r["case_id"]]["real"]["rank"] == 1]
    if not dev_ids:
        dev_ids = [r["case_id"] for r in bench if r["split"] == "dev"]
    best = None
    for i, a in enumerate(ALPHAS):
        top1 = rate([res1[c][arm]["top1"][i].item() for c in dev_ids])
        gain = rate([math.log(res1[c]["base"]["rank_new"][0].item()) - math.log(res1[c][arm]["rank_new"][i].item()) for c in dev_ids])
        key = (round(top1, 6), gain, -a)
        if best is None or key > best[0]:
            best = (key, i)
    return best[1]


def stage1_table(bench, edits, res1, subset, label):
    ids = [r["case_id"] for r in bench if r["split"] == "test" and (subset == "all" or edits[r["case_id"]]["real"]["rank"] == 1)]
    lines = [f"  {label}: {len(ids)} test records"]
    base_rank = med([res1[c]["base"]["rank_new"][0].item() for c in ids])
    lines.append(f"    unchanged medium: top-1 {pct(rate([res1[c]['base']['top1'][0].item() for c in ids]))}, ES {pct(rate([res1[c]['base']['es'][0].item() for c in ids]))}, median rank of the target {base_rank:.0f}")
    lines.append(f"    small with the edit itself: top-1 {pct(rate([res1[c]['small_native']['top1'][0].item() for c in ids]))}, ES {pct(rate([res1[c]['small_native']['es'][0].item() for c in ids]))}")
    lines.append(f"    {'arm':<20}" + "".join(f"{'dose ' + str(a):>16}" for a in ALPHAS) + f"{'norm matched':>16}    (cells: top-1 / median rank)")
    for arm in ARM_NAMES + ["translated_withbos"]:
        if arm == "randtarget":
            pick = [c for c in ids if "randtarget" in res1[c]]
        else:
            pick = ids
        cells = []
        for i, a in enumerate(ALPHAS):
            cells.append(f"{pct(rate([res1[c][arm]['top1'][i].item() for c in pick]))} {med([res1[c][arm]['rank_new'][i].item() for c in pick]):>5.0f}".rjust(16))
        if arm + "_nm" in res1[ids[0]] if ids else False:
            cells.append(f"{pct(rate([res1[c][arm + '_nm']['top1'][0].item() for c in pick if arm + '_nm' in res1[c]]))} {med([res1[c][arm + '_nm']['rank_new'][0].item() for c in pick if arm + '_nm' in res1[c]]):>5.0f}".rjust(16))
        lines.append(f"    {arm:<20}" + "".join(cells))
    return "\n".join(lines)


def gate3(bench, edits, res1, i_star):
    from scipy.stats import wilcoxon
    ids = [r["case_id"] for r in bench if r["split"] == "test" and edits[r["case_id"]]["real"]["rank"] == 1]
    if len(ids) < 5:
        return {"n": len(ids), "note": "too few test records whose edit reached rank 1 in small"}
    tr_top1 = rate([res1[c]["translated"]["top1"][i_star].item() for c in ids])
    base_key = lambda arm: "randtarget_base" if arm == "randtarget" else "base"          # each arm's gain is measured against ITS OWN word's starting rank
    gain = lambda arm, i, pick: [math.log(res1[c][base_key(arm)]["rank_new"][0].item()) - math.log(res1[c][arm]["rank_new"][i].item()) for c in pick]
    best_ctrl = {}
    for arm in CONTROLS:
        pick = [c for c in ids if arm in res1[c]]
        if not pick:
            continue
        idx = max(range(len(ALPHAS)), key=lambda i: (round(rate([res1[c][arm]["top1"][i].item() for c in pick]), 6), rate(gain(arm, i, pick))))
        best_ctrl[arm] = {"dose": ALPHAS[idx], "top1": rate([res1[c][arm]["top1"][idx].item() for c in pick]), "idx": idx, "n": len(pick)}
    strongest = max(best_ctrl, key=lambda a: (round(best_ctrl[a]["top1"], 6), rate(gain(a, best_ctrl[a]["idx"], [c for c in ids if a in res1[c]]))))
    pick = [c for c in ids if strongest in res1[c]]
    x, y = gain("translated", i_star, pick), gain(strongest, best_ctrl[strongest]["idx"], pick)
    try:
        p = float(wilcoxon(x, y, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    a_ok, b_ok, c_ok = tr_top1 >= GATE_RATE, (tr_top1 - best_ctrl[strongest]["top1"]) >= GATE_MARGIN, p < GATE_P
    return {"n": len(ids), "dose": ALPHAS[i_star], "translated_top1": tr_top1, "controls": {k: {kk: vv for kk, vv in v.items() if kk != "idx"} for k, v in best_ctrl.items()},
            "strongest_control": strongest, "wilcoxon_p": p, "a_rate": bool(a_ok), "b_margin": bool(b_ok), "c_paired": bool(c_ok), "passes": bool(a_ok and b_ok and c_ok)}


def stage2_table(bench, edits, res2, subset):
    ids = [r["case_id"] for r in bench if r["split"] == "test" and r["case_id"] in res2 and (subset == "all" or edits[r["case_id"]]["real"]["rank"] == 1)]
    lines = [f"  {len(ids)} test records ({'all' if subset == 'all' else 'edit reached rank 1 in small'}); cells: PS (paraphrases: new beats true) / NS (neighbours: true still beats new) / unrelated KL / unrelated top-1 flips"]

    def agg(rows_of, arm, col):
        vals = []
        for c in ids:
            rows = res2[c][rows_of]
            for row in rows:
                if arm in row:
                    vals.append(row[arm][col])
        return vals

    def cell(arm, which):
        # which: 0 = chosen dose, 1 = norm matched, None = single row (base)
        def get(kind, col):
            v = []
            for c in ids:
                for row in res2[c][kind]:
                    r = row[arm]
                    v.append(float(r[col][which] if which is not None else r[col][0]))
            return rate(v)
        ps = get("paraphrase", "es")
        ns = 1 - get("neighbour", "es")
        kl = get("unrelated", "kl") if "kl" in res2[ids[0]]["unrelated"][0][arm] else float("nan")
        fl = get("unrelated", "flip") if "flip" in res2[ids[0]]["unrelated"][0][arm] else float("nan")
        return ps, ns, kl, fl

    lines.append(f"    {'arm':<26}{'PS':>7}{'NS':>7}{'KL':>8}{'flips':>8}")
    for name, arm, which in [("medium unchanged", "base", None), ("small unchanged", "small_base", None), ("small with the edit", "small_native", None)]:
        ps, ns, kl, fl = cell(arm, which)
        lines.append(f"    {name:<26}{pct(ps):>7}{pct(ns):>7}{kl:>8.3f}{pct(fl):>8}")
    ic = {kind: rate([float(row["in_context"]["es"][0]) for c in ids for row in res2[c][kind] if "in_context" in row]) for kind in ("paraphrase", "neighbour")}
    lines.append(f"    {'medium, fact in prompt':<26}{pct(ic['paraphrase']):>7}{pct(1 - ic['neighbour']):>7}{'':>8}{'':>8}")
    for arm in ["translated", "random", "wrong_recipe", "word_push"]:
        for which, tag in ((0, "chosen dose"), (1, "norm matched")):
            ps, ns, kl, fl = cell(arm, which)
            lines.append(f"    {arm + ' (' + tag + ')':<26}{pct(ps):>7}{pct(ns):>7}{kl:>8.3f}{pct(fl):>8}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keys", default="s2m_L8_L12,s2m_L8_L16")
    ap.add_argument("--translator", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    suffix = "_quick" if a.quick else ""
    tag = f"{a.translator}{suffix}"
    dev = pick_device(a.device)
    t0 = time.time()
    bench = json.load(open(os.path.join(OUT_DIR, f"d_benchmark{suffix}.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, f"d_edits{suffix}.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    maps = torch.load(os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt"))["maps"]
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    print(f"{len(bench)} records ({sum(r['split'] == 'dev' for r in bench)} dev, {sum(r['split'] == 'test' for r in bench)} test); translator {a.translator}; keys {a.keys}", flush=True)

    allres = {}
    for key in a.keys.split(","):
        q = int(key.split("_L")[2])
        tr = get_translator(a.translator, key, dev, maps)
        env = {"small": small, "medium": medium, "sae": sae, "tr": tr, "q": q, "dev": dev}
        stage(f"{key} ({a.translator}): stage 1, the tuned prompt, all doses, all arms")
        res1 = {}
        for n, r in enumerate(bench):
            wrong = edits[bench[(n + 7) % len(bench)]["case_id"]]
            res1[r["case_id"]] = stage1_record(env, r, edits[r["case_id"]], wrong)
            if (n + 1) % 50 == 0:
                print(f"   {n + 1}/{len(bench)} ({time.time() - t0:.0f} s)", flush=True)
        i_star = pick_alpha(bench, edits, res1)
        alpha_star = ALPHAS[i_star]
        print(f"   dose chosen on dev: {alpha_star}", flush=True)
        stage(f"{key} ({a.translator}): stage 2, rewordings, neighbours and unrelated prompts (test records)")
        res2, clean_cache = {}, {}
        test_recs = [r for r in bench if r["split"] == "test"]
        for n, r in enumerate(test_recs):
            idx = bench.index(r)
            wrong = edits[bench[(idx + 7) % len(bench)]["case_id"]]
            res2[r["case_id"]] = stage2_record(env, r, edits[r["case_id"]], wrong, alpha_star, clean_cache)
            if (n + 1) % 25 == 0:
                print(f"   {n + 1}/{len(test_recs)} ({time.time() - t0:.0f} s)", flush=True)
        allres[key] = {"res1": res1, "res2": res2, "alpha_star": alpha_star, "i_star": i_star}
        torch.save({"bench": bench, "allres": allres}, os.path.join(OUT_DIR, f"d_eval_{tag}.pt"))

    # ---------------- result block -------------------------------------------------------------------------------------------------
    summary = {}
    print("\n" + "=" * 130)
    print(f"RESULT  steps D3 and D4 (controlled export, translator: {a.translator})" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    for key, d in allres.items():
        res1, res2, i_star = d["res1"], d["res2"], d["i_star"]
        print(f"\n  ====== {key}: dose chosen on dev = {d['alpha_star']} ======")
        print(stage1_table(bench, edits, res1, "small_ok", "TUNED PROMPT, test records whose edit reached rank 1 in small (primary)"))
        print(stage1_table(bench, edits, res1, "all", "TUNED PROMPT, all test records"))
        g = gate3(bench, edits, res1, i_star)
        summary[key] = {"alpha_star": d["alpha_star"], "gate3": g}
        if "note" in g:
            print(f"  GATE 3: {g['note']}")
        else:
            c = g["controls"][g["strongest_control"]]
            print(f"  GATE 3 ({g['n']} records): (a) top-1 {pct(g['translated_top1'])} >= {GATE_RATE:.0%}: {'ok' if g['a_rate'] else 'NO'};  "
                  f"(b) strongest control {g['strongest_control']} at dose {c['dose']} = {pct(c['top1'])}, margin >= {100 * GATE_MARGIN:.0f} points: {'ok' if g['b_margin'] else 'NO'};  "
                  f"(c) paired p = {g['wilcoxon_p']:.1e} < {GATE_P}: {'ok' if g['c_paired'] else 'NO'}   ->  {'PASS' if g['passes'] else 'FAIL'}")
        if res2:
            print("\n  STAGE 2, " + key)
            print(stage2_table(bench, edits, res2, "small_ok"))
            print(stage2_table(bench, edits, res2, "all"))
        rt = [c for c in res1 if "randtarget" in res1[c]]
        ids = [r["case_id"] for r in bench if r["split"] == "test" and edits[r["case_id"]]["real"]["rank"] == 1 and r["case_id"] in rt]
        if ids:
            print(f"\n  plausible counterfactual target vs rank-matched random word (test records with a successful real edit, dose {d['alpha_star']}): "
                  f"top-1 for the counterfactual target {pct(rate([res1[c]['translated']['top1'][i_star].item() for c in ids]))}, for the random word {pct(rate([res1[c]['randtarget']['top1'][i_star].item() for c in ids]))}; "
                  f"median rank {med([res1[c]['translated']['rank_new'][i_star].item() for c in ids]):.0f} vs {med([res1[c]['randtarget']['rank_new'][i_star].item() for c in ids]):.0f}")
    out_json = os.path.join(OUT_DIR, f"d_eval_{tag}.json")
    json.dump(summary, open(out_json, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(out_json, ROOT)} and d_eval_{tag}.pt (per-record numbers)")
    print("=" * 130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
