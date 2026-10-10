"""
Steps E2 to E4 of docs/cross_model_transfer/PLAN.md: the Import experiment (medium's state written into small). The design, the arms, the controls, Gate 8
(confirmed by the author) and the readings are in the plan, written BEFORE this tool was built; this tool follows them.

    .venv\\Scripts\\python.exe tools/transfer/19_import_experiment.py --smoke     # one-minute rehearsal on 1,500 facts (numbers mean nothing)
    .venv\\Scripts\\python.exe tools/transfer/19_import_experiment.py             # the real run, about 15 minutes, progress printed all the time

Primary arm R (replacement, the "stitched model"): small's layer-8 state is replaced, at every position but the first, by medium's layer-16 state translated
by the A1 map m2s_L16_L8. Nothing is trained or tuned for R. Controls: C1 wrong-sentence donor, C2 floor (translated average medium state), C3 random map.
Measures: rescue on the candidates (medium right, small wrong), survival of the "both right" and "small right, medium wrong" facts, accuracy over all facts for
small, medium and the stitched model, agreement of small's answers with medium's (steering), rewordings, unrelated prompts. Secondary arms F (full difference) and S
(SAE-filtered difference) use a fixed dev/test split of the candidates; four other layer pairs are an exploratory table. Saves outputs/transfer/e2_import.json.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from src.sae_utils import load_sae_for_layer
from src.transfer.benchmark import load_counterfact
from src.transfer.import_eval import (length_batches, make_facts, sae_filtered_delta, small_last_logits, states_batch, top_and_rank, translate_state)
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to
from src.transfer.swap import NEUTRAL

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
PRIMARY = (16, 8)
OTHER_PAIRS = [(12, 8), (8, 6), (4, 2), (20, 10)]
ALPHAS = [0.25, 0.5, 1.0, 2.0, 3.0]
KS = [5, 20, 100]
G_RESCUE, G_NET, G_MARGIN, G_P, G_SURVIVE, STEER_POINTS = 0.50, 0.50, 0.10, 0.01, 0.90, 0.10


def pct(x):
    return "  n/a" if x != x else f"{100 * x:5.1f}%"


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log_path = tee_to("19_import_experiment")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = pick_device(a.device)
    t0 = time.time()
    ds = load_counterfact()
    rome_dev = set(random.Random(0).sample(range(len(ds)), 100))
    pilot = {r["case_id"] for r in json.load(open(os.path.join(OUT_DIR, "d_pilot.json"), encoding="utf8"))["records"]}
    small, medium, sae = load_model("gpt2", dev), load_model("gpt2-medium", dev), load_sae_for_layer(8, dev)
    maps = torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"]
    ent = {k: {"W": v["W"].to(dev), "mu_src": v["mu_src"].to(dev), "mu_dst": v["mu_dst"].to(dev)} for k, v in maps.items() if k.startswith("m2s")}
    facts = make_facts(small, ds, rome_dev, pilot)
    if a.smoke:
        facts = facts[:1500]
    use = [j for j, f in enumerate(facts) if not f["excluded"]]
    lens = [f["n_tok"] for f in facts]
    tid = torch.tensor([f["tid"] for f in facts], device=dev)
    n_all = len(use)
    print(f"{len(facts)} facts, {n_all} outside the ROME dev and pilot records", flush=True)

    def toks_of(idx):
        return small.to_tokens([facts[j]["prompt"] for j in idx])

    # ---------------- baselines: small and medium, no change --------------------------------------------------------------------------
    stage("1/6  baselines: small and medium on every fact (no edit, no translator)")
    N = len(facts)
    s_top, s_rank, m_top, m_rank = (torch.zeros(N, dtype=torch.long) for _ in range(4))
    m_prob = torch.zeros(N)
    done = 0
    for b in length_batches(use, lens, 128):
        toks = toks_of(b)
        t = tid[b]
        with torch.no_grad():
            ls, lm = small(toks)[:, -1].float(), medium(toks)[:, -1].float()
        st, sr = top_and_rank(ls, t)
        mt, mr = top_and_rank(lm, t)
        s_top[b], s_rank[b], m_top[b], m_rank[b] = st.cpu(), sr.cpu(), mt.cpu(), mr.cpu()
        m_prob[b] = F.softmax(lm, -1).gather(1, t[:, None])[:, 0].cpu()
        done += len(b)
        if done // 4096 > (done - len(b)) // 4096:
            print(f"   {done}/{n_all}", flush=True)
    s_ok, m_ok = s_rank == 1, m_rank == 1
    mask_use = torch.zeros(N, dtype=torch.bool)
    mask_use[use] = True
    cand = [j for j in use if m_ok[j] and not s_ok[j]]
    bothr = [j for j in use if m_ok[j] and s_ok[j]]
    smallonly = [j for j in use if s_ok[j] and not m_ok[j]]
    acc_s, acc_m = float(s_ok[use].float().mean()), float(m_ok[use].float().mean())
    agree_small = float((s_top[use] == m_top[use]).float().mean())
    print(f"   groups: candidates (medium right, small wrong) {len(cand)}; both right {len(bothr)}; small right, medium wrong {len(smallonly)}; both wrong {n_all - len(cand) - len(bothr) - len(smallonly)}", flush=True)
    print(f"   accuracy over all facts: small {pct(acc_s)} ({int(s_ok[use].sum())}), medium {pct(acc_m)} ({int(m_ok[use].sum())}); small's top answer equals medium's on {pct(agree_small)} of the facts", flush=True)

    # ---------------- primary arm R and the other layer pairs -----------------------------------------------------------------------------
    def run_R(pair, idx):
        qM, qS = pair
        e = ent[f"m2s_L{qM}_L{qS}"]
        top, rank = torch.zeros(N, dtype=torch.long), torch.zeros(N, dtype=torch.long)
        for b in length_batches(idx, lens, 64):
            toks = toks_of(b)
            lg = small_last_logits(small, toks, qS, replace=translate_state(states_batch(medium, toks, qM), e))
            t_, r_ = top_and_rank(lg, tid[b])
            top[b], rank[b] = t_.cpu(), r_.cpu()
        return top, rank

    stage("2/6  primary arm R (replacement) on every fact, pair m2s_L16_L8")
    R_top, R_rank = run_R(PRIMARY, use)
    R_ok = R_rank == 1

    def metrics(top, rank):
        ok = rank == 1
        return {"rescue": float(ok[cand].float().mean()) if cand else float("nan"), "survive_bothright": float(ok[bothr].float().mean()) if bothr else float("nan"),
                "survive_smallonly": float(ok[smallonly].float().mean()) if smallonly else float("nan"), "accuracy": float(ok[use].float().mean()),
                "agreement_with_medium": float((top[use] == m_top[use]).float().mean())}

    mR = metrics(R_top, R_rank)
    print(f"   R: rescue {pct(mR['rescue'])} of the candidates; survival {pct(mR['survive_bothright'])} of both-right and {pct(mR['survive_smallonly'])} of small-right-medium-wrong facts; "
          f"accuracy over all facts {pct(mR['accuracy'])}; agreement with medium's answers {pct(mR['agreement_with_medium'])} (small alone {pct(agree_small)})", flush=True)

    stage("3/6  controls on the candidates: C1 wrong-sentence donor, C2 floor (average medium state), C3 random map")
    qM, qS = PRIMARY
    e = ent[f"m2s_L{qM}_L{qS}"]
    g = torch.Generator(device=dev).manual_seed(0)
    W_rand = torch.randn(e["W"].shape, generator=g, device=dev)
    W_rand = W_rand * (e["W"].norm() / W_rand.norm())
    by_len = {}
    for j in use:
        by_len.setdefault(lens[j], []).append(j)
    rng = random.Random(0)
    ctrl_ok = {"C1": torch.zeros(N, dtype=torch.bool), "C2": torch.zeros(N, dtype=torch.bool), "C3": torch.zeros(N, dtype=torch.bool)}
    ctrl_rank = {k: torch.zeros(N, dtype=torch.long) for k in ctrl_ok}
    for b in length_batches(cand, lens, 64):
        toks = toks_of(b)
        hM = states_batch(medium, toks, qM)
        donors = [rng.choice([d for d in by_len[lens[j]] if d != j] or by_len[lens[j]]) for j in b]
        Td = translate_state(states_batch(medium, toks_of(donors), qM), e)
        reps = {"C1": Td, "C2": torch.zeros_like(hM[:, :, :768]) + e["mu_dst"], "C3": translate_state(hM, e, W_rand)}
        for k, rp in reps.items():
            t_, r_ = top_and_rank(small_last_logits(small, toks, qS, replace=rp), tid[b])
            ctrl_ok[k][b], ctrl_rank[k][b] = (r_ == 1).cpu(), r_.cpu()
    ctrl_rescue = {k: float(ctrl_ok[k][cand].float().mean()) for k in ctrl_ok}
    print("   rescue of the controls: " + ", ".join(f"{k} {pct(v)}" for k, v in ctrl_rescue.items()) + f"   (R {pct(mR['rescue'])})", flush=True)
    strongest = max(ctrl_rescue, key=ctrl_rescue.get)
    b_ = sum(1 for j in cand if R_ok[j] and not ctrl_ok[strongest][j])
    c_ = sum(1 for j in cand if (not R_ok[j]) and ctrl_ok[strongest][j])
    from scipy.stats import binomtest
    p_mc = float(binomtest(b_, b_ + c_, 0.5, alternative="greater").pvalue) if b_ + c_ > 0 else 1.0
    print(f"   paired test, R against the strongest control ({strongest}): R right and control wrong on {b_} candidates, the reverse on {c_}; one-sided exact p = {p_mc:.1e}", flush=True)

    # ---------------- rewordings and unrelated prompts for R --------------------------------------------------------------------------------
    stage("4/6  rewordings of the candidates and unrelated prompts (R)")
    para = [(j, p) for j in cand for p in facts[j]["paraphrases"]]
    ptoks_len = [len(small.tokenizer(p)["input_ids"]) for _, p in para]
    pr = {"small": 0, "medium": 0, "R": 0}
    for b in length_batches(list(range(len(para))), ptoks_len, 64):
        toks = small.to_tokens([para[i][1] for i in b])
        t = torch.tensor([facts[para[i][0]]["tid"] for i in b], device=dev)
        with torch.no_grad():
            ls, lm = small(toks)[:, -1].float(), medium(toks)[:, -1].float()
        lr = small_last_logits(small, toks, qS, replace=translate_state(states_batch(medium, toks, qM), e))
        pr["small"] += int((ls.argmax(-1) == t).sum())
        pr["medium"] += int((lm.argmax(-1) == t).sum())
        pr["R"] += int((lr.argmax(-1) == t).sum())
    npara = max(len(para), 1)
    print(f"   on {len(para)} rewordings of the candidates, the true answer is top-1 for: small {pct(pr['small'] / npara)}, medium {pct(pr['medium'] / npara)}, R {pct(pr['R'] / npara)}", flush=True)
    kls, flips = [], []
    for p in NEUTRAL:
        toks = small.to_tokens(p)
        with torch.no_grad():
            lc = small(toks)[:, -1].float()
        lr = small_last_logits(small, toks, qS, replace=translate_state(states_batch(medium, toks, qM), e))
        kls.append(float((F.softmax(lc, -1) * (F.log_softmax(lc, -1) - F.log_softmax(lr, -1))).sum()))
        flips.append(float(lc.argmax(-1) != lr.argmax(-1)))
    print(f"   on 12 unrelated prompts: KL(small || R) mean {sum(kls) / len(kls):.2f}, top-1 flips {pct(sum(flips) / len(flips))}", flush=True)

    # ---------------- secondary arms F and S -----------------------------------------------------------------------------------------------
    stage("5/6  secondary arms on the candidates: F (full difference) and S (SAE-filtered difference); k and alpha chosen on the dev facts")
    rs = random.Random(5)
    n_dev = min(100, max(len(cand) // 5, 1))
    dev_c = sorted(rs.sample(cand, n_dev))
    test_c = [j for j in cand if j not in set(dev_c)]
    base_rank = s_rank.float()

    def run_arm(kind, k, alpha, idx, push=False):
        rank = torch.zeros(N, dtype=torch.long)
        for b in length_batches(idx, lens, 32):
            toks = toks_of(b)
            hS, hM = states_batch(small, toks, qS), states_batch(medium, toks, qM)
            T = translate_state(hM, e)
            delta = (T - hS) if kind == "F" else sae_filtered_delta(sae, T, hS, k)
            delta = alpha * delta
            if push:
                u = small.W_U[:, tid[b]].T
                u = u / u.norm(dim=-1, keepdim=True)
                p_ = torch.zeros_like(delta)
                p_[:, -1] = u * delta[:, -1].norm(dim=-1, keepdim=True)
                delta = p_
            _, r_ = top_and_rank(small_last_logits(small, toks, qS, add=delta), tid[b])
            rank[b] = r_.cpu()
        return rank

    def gain(rank, idx):
        return float((base_rank[idx].log() - rank[idx].float().log()).mean())

    sec = {}
    for kind, ks in (("F", [0]), ("S", KS)):
        best = None
        for k in ks:
            for al in ALPHAS:
                r_ = run_arm(kind, k, al, dev_c)
                g_ = gain(r_, dev_c)
                key = (round(g_, 6), -al, -k)
                if best is None or key > best[0]:
                    best = (key, k, al)
        _, k, al = best
        r_ = run_arm(kind, k, al, test_c)
        rp = run_arm(kind, k, al, test_c, push=True) if kind == "S" else None
        sec[kind] = {"k": k, "alpha": al, "rescue": float((r_[test_c] == 1).float().mean()), "gain": gain(r_, test_c), "median_rank": float(r_[test_c].sort().values[len(test_c) // 2]),
                     "push_rescue": float((rp[test_c] == 1).float().mean()) if rp is not None else float("nan")}
        s_ = sec[kind]
        print(f"   {kind}: chosen on {n_dev} dev facts: k {k}, alpha {al}; on the {len(test_c)} test candidates rescue {pct(s_['rescue'])}, gain {s_['gain']:.2f}, median rank {s_['median_rank']:.0f}"
              + (f"; word push of the same size {pct(s_['push_rescue'])}" if kind == "S" else ""), flush=True)
    R_test = float(R_ok[test_c].float().mean()) if test_c else float("nan")
    print(f"   (R on the same {len(test_c)} test candidates: rescue {pct(R_test)})", flush=True)

    # ---------------- exploratory pairs -------------------------------------------------------------------------------------------------------
    stage("6/6  exploratory table: replacement for four other layer pairs (not used to choose anything)")
    pairs = {}
    for pair in [PRIMARY] + OTHER_PAIRS:
        if pair == PRIMARY:
            pt, pr_ = R_top, R_rank
        else:
            pt, pr_ = run_R(pair, use)
        pairs[f"m2s_L{pair[0]}_L{pair[1]}"] = metrics(pt, pr_)
    print(f"   {'pair':<14}{'rescue':>9}{'survive both-right':>20}{'survive small-only':>20}{'accuracy':>10}{'agreement':>11}")
    for k, m in pairs.items():
        print(f"   {k:<14}{pct(m['rescue']):>9}{pct(m['survive_bothright']):>20}{pct(m['survive_smallonly']):>20}{pct(m['accuracy']):>10}{pct(m['agreement_with_medium']):>11}")

    # ---------------- result block -------------------------------------------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("RESULT  steps E2 to E4 (Import: medium's state written into small)" + ("  [smoke: numbers mean nothing]" if a.smoke else ""))
    print(f"  facts outside the ROME dev and pilot records: {n_all}; candidates {len(cand)}; both right {len(bothr)}; small right, medium wrong {len(smallonly)}")
    print(f"  accuracy over all facts: small {pct(acc_s)}, medium {pct(acc_m)}, stitched model R {pct(mR['accuracy'])}")
    print(f"  agreement of small's top answer with medium's: small alone {pct(agree_small)}, R {pct(mR['agreement_with_medium'])}")
    print(f"  candidates rescued by R {pct(mR['rescue'])}; controls C1 {pct(ctrl_rescue['C1'])}, C2 {pct(ctrl_rescue['C2'])}, C3 {pct(ctrl_rescue['C3'])}")
    print(f"  survival under R: both-right facts {pct(mR['survive_bothright'])}, small-right-medium-wrong facts {pct(mR['survive_smallonly'])}")
    ga = mR["rescue"] >= G_RESCUE
    half = acc_s + G_NET * (acc_m - acc_s)
    gb = mR["accuracy"] >= half
    gc = (mR["rescue"] - ctrl_rescue[strongest]) >= G_MARGIN and p_mc < G_P
    gd = mR["survive_bothright"] >= G_SURVIVE
    steer = mR["agreement_with_medium"] - agree_small
    best_sec = max((sec[k]["rescue"] for k in sec), default=0.0)
    if ga and gb and gc and gd:
        reading = "all four hold: replacing small's state with medium's translated state carries medium's knowledge into small without telling it the answer"
    elif (not ga) and steer >= STEER_POINTS:
        reading = "(a) fails but small's answers move clearly toward medium's: small is steered toward medium's behaviour without the knowledge being carried (an intervention on another model with no SAE on the donor, not a knowledge transfer)"
    elif (not ga) and (not gb) and best_sec > R_test:
        reading = "(a) and (b) fail but F or S rescue more than R: the full replacement carries too much noise and filtering helps"
    elif not ga:
        reading = "(a) fails, the agreement with medium does not rise clearly and no secondary arm does better: Import with this map does not work"
    else:
        reading = "mixed: (a) holds but not all of (b) to (d); see the numbers"
    print(f"\n  GATE 8 (arm R, pair m2s_L16_L8): (a) rescue {pct(mR['rescue'])} >= {G_RESCUE:.0%}: {'ok' if ga else 'NO'};  (b) accuracy {pct(mR['accuracy'])} >= halfway {pct(half)} between small and medium: {'ok' if gb else 'NO'};  "
          f"(c) {int(100 * (mR['rescue'] - ctrl_rescue[strongest]))} points above the strongest control ({strongest}, >= {int(100 * G_MARGIN)}) with p = {p_mc:.1e} (< {G_P}): {'ok' if gc else 'NO'};  "
          f"(d) both-right survival {pct(mR['survive_bothright'])} >= {G_SURVIVE:.0%}: {'ok' if gd else 'NO'}   ->  {'PASS' if (ga and gb and gc and gd) else 'FAIL'}")
    print(f"  reading (fixed in the plan before the run): {reading}")

    # breakdowns for R on the candidates
    ans = Counter(facts[j]["true"] for j in cand)
    print("\n  descriptive breakdowns of the candidates (R):")
    print("   by the ten most common true answers (rescued / total): " + ", ".join(f"{k!r} {int(sum(R_ok[j] for j in cand if facts[j]['true'] == k))}/{v}" for k, v in ans.most_common(10)))
    bins = [("rank 2", lambda r: r == 2), ("rank 3 to 5", lambda r: 3 <= r <= 5), ("rank 6 to 20", lambda r: 6 <= r <= 20), ("rank over 20", lambda r: r > 20)]
    print("   by small's starting rank of the true answer: " + ";  ".join(f"{n}: {sum(1 for j in cand if f(int(s_rank[j])) and R_ok[j])}/{sum(1 for j in cand if f(int(s_rank[j])))}" for n, f in bins))
    hi = [j for j in cand if m_prob[j] >= 0.5]
    print(f"   by medium's confidence: at least 0.5 on {len(hi)} facts, R rescues {sum(1 for j in hi if R_ok[j])} of them; below 0.5 on {len(cand) - len(hi)} facts, R rescues {sum(1 for j in cand if m_prob[j] < 0.5 and R_ok[j])}")
    out = {"n_facts": n_all, "groups": {"candidates": len(cand), "both_right": len(bothr), "small_only": len(smallonly)}, "accuracy": {"small": acc_s, "medium": acc_m, "R": mR["accuracy"]},
           "agreement_with_medium": {"small": agree_small, "R": mR["agreement_with_medium"]}, "R": mR, "controls_rescue": ctrl_rescue, "strongest_control": strongest, "mcnemar_p": p_mc,
           "gate8": {"a": bool(ga), "b": bool(gb), "c": bool(gc), "d": bool(gd), "passes": bool(ga and gb and gc and gd), "reading": reading, "halfway_accuracy": half},
           "rewordings": {k: v / npara for k, v in pr.items()}, "unrelated": {"kl": sum(kls) / len(kls), "flips": sum(flips) / len(flips)}, "secondary": sec, "pairs": pairs,
           "candidates": [{"case_id": facts[j]["case_id"], "true": facts[j]["true"], "small_rank": int(s_rank[j]), "R_rank": int(R_rank[j]), "medium_prob": float(m_prob[j])} for j in cand]}
    path = os.path.join(OUT_DIR, "e2_import" + ("_smoke" if a.smoke else "") + ".json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)} and the log {os.path.relpath(log_path, ROOT)}")
    print("=" * 120)


if __name__ == "__main__":
    main()
