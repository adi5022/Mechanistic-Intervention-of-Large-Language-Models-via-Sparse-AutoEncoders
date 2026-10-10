"""
Step E5 of docs/cross_model_transfer/PLAN.md: a softer Import (a descriptive diagnostic; the reading rule is in the plan, written before this tool was built).

    .venv\\Scripts\\python.exe tools/transfer/23_softer_import.py --smoke     # rehearsal on 1,500 facts (numbers mean nothing)
    .venv\\Scripts\\python.exe tools/transfer/23_softer_import.py             # about 10 minutes

Arm P(alpha): small's layer-8 state (positions but the first) becomes (1 - alpha) h_S + alpha T(h_M), with T the A1 map m2s_L16_L8; alpha = 1 is the replacement of E2 to E4.
Arm G(alpha, tau): P(alpha) only where unchanged medium's top-1 probability at the last position is at least tau, small unchanged elsewhere. Facts are split into two halves with a fixed
seed: half A chooses the setting (highest overall accuracy among those keeping at least 90% of the "both right" facts), half B reports it. Saves outputs/transfer/e5_softer_import.json.
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from scipy.stats import binomtest

from src.transfer.benchmark import load_counterfact
from src.transfer.import_eval import length_batches, make_facts, small_last_logits, states_batch, top_and_rank, translate_state
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
QM, QS = 16, 8
ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
TAUS = [0.1, 0.2, 0.3, 0.5, 0.7]
G_GAIN, G_SURVIVE, G_P = 0.005, 0.90, 0.01


def pct(x):
    return "  n/a" if x != x else f"{100 * x:5.1f}%"


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    print(f"log: {os.path.relpath(tee_to('23_softer_import'), ROOT)}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = pick_device(a.device)
    t0 = time.time()
    ds = load_counterfact()
    rome_dev = set(random.Random(0).sample(range(len(ds)), 100))
    pilot = {r["case_id"] for r in json.load(open(os.path.join(OUT_DIR, "d_pilot.json"), encoding="utf8"))["records"]}
    small, medium = load_model("gpt2", dev), load_model("gpt2-medium", dev)
    e = {k: v.to(dev) for k, v in torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"][f"m2s_L{QM}_L{QS}"].items() if torch.is_tensor(v)}
    facts = make_facts(small, ds, rome_dev, pilot)
    if a.smoke:
        facts = facts[:1500]
    use = [j for j, f in enumerate(facts) if not f["excluded"]]
    lens = [f["n_tok"] for f in facts]
    tid = torch.tensor([f["tid"] for f in facts], device=dev)
    N = len(facts)
    toks_of = lambda idx: small.to_tokens([facts[j]["prompt"] for j in idx])

    stage("1/3  baselines: small and medium on every fact; medium's confidence")
    s_top, s_rank, m_top, m_rank = (torch.zeros(N, dtype=torch.long) for _ in range(4))
    conf = torch.zeros(N)
    for b in length_batches(use, lens, 128):
        toks = toks_of(b)
        with torch.no_grad():
            ls, lm = small(toks)[:, -1].float(), medium(toks)[:, -1].float()
        st, sr = top_and_rank(ls, tid[b])
        mt, mr = top_and_rank(lm, tid[b])
        s_top[b], s_rank[b], m_top[b], m_rank[b], conf[b] = st.cpu(), sr.cpu(), mt.cpu(), mr.cpu(), F.softmax(lm, -1).max(-1).values.cpu()
    s_ok, m_ok = s_rank == 1, m_rank == 1
    cand = torch.zeros(N, dtype=torch.bool)
    bothr = torch.zeros(N, dtype=torch.bool)
    smallonly = torch.zeros(N, dtype=torch.bool)
    for j in use:
        cand[j], bothr[j], smallonly[j] = bool(m_ok[j] and not s_ok[j]), bool(m_ok[j] and s_ok[j]), bool(s_ok[j] and not m_ok[j])
    print(f"   {len(use)} facts: candidates {int(cand.sum())}, both right {int(bothr.sum())}, small right and medium wrong {int(smallonly.sum())}; small {pct(float(s_ok[use].float().mean()))}, medium {pct(float(m_ok[use].float().mean()))} right", flush=True)

    stage(f"2/3  P(alpha) for alpha in {ALPHAS} on every fact")
    P_ok, P_top = {}, {}
    for al in ALPHAS:
        ok, top = torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.long)
        for b in length_batches(use, lens, 64):
            toks = toks_of(b)
            hS, hM = states_batch(small, toks, QS), states_batch(medium, toks, QM)
            lg = small_last_logits(small, toks, QS, replace=(1 - al) * hS + al * translate_state(hM, e))
            t_, r_ = top_and_rank(lg, tid[b])
            ok[b], top[b] = (r_ == 1).cpu(), t_.cpu()
        P_ok[al], P_top[al] = ok, top
        print(f"   alpha {al}: accuracy {pct(float(ok[use].float().mean()))}, rescue {pct(float(ok[cand].float().mean()))}, both-right kept {pct(float(ok[bothr].float().mean()))}, ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- settings, halves, choice ----------------------------------------------------------------------------------------------
    settings = {("P", al, None): (P_ok[al], P_top[al]) for al in ALPHAS}
    for al in ALPHAS:
        for tau in TAUS:
            g = conf >= tau
            settings[("G", al, tau)] = (torch.where(g, P_ok[al], s_ok), torch.where(g, P_top[al], s_top))
    rs = random.Random(11)
    order = list(use)
    rs.shuffle(order)
    half = {"A": torch.zeros(N, dtype=torch.bool), "B": torch.zeros(N, dtype=torch.bool)}
    half["A"][order[:len(order) // 2]] = True
    half["B"][order[len(order) // 2:]] = True

    def stats(ok, top, mask):
        c, b_, s_ = cand & mask, bothr & mask, smallonly & mask
        return {"acc": float(ok[mask].float().mean()), "rescue": float(ok[c].float().mean()) if c.any() else float("nan"), "survive_both": float(ok[b_].float().mean()) if b_.any() else float("nan"),
                "survive_small_only": float(ok[s_].float().mean()) if s_.any() else float("nan"), "agree": float((top[mask] == m_top[mask]).float().mean())}

    stage("3/3  choosing the setting on half A and reporting it on half B")
    table = {k: {"A": stats(ok, top, half["A"]), "B": stats(ok, top, half["B"])} for k, (ok, top) in settings.items()}
    small_A, small_B = stats(s_ok, s_top, half["A"]), stats(s_ok, s_top, half["B"])
    med_B = stats(m_ok, m_top, half["B"])
    feasible = [k for k in table if table[k]["A"]["survive_both"] >= G_SURVIVE]
    pool_ = feasible if feasible else list(table)
    best = max(pool_, key=lambda k: (round(table[k]["A"]["acc"], 6), -k[1]))
    ok_b, top_b = settings[best]
    cB = table[best]["B"]
    maskB = half["B"]
    n10 = int(((ok_b & ~s_ok) & maskB).sum())
    n01 = int(((~ok_b & s_ok) & maskB).sum())
    p = float(binomtest(n10, n10 + n01, 0.5, alternative="greater").pvalue) if n10 + n01 > 0 else 1.0
    gain = cB["acc"] - small_B["acc"]
    verdict = ("A SOFTER IMPORT IS NET-POSITIVE" if (gain >= G_GAIN and cB["survive_both"] >= G_SURVIVE and p < G_P) else ("MARGINAL: a gain above 0 that misses a condition" if gain > 0 else "NO SOFTER VARIANT HELPS: Import by state injection costs more than it gains with this map"))
    print("\n" + "=" * 130)
    print("RESULT  step E5 (a softer Import)" + ("  [smoke: numbers mean nothing]" if a.smoke else ""))
    print(f"  half A {int(half['A'].sum())} facts (choose), half B {int(half['B'].sum())} facts (report). small alone on B: accuracy {pct(small_B['acc'])}; medium alone on B {pct(med_B['acc'])}")
    print(f"\n  all settings on half A (accuracy, rescue of the candidates, both-right kept, small-only kept, agreement with medium); '*' = keeps at least 90% of the both-right facts")
    print(f"   {'setting':<26}{'accuracy':>10}{'rescue':>9}{'both kept':>11}{'small-only':>12}{'agreement':>11}")
    for k in sorted(table, key=lambda k: -table[k]["A"]["acc"])[:14]:
        s_ = table[k]["A"]
        name = f"P(alpha={k[1]})" if k[0] == "P" else f"G(alpha={k[1]}, tau={k[2]})"
        print(f"   {name:<26}{pct(s_['acc']):>10}{pct(s_['rescue']):>9}{pct(s_['survive_both']):>11}{pct(s_['survive_small_only']):>12}{pct(s_['agree']):>11} {'*' if s_['survive_both'] >= G_SURVIVE else ''}")
    name = f"P(alpha={best[1]})" if best[0] == "P" else f"G(alpha={best[1]}, tau={best[2]})"
    print(f"\n  chosen on half A: {name} ({'meets' if feasible else 'no setting meets'} the 90% both-right condition on A)")
    print(f"  on half B: accuracy {pct(cB['acc'])} (small alone {pct(small_B['acc'])}, medium alone {pct(med_B['acc'])}); gain {100 * gain:+.2f} points; rescue {pct(cB['rescue'])}; both-right kept {pct(cB['survive_both'])}; "
          f"small-only kept {pct(cB['survive_small_only'])}; agreement with medium {pct(cB['agree'])} (small alone {pct(small_B['agree'])})")
    print(f"  paired exact test against small alone on B: chosen right and small wrong on {n10} facts, the reverse on {n01}; one-sided p = {p:.1e}")
    print(f"  reading (fixed in the plan before the run): {verdict}")
    out = {"half_sizes": {k: int(v.sum()) for k, v in half.items()}, "small_B": small_B, "medium_B": med_B, "chosen": [best[0], best[1], best[2]], "chosen_B": cB, "gain_points": 100 * gain, "n10": n10, "n01": n01, "p": p, "verdict": verdict,
           "table": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in table.items()}}
    path = os.path.join(OUT_DIR, "e5_softer_import" + ("_smoke" if a.smoke else "") + ".json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1, default=float)
    print(f"\n  total time {time.time() - t0:.0f} s; saved {os.path.relpath(path, ROOT)}")
    print("=" * 130)


if __name__ == "__main__":
    main()
