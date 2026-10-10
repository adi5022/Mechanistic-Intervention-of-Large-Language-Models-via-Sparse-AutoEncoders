"""
Step D6b of docs/cross_model_transfer/PLAN.md: how different is D6's solution (the 200 SAE dials re-tuned through the translator against medium's output) from the
original edit (tuned in small)? A descriptive diagnostic; the reading rule is in the plan, written before this run.

    .venv\\Scripts\\python.exe tools/transfer/20_d6_vs_original.py        # about a minute

For each of the 150 test records the two recipes (same 200 features) give a change of small's layer-8 state; translated with the ridge map small 8 -> medium 16 and
times dose 2.0 it is the injected change in medium. Compared: the dial vectors, and the injected changes (last position, all positions but the first), with chance
levels from other records. Saves outputs/transfer/d6b_compare.json.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.sae_utils import load_sae_for_layer
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to
from src.transfer.stitch import states

OUT = os.path.join(ROOT, "outputs", "transfer")
DOSE = 2.0


def cos(x, y):
    return float((x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-12))


def quart(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 4], xs[n // 2], xs[(3 * n) // 4]


def main():
    print(f"log: {os.path.relpath(tee_to('20_d6_vs_original'), ROOT)}", flush=True)
    dev = pick_device(None)
    bench = json.load(open(os.path.join(OUT, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT, "d_edits.pt"), weights_only=False)
    d6 = torch.load(os.path.join(OUT, "d6_recipes.pt"), weights_only=False)
    bench = [r for r in bench if r["case_id"] in edits]
    test = [r for r in bench if r["split"] == "test" and r["case_id"] in d6]
    W = torch.load(os.path.join(OUT, "maps_gpt2_to_gpt2-medium.pt"))["maps"]["s2m_L8_L16"]["W"].to(dev)
    small, sae = load_model("gpt2", dev), load_sae_for_layer(8, dev)
    rows = []
    with torch.no_grad():
        for r in test:
            e0, e1 = edits[r["case_id"]]["real"], d6[r["case_id"]]["real"]
            assert list(e0["fids"]) == list(e1["fids"])
            tok = small.to_tokens(r["prompt"])
            h = states(small, tok, 8)[0]
            fid = torch.as_tensor(e0["fids"], device=dev)
            acts, wd = sae.encode(h)[:, fid], sae.W_dec[fid].detach()
            a0, a1 = torch.tensor(e0["a"], device=dev), torch.tensor(e1["a"], device=dev)
            inj = []
            for a in (a0, a1):
                d = torch.einsum("pk,k,kd->pd", acts, a, wd)
                d[0] = 0
                inj.append(DOSE * (d @ W))
            i0, i1 = inj
            big = ((a0.abs() > 0.1) | (a1.abs() > 0.1))
            c = float((i1[-1] * i0[-1]).sum() / (i0[-1] @ i0[-1]).clamp(min=1e-12))
            rows.append({"case_id": r["case_id"], "primary": e0["rank"] == 1, "d6_ok": d6[r["case_id"]]["real"]["train_rank"] == 1,
                         "dial_cos": cos(a0, a1), "dial_pearson": float(torch.corrcoef(torch.stack([a0, a1]))[0, 1]), "dial_sign": float((a0[big].sign() == a1[big].sign()).float().mean()) if big.any() else float("nan"),
                         "dial_norm_ratio": float(a1.norm() / a0.norm().clamp(min=1e-12)), "last_cos": cos(i1[-1], i0[-1]), "all_cos": cos(i1[1:].flatten(), i0[1:].flatten()),
                         "last_scale": c, "last_norm_ratio": float(i1[-1].norm() / i0[-1].norm().clamp(min=1e-12)), "_i0": i0[-1].cpu(), "_i1": i1[-1].cpu()})
    n = len(rows)
    for k, row in enumerate(rows):
        o = rows[(k + 7) % n]
        row["chance_d6_vs_other_original"] = cos(row["_i1"], o["_i0"])
        row["chance_d6_vs_other_d6"] = cos(row["_i1"], o["_i1"])
    print("\n" + "=" * 110)
    print("RESULT  step D6b (how different is D6's solution from the original edit?)")
    out = {}
    for label, sel in (("PRIMARY and D6 reached rank 1 in training", [x for x in rows if x["primary"] and x["d6_ok"]]), ("ALL test records where D6 reached rank 1 in training", [x for x in rows if x["d6_ok"]])):
        print(f"\n  {label}: {len(sel)} records (median, quartiles)")
        out[label] = {"n": len(sel)}
        for k, nm in (("dial_cos", "dial vectors: cosine"), ("dial_pearson", "dial vectors: Pearson correlation"), ("dial_sign", "dial vectors: sign agreement (dials above 0.1)"), ("dial_norm_ratio", "dial vectors: size, D6 over original"),
                      ("last_cos", "injected change, last position: cosine D6 vs original"), ("all_cos", "injected change, all positions but the first: cosine"), ("last_scale", "last position: best scalar multiple of the original"),
                      ("last_norm_ratio", "last position: size, D6 over original"), ("chance_d6_vs_other_original", "CHANCE: D6 vs another record's original (last position)"), ("chance_d6_vs_other_d6", "CHANCE: D6 vs another record's D6 (last position)")):
            q = quart([x[k] for x in sel if x[k] == x[k]])
            out[label][k] = q
            print(f"    {nm:<62}{q[1]:>8.2f}   ({q[0]:.2f} to {q[2]:.2f})")
    prim = [x for x in rows if x["primary"] and x["d6_ok"]]
    med = sorted(x["last_cos"] for x in prim)[len(prim) // 2]
    ch = sorted(x["chance_d6_vs_other_original"] for x in prim)[len(prim) // 2]
    ch2 = sorted(x["chance_d6_vs_other_d6"] for x in prim)[len(prim) // 2]
    reading = "at least 0.5: the same direction with a different strength (a dose or map fix is plausible)" if med >= 0.5 else (
        "0.2 to 0.5: partly the same direction" if med >= 0.2 else "under 0.2: a different direction; the original edit does not contain what medium needs, so a map alone cannot fix it (favours changing the edit side)")
    print(f"\n  median last-position cosine D6 vs original {med:.2f} (chance levels {ch:.2f} and {ch2:.2f})  ->  {reading}")
    print("  reading rule fixed in the plan before the run")
    json.dump({"summary": out, "median_last_cos": med, "chance_other_original": ch, "chance_other_d6": ch2, "reading": reading,
               "records": [{k: v for k, v in x.items() if not k.startswith("_")} for x in rows]}, open(os.path.join(OUT, "d6b_compare.json"), "w", encoding="utf8"), indent=1)
    print(f"\n  saved outputs/transfer/d6b_compare.json")
    print("=" * 110)


if __name__ == "__main__":
    main()
