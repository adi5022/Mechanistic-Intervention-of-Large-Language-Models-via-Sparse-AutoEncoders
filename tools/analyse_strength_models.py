"""
Read the output of tools/train_strength_models.py and compare the methods PROMPT BY PROMPT (needs the local cache and tables).

The training script prints averages; this tool re-evaluates the exported strengths of the reference (0.6 / 0.5), the learned
fixed pair, the selected PromptNet and the per-prompt tuned "oracle" on the 450 held-out prompts (validation + test-seen +
test-unseen) with the same prefix proxy, then reports paired comparisons (exact McNemar test on rank 1), side effects and a
breakdown by starting rank. The numbers are from the PROXY (not the real sweep).

    .venv\\Scripts\\python.exe tools/analyse_strength_models.py --dir outputs/strength_models/<time>
    .venv\\Scripts\\python.exe tools/analyse_strength_models.py --dir ~/Downloads        (a folder holding results.json + strengths_export.json)
"""
import argparse
import json
import os
import statistics as st
import sys
from argparse import Namespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from scipy.stats import binomtest

from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src import strength_models as sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with results.json and strengths_export.json")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs", "strength_cache"))
    ap.add_argument("--tables-dir", default=os.path.join(ROOT, "outputs", "strength_tables"))
    ap.add_argument("--split-file", default=os.path.join(ROOT, "data", "counterfact_split.json"))
    ap.add_argument("--lam-kl", type=float, default=10.0)
    ap.add_argument("--lam-size", type=float, default=0.02)
    a0 = ap.parse_args()
    R = json.load(open(os.path.join(a0.dir, "results.json")))
    EX = json.load(open(os.path.join(a0.dir, "strengths_export.json")))
    print("selected:", R.get("selected"), "| learned fixed pair:", EX["constant"])
    big = max(x["size"] for x in R["runs"])
    print(f"\nSize {big}, mean +- std over seeds (rank-1 %, from the training run):")
    for kind in ("constant", "promptnet"):
        rs = [x for x in R["runs"] if x["kind"] == kind and x["size"] == big]
        for s in ("val", "test_seen", "test_unseen"):
            v = [100 * x[s]["rank1_rate"] for x in rs]
            print(f"  {kind:<10}{s:<12}{st.mean(v):5.1f} +- {st.pstdev(v):.2f}   seeds {[round(a, 1) for a in v]}")
    print("epochs used (promptnet):", [x["epochs"] for x in R["runs"] if x["kind"] == "promptnet" and x["size"] == big])

    dev = get_default_device()
    model = load_base_model(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    W = load_sae_for_layer(layer=8).W_dec.detach()
    split = json.load(open(a0.split_file, encoding="utf-8"))
    a = Namespace(batch=32, prefix_list=sm.DEFAULT_PREFIXES, train_prefix_list=[1, 3, 8, 21, 0], margin=0.3, lam_kl=a0.lam_kl, lam_size=a0.lam_size)
    items = []
    for name in ("val", "test_seen", "test_unseen"):
        items += [sm.load_item(i, name, a0.cache_dir, a0.tables_dir) for i in split[name]]
    CP = EX["constant"]

    def fn_const(mu, be):
        return lambda bt: (torch.full((bt["B"],), mu, device=dev), torch.full((bt["B"],), be, device=dev))

    def per_case(table):
        def fn(bt):
            return (torch.tensor([table[str(c)]["mu"] for c in bt["case_ids"]], device=dev),
                    torch.tensor([table[str(c)]["beta"] for c in bt["case_ids"]], device=dev))
        return fn

    methods = {"reference": fn_const(0.6, 0.5), "constant": fn_const(CP["mu"], CP["beta"]),
               "promptnet": per_case(EX["promptnet"]), "oracle": per_case(EX["oracle"])}
    res = {}
    for m, fn in methods.items():
        out = sm.evaluate(model, fn, items, a, W, dev)
        res[m] = {c: (r, k, mu, be) for c, r, k, mu, be in zip(out["case_id"], out["rank"], out["kl"], out["mu"], out["beta"])}
    sets = {it["case_id"]: it["set"] for it in items}
    start = {it["case_id"]: it["baseline_rank"] for it in items}
    ids = [it["case_id"] for it in items]
    rate = lambda m, sub: sum(1 for c in sub if res[m][c][0] == 1)

    print("\nPer-prompt re-evaluation (prefix proxy, from the exported strengths):")
    groups = {"val": [c for c in ids if sets[c] == "val"], "test_seen": [c for c in ids if sets[c] == "test_seen"],
              "test_unseen": [c for c in ids if sets[c] == "test_unseen"],
              "TEST ONLY (300)": [c for c in ids if sets[c] != "val"], "ALL 450": ids}
    for g, sub in groups.items():
        print(f"  {g:<16}" + "  ".join(f"{m}: {rate(m, sub)}/{len(sub)} ({100 * rate(m, sub) / len(sub):.1f}%)" for m in methods))

    for title, sub in (("all 450 held-out prompts", ids), ("the 300 TEST prompts only (validation excluded)", groups["TEST ONLY (300)"])):
        print(f"\nPaired comparisons on {title} (exact McNemar test on rank 1):")
        for x, y in (("constant", "reference"), ("promptnet", "reference"), ("promptnet", "constant"), ("oracle", "promptnet"), ("oracle", "constant")):
            b = sum(1 for c in sub if res[x][c][0] == 1 and res[y][c][0] != 1)
            cc = sum(1 for c in sub if res[x][c][0] != 1 and res[y][c][0] == 1)
            both = sum(1 for c in sub if res[x][c][0] == 1 and res[y][c][0] == 1)
            p = binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0
            print(f"  {x:<10} vs {y:<10}: {x} only {b:>3} | {y} only {cc:>3} | both succeed {both:>3} | p = {p:.4f}")

    print("\nSide effects (mean KL among prompts that reach rank 1) and mean strengths:")
    for m in methods:
        ks = [res[m][c][1] for c in ids if res[m][c][0] == 1]
        print(f"  {m:<10} KL(succ) {st.mean(ks):.3f}   mean mu {st.mean(res[m][c][2] for c in ids):.2f}  mean beta {st.mean(res[m][c][3] for c in ids):.2f}")

    bands = [(2, 5), (6, 20), (21, 100), (101, 1000)]
    print("\nBy starting rank (rank-1 counts):")
    print(f"  {'band':<10}" + "".join(f"{m:>12}" for m in methods) + "     n")
    for lo, hi in bands:
        sub = [c for c in ids if lo <= start[c] <= hi]
        print(f"  {lo}-{hi:<7}" + "".join(f"{rate(m, sub):>12}" for m in methods) + f"   {len(sub)}")
    print("\nDoes the choice depend on how deep the target starts? (mean strengths by starting rank)")
    for lo, hi in bands:
        sub = [c for c in ids if lo <= start[c] <= hi]
        print(f"  rank {lo}-{hi:<5} promptnet mu {st.mean(res['promptnet'][c][2] for c in sub):.2f} beta {st.mean(res['promptnet'][c][3] for c in sub):.2f}"
              f"   | oracle mu {st.mean(res['oracle'][c][2] for c in sub):.2f} beta {st.mean(res['oracle'][c][3] for c in sub):.2f}")
    pm, pb = [res["promptnet"][c][2] for c in ids], [res["promptnet"][c][3] for c in ids]
    om, ob = [res["oracle"][c][2] for c in ids], [res["oracle"][c][3] for c in ids]
    print(f"  promptnet: mu {min(pm):.2f} to {max(pm):.2f}, beta {min(pb):.2f} to {max(pb):.2f}   (allowed range mu 0.2 to 1.0, beta 0.25 to 2.0)")
    print(f"  oracle:    mu {min(om):.2f} to {max(om):.2f}, beta {min(ob):.2f} to {max(ob):.2f}")


if __name__ == "__main__":
    main()
