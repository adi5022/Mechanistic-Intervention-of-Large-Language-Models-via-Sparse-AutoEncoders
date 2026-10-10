"""
Writes the per-configuration test table of step D5 (10_multilayer_export.py) into the data pack, from the local per-record numbers
(outputs/transfer/d5_stage1.pt, d5_multilayer.pt), so that the figures can be drawn from committed files only.

    .venv\Scripts\python.exe tools/transfer/13_pack_d5_table.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
PACK = os.path.join(ROOT, "docs", "Research_Journal", "packs", "cross_model_transfer")
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    edits = torch.load(os.path.join(OUT_DIR, "d_edits.pt"), weights_only=False)
    d5 = torch.load(os.path.join(OUT_DIR, "d5_multilayer.pt"), weights_only=False)
    res1, sel = d5["res1"], d5["sel"]
    ids = [r["case_id"] for r in bench if r["split"] == "test" and r["case_id"] in edits and edits[r["case_id"]]["real"]["rank"] == 1]
    base = [res1[c]["base"]["rank_new"][0].item() for c in ids]
    out = {"n_test_records": len(ids), "unchanged_medium": {"top1": sum(res1[c]["base"]["top1"][0].item() for c in ids) / len(ids), "median_rank": med(base)}, "configs": {}}
    for name, (i, dev_top1, dev_gain) in sel.items():
        t = lambda arm, j: [res1[c][name][arm]["top1"][j].item() for c in ids]
        r = lambda arm, j: [res1[c][name][arm]["rank_new"][j].item() for c in ids]
        out["configs"][name] = {
            "dose": ALPHAS[i], "dev_top1": dev_top1, "dev_gain": dev_gain,
            "test_top1": sum(t("translated", i)) / len(ids), "test_median_rank": med(r("translated", i)),
            "norm_matched_top1": sum(t("translated", -1)) / len(ids), "norm_matched_median_rank": med(r("translated", -1)),
            "random_top1_same_dose": sum(t("random", i)) / len(ids), "wrong_recipe_top1_same_dose": sum(t("wrong_recipe", i)) / len(ids),
        }
    path = os.path.join(PACK, "d5_configs.json")
    json.dump(out, open(path, "w", encoding="utf8"), indent=1)
    print("wrote", os.path.relpath(path, ROOT))
    for k, v in out["configs"].items():
        print(f"  {k:<22} dose {v['dose']:<4} test top-1 {100 * v['test_top1']:.0f}%  median rank {v['test_median_rank']:.0f}")


if __name__ == "__main__":
    main()
