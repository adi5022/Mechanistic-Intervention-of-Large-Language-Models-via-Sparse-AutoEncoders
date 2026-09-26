"""
Summarise the 21-prompt x 12-layer Layer Intervention Benchmark run (benchmark_results/layer_benchmark_20260909_144450_707979.json)
so the numbers quoted in the paper draft can be regenerated from the raw artifact.

    python tools/summarize_layer_benchmark.py

Writes benchmark_results/layer_benchmark_20260909_summary.csv and prints a table.
Only prompts whose target is NOT already rank 1 in the clean model (clean_rank > 1) are summarised.
"Reached rank 1" means final_rank == 1; "improved" means final_rank < clean_rank. The benchmark's own `success` flag
(rank OR probability improved) is deliberately not used.
"""
import collections
import csv
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "benchmark_results", "layer_benchmark_20260909_144450_707979.json")
OUT = os.path.join(ROOT, "benchmark_results", "layer_benchmark_20260909_summary.csv")


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    per_layer = collections.defaultdict(list)
    kept, dropped = [], []
    for p in d["prompts"]:
        clean = int(p["layers"][0]["clean_rank"])
        (kept if clean > 1 else dropped).append((p["prompt"], p["target"], clean))
        if clean <= 1:
            continue
        for L in p["layers"]:
            per_layer[int(L["layer"])].append((int(L["clean_rank"]), int(L["final_rank"]),
                                               float(L["clean_probability"]), float(L["final_probability"])))
    rows = []
    for layer in sorted(per_layer):
        v = per_layer[layer]
        rows.append({
            "layer": layer, "prompts": len(v),
            "reached_rank1": sum(1 for c, f, _, _ in v if f == 1),
            "improved_rank": sum(1 for c, f, _, _ in v if f < c),
            "worsened_rank": sum(1 for c, f, _, _ in v if f > c),
            "mean_rank_gain": round(st.mean(c - f for c, f, _, _ in v), 3),
            "median_rank_gain": st.median(c - f for c, f, _, _ in v),
            "median_final_rank": st.median(f for _, f, _, _ in v),
            "mean_prob_gain_pct_points": round(100 * st.mean(fp - cp for _, _, cp, fp in v), 4),
        })
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("parameters:", d["parameters"], "| prompts in file:", len(d["prompts"]),
          "| summarised (clean rank > 1):", len(kept), "| excluded (already rank 1):", len(dropped))
    print("clean ranks of summarised prompts:", sorted(c for _, _, c in kept))
    print(f"{'layer':>5} {'n':>3} {'rank1':>6} {'improved':>9} {'worse':>6} {'mean gain':>10} {'median gain':>12} {'median final':>13} {'mean dP (pp)':>13}")
    for r in rows:
        print(f"{r['layer']:>5} {r['prompts']:>3} {r['reached_rank1']:>6} {r['improved_rank']:>9} {r['worsened_rank']:>6} "
              f"{r['mean_rank_gain']:>10} {r['median_rank_gain']:>12} {r['median_final_rank']:>13} {r['mean_prob_gain_pct_points']:>13}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
