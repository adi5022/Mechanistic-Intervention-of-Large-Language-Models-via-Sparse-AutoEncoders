import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(REPO / "benchmark_results" / "rome_vs_scalpel" / "layer_sweep_dev.csv")))
labels = ["no edit" if r["layer"] == "none" else f"layer {r['layer']}" for r in rows]
x = np.arange(len(rows))
w = 0.2
fig, ax = plt.subplots(figsize=(8, 4))
for i, (k, name) in enumerate([("ES", "efficacy"), ("PS", "paraphrase"), ("NS", "neighborhood"), ("S", "overall S")]):
    ax.bar(x + (i - 1.5) * w, [float(r[k]) for r in rows], w, label=name)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.set_title(f"ROME on GPT-2 small, edit-layer sweep (dev set, n={rows[0]['n']})")
ax.legend(ncol=4, loc="lower center")
fig.tight_layout()
out = REPO / "docs" / "rome_baseline" / "layer_sweep.png"
fig.savefig(out, dpi=150)
print("saved", out)
