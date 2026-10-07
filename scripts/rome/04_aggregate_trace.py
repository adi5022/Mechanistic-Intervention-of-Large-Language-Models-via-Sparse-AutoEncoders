import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[2]
cache = REPO / "data" / "comparison" / "raw" / "trace"
outdir = REPO / "docs" / "rome_baseline" / "tracing"
outdir.mkdir(parents=True, exist_ok=True)
KINDS = ("resid", "mlp", "attn")
CMAPS = {"resid": "Purples", "mlp": "Greens", "attn": "Reds"}
ROLES = ["first subject", "middle subject", "last subject", "later tokens", "last token"]


def by_role(scores, low, rng):
    s0, s1 = int(rng[0]), int(rng[1])
    n = scores.shape[0]
    eff = scores - low
    out = np.full((5, eff.shape[1]), np.nan)
    if s1 - s0 >= 2:
        out[0] = eff[s0]
    if s1 - s0 >= 3:
        out[1] = eff[s0 + 1 : s1 - 1].mean(0)
    out[2] = eff[s1 - 1]
    if n - 1 > s1:
        out[3] = eff[s1 : n - 1].mean(0)
    out[4] = eff[n - 1]
    return out


ids = sorted({int(p.stem.split("_")[1]) for p in cache.glob("fact_*_resid.npz")})
ids = [i for i in ids if all((cache / f"fact_{i}_{k}.npz").exists() for k in KINDS)]
agg = {k: [] for k in KINDS}
highs, lows = [], []
for i in ids:
    for k in KINDS:
        d = np.load(cache / f"fact_{i}_{k}.npz")
        agg[k].append(by_role(d["scores"], float(d["low"]), d["subject_range"]))
        if k == "resid":
            highs.append(float(d["high"]))
            lows.append(float(d["low"]))
mean = {k: np.nanmean(np.stack(v), axis=0) for k, v in agg.items()}
nl = mean["resid"].shape[1]
print("facts:", len(ids), " mean clean p:", round(np.mean(highs), 3), " mean corrupted p:", round(np.mean(lows), 3))

fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, k in zip(axes, KINDS):
    im = ax.imshow(mean[k], cmap=CMAPS[k], aspect="auto")
    ax.set_xticks(range(nl))
    ax.set_yticks(range(5))
    ax.set_yticklabels(ROLES)
    ax.set_xlabel("layer")
    ax.set_title({"resid": "Residual stream", "mlp": "MLP (3-layer window)", "attn": "Attention (3-layer window)"}[k])
    fig.colorbar(im, ax=ax, label="avg gain in p(answer)")
fig.suptitle(f"GPT-2 small causal tracing, averaged over {len(ids)} facts")
fig.tight_layout()
fig.savefig(outdir / "average_trace.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.6))
for k in KINDS:
    ax.plot(range(nl), mean[k][2], marker="o", label=k)
ax.set_xticks(range(nl))
ax.set_xlabel("layer")
ax.set_ylabel("avg gain in p(answer)")
ax.set_title("Last subject token")
ax.legend()
fig.tight_layout()
fig.savefig(outdir / "last_subject_token_lines.png", dpi=150)
plt.close(fig)

print("last subject token, avg gain by layer")
print("layer  resid    mlp    attn")
for L in range(nl):
    print(f"{L:5d}  {mean['resid'][2, L]:.4f}  {mean['mlp'][2, L]:.4f}  {mean['attn'][2, L]:.4f}")
top = np.argsort(-mean["mlp"][2])[:3].tolist()
print("top 3 MLP layers at last subject token:", top)

summary = dict(
    n_facts=len(ids),
    mean_clean_p=float(np.mean(highs)),
    mean_corrupted_p=float(np.mean(lows)),
    roles=ROLES,
    effects={k: np.round(mean[k], 5).tolist() for k in KINDS},
    top_mlp_layers_last_subject=top,
)
(REPO / "data" / "comparison" / "trace_summary.json").write_text(json.dumps(summary, indent=1))
print("saved plots to", outdir)
