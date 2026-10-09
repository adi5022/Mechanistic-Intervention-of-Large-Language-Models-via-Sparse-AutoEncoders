"""
Step A1 of docs/cross_model_transfer/PLAN.md: fit linear translators between GPT-2 small and GPT-2 medium, and score them on unseen text.

    .venv\\Scripts\\python.exe tools/transfer/01_fit_maps.py --selftest   # 5 seconds, no models: checks the maths on made-up data
    .venv\\Scripts\\python.exe tools/transfer/01_fit_maps.py --quick      # about a minute: 20,000 fit tokens, 10,000 held-out tokens
    .venv\\Scripts\\python.exe tools/transfer/01_fit_maps.py              # full: 500,000 fit tokens, 100,000 held-out tokens
    add --sgd-check to also train ONE pair by gradient descent and compare it with the exact solution (about 1 minute more)

What it does
 Pass 1  runs both models over the same wikitext-2 sequences and keeps running sums for 25 layer pairs (small layers 2,4,6,8,10 x
         medium layers 4,8,12,16,20, hook blocks.L.hook_resid_pre; the first token of each sequence is skipped).
 Pass 2  solves the ridge map for every pair, in BOTH directions (small->medium and medium->small), for 5 ridge strengths.
 Pass 3  runs both models over held-out text (never used for fitting) and scores every map. The held-out text is split in two halves:
         the ridge strength is chosen on half A, and the numbers reported are from half B.
Scores (held-out): per-dimension R-squared (the honest one: every dimension counts equally), raw R-squared (dominated by the largest
dimensions), and the cosine between predicted and true state after removing the mean. 0 = no better than predicting the average, 1 = perfect.

There is no pass/fail here: step A2 asks the questions that matter (does a model still work on a translated state, do translated
DIFFERENCES match). This table only tells us where to look.

Writes to outputs/transfer/ (ignored by git): maps_gpt2_to_gpt2-medium.pt (all 50 maps with their scores), maps_summary.json, maps_summary.csv
(files get a _quick suffix in --quick mode).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.transfer.maps import MapScore, RidgeSolver, TargetStats
from src.transfer.models import load_model, pick_device
from src.transfer.paired_acts import PairedStats, capture

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
SMALL, MEDIUM = "gpt2", "gpt2-medium"
SEQ = 128
POS_PER_SEQ = SEQ - 1                     # first token skipped


def parse_layers(s):
    return [int(v) for v in s.split(",") if v.strip()]


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0, self.next = total, label, time.time(), 0.1

    def update(self, done):
        if done / self.total >= self.next:
            el = time.time() - self.t0
            print(f"   {self.label}: {100 * done / self.total:3.0f}%  ({done:,} of {self.total:,} sequences, {el:.0f} s, about {el * (self.total - done) / max(done, 1):.0f} s left)", flush=True)
            self.next += 0.1


# ----------------------------------------------------------------------------------------------------------------------------------
def selftest():
    """Made-up data with a known linear map: the exact solution must find it, and the scores must match the known noise level."""
    print("SELFTEST: recover a known linear map from streamed sums")
    torch.manual_seed(0)
    dev = "cpu"
    dx, dy, n_batches, bs = 48, 64, 40, 500
    A = torch.randn(dx, dx) * 0.7 + torch.eye(dx)
    scale = torch.ones(dx); scale[:3] = 30.0                          # a few huge dimensions, like a real residual stream
    W_true, b_true = torch.randn(dx, dy), torch.randn(dy) * 5
    noise = 0.3
    stats = PairedStats({0: dx}, {0: dy}, dev)
    gen = lambda: ((torch.randn(bs, dx) @ A) * scale + 100.0)
    xs, ys = [], []
    for _ in range(n_batches):
        x = gen(); y = (x / scale) @ W_true + b_true + noise * torch.randn(bs, dy)
        stats.add({0: x}, {0: y}); xs.append(x); ys.append(y)
    c = stats.centered()
    solver = RidgeSolver(c["Sxx"][0])
    W = solver.solve_many(c["Sxy"][(0, 0)], [1e-9])[1e-9].float()
    # (1) exactness: the streamed solution must equal a direct least-squares solve on all the data held in memory at once
    X, Y = torch.cat(xs).double(), torch.cat(ys).double()
    W_direct = torch.linalg.lstsq(X - X.mean(0), Y - Y.mean(0)).solution.float()
    diff = (W - W_direct).abs().max().item()
    # (2) recovery: close to the true map, up to the statistical floor of this data (its condition number is about 126, measured 0.044)
    err = (W - W_true / scale[:, None]).abs().max().item()
    scorer, tgt = MapScore(dy, dev), TargetStats(dy, dev)
    for i in range(10):
        x = gen(); y = (x / scale) @ W_true + b_true + noise * torch.randn(bs, dy)
        tgt.add(i % 2, y); scorer.add(i % 2, x, y, W, c["mu_x"][0].float(), c["mu_y"][0].float())
    m = scorer.metrics(1, tgt.sst(1))
    print(f"   streamed solution vs direct least squares on the same data: largest difference {diff:.2e}   (limit 2e-3)")
    print(f"   streamed solution vs the true map: largest difference {err:.2e}   (limit 0.1; no method can do better than about 0.044 on this data)")
    print(f"   held-out per-dim R2 {m['perdim_r2']:.4f}, raw R2 {m['raw_r2']:.4f}, cosine {m['cosine']:.4f}   (all must be above 0.97)")
    ok = diff < 2e-3 and err < 0.1 and min(m["perdim_r2"], m["raw_r2"], m["cosine"]) > 0.97
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------------------------------------------
def grid(rows, direction, metric, src_layers, dst_layers):
    """Text grid: rows = source layer, columns = target layer."""
    lines = []
    if direction == "s2m":
        rl, cl, rn, cn = src_layers, dst_layers, "small", "medium"
    else:
        rl, cl, rn, cn = dst_layers, src_layers, "medium", "small"
    lines.append(f"   {rn} layer (rows) -> {cn} layer (columns)    " + "".join(f"{c:>8}" for c in cl))
    by = {(r["src_layer"], r["dst_layer"]): r for r in rows if r["dir"] == direction}
    for r_ in rl:      # in both directions the row layer belongs to the SOURCE model and the column layer to the TARGET model
        lines.append(f"   {r_:>34}    " + "".join(f"{by[(r_, c_)][metric]:8.3f}" for c_ in cl))
    return "\n".join(lines)


def sgd_check(small, medium, fit_ids, held_ids, dev, best, cs, a):
    """Train ONE small->medium pair by gradient descent (Adam) on a subset and compare with the exact ridge solution on the same subset."""
    l, m = best["src_layer"], best["dst_layer"]
    stage(f"SGD CHECK  small layer {l} -> medium layer {m}: Adam versus the exact solution, same {a.sgd_tokens:,} fit tokens")
    def collect(ids, tokens):
        xs, ys, got = [], [], 0
        for i in range(0, len(ids), a.batch):
            b = ids[i:i + a.batch].to(dev)
            xs.append(capture(small, b, [l])[l]); ys.append(capture(medium, b, [m])[m]); got += xs[-1].shape[0]
            if got >= tokens:
                break
        return torch.cat(xs)[:tokens], torch.cat(ys)[:tokens]
    X, Y = collect(fit_ids, a.sgd_tokens)
    Xh, Yh = collect(held_ids, a.sgd_tokens // 4)
    mx, my = X.mean(0), Y.mean(0)
    def perdim_r2(pred):
        sse = ((Yh - pred) ** 2).sum(0); sst = ((Yh - Yh.mean(0)) ** 2).sum(0).clamp(min=1e-12)
        return float((1 - sse / sst).mean())
    # exact
    Xc, Yc = (X - mx).double(), (Y - my).double()
    solver = RidgeSolver(Xc.T @ Xc)
    exact_by_c = {c: perdim_r2((Xh - mx) @ W.float() + my) for c, W in solver.solve_many(Xc.T @ Yc, cs).items()}
    c_best = max(exact_by_c, key=exact_by_c.get)
    exact_none, exact_best = exact_by_c[min(cs)], exact_by_c[c_best]
    # Adam on standardised data
    sx, sy = X.std(0).clamp(min=1e-6), Y.std(0).clamp(min=1e-6)
    Xs, Ys = (X - mx) / sx, (Y - my) / sy
    Wg = torch.zeros(X.shape[1], Y.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.Adam([Wg], lr=3e-3)
    steps, bs = a.sgd_steps, 4096
    g = torch.Generator(device="cpu").manual_seed(0)
    t0 = time.time()
    for s in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * (1 - s / steps)
        idx = torch.randint(0, Xs.shape[0], (bs,), generator=g).to(dev)
        loss = ((Xs[idx] @ Wg - Ys[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % max(steps // 5, 1) == 0:
            with torch.no_grad():
                cur = perdim_r2((((Xh - mx) / sx) @ Wg) * sy + my)
            print(f"   step {s + 1:>5}: train loss {loss.item():.4f}, held-out per-dim R2 {cur:.4f}", flush=True)
    with torch.no_grad():
        sgd = perdim_r2((((Xh - mx) / sx) @ Wg) * sy + my)
    print(f"   exact, almost no regularisation (c={min(cs):g}): {exact_none:.4f}")
    print(f"   exact, best ridge strength (c={c_best:g}):          {exact_best:.4f}")
    print(f"   gradient descent after {steps} steps ({time.time() - t0:.0f} s):  {sgd:.4f}")
    print("   (stopping gradient descent early acts like a ridge penalty, so it can beat the unregularised fit when data are few)")
    return {"pair": [l, m], "tokens": a.sgd_tokens, "exact_unregularised_perdim_r2": exact_none, "exact_best_ridge_perdim_r2": exact_best,
            "best_ridge_c": c_best, "sgd_perdim_r2": sgd, "sgd_steps": steps}


# ----------------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quick", action="store_true", help="20,000 fit tokens and 10,000 held-out tokens")
    ap.add_argument("--fit-tokens", type=int, default=500_000)
    ap.add_argument("--val-tokens", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=16, help="sequences per batch")
    ap.add_argument("--src-layers", default="2,4,6,8,10", help="small-model layers")
    ap.add_argument("--dst-layers", default="4,8,12,16,20", help="medium-model layers")
    ap.add_argument("--cs", default="1e-5,1e-4,1e-3,1e-2,1e-1", help="relative ridge strengths")
    ap.add_argument("--sgd-check", action="store_true")
    ap.add_argument("--sgd-tokens", type=int, default=60_000)
    ap.add_argument("--sgd-steps", type=int, default=1500)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.quick:
        a.fit_tokens, a.val_tokens, a.sgd_tokens, a.sgd_steps = 20_000, 10_000, 10_000, 300

    t_start = time.time()
    src_layers, dst_layers, cs = parse_layers(a.src_layers), parse_layers(a.dst_layers), [float(v) for v in a.cs.split(",")]
    dev = pick_device(a.device)
    suffix = "_quick" if a.quick else ""
    tok_path = os.path.join(OUT_DIR, f"wikitext2_tokens{suffix}.pt")
    if not os.path.exists(tok_path):
        print(f"missing {tok_path}: run step A0 first (tools/transfer/00_setup_check.py{' --quick' if a.quick else ''})")
        return 1
    data = torch.load(tok_path)
    if "train" in data:
        fit_pool, held_pool = data["train"], data["validation"]
    else:                                                     # the quick token file has validation text only
        fit_pool, held_pool = data["validation"], data["validation"].flip(0)
    fit_ids = fit_pool[: math.ceil(a.fit_tokens / POS_PER_SEQ)]
    held_ids = held_pool[: math.ceil(a.val_tokens / POS_PER_SEQ)]
    print(f"device {dev}; fit sequences {len(fit_ids):,} (about {len(fit_ids) * POS_PER_SEQ:,} tokens), held-out sequences {len(held_ids):,} (about {len(held_ids) * POS_PER_SEQ:,} tokens)")
    print(f"small layers {src_layers} x medium layers {dst_layers}; ridge strengths {cs}")

    stage("loading models")
    small, medium = load_model(SMALL, dev), load_model(MEDIUM, dev)
    dx, dy = small.cfg.d_model, medium.cfg.d_model
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # ---- pass 1: sums -----------------------------------------------------------------------------------------------------------
    stage("pass 1/3: fitting sums over the fit text (both models read the same tokens)")
    stats = PairedStats({l: dx for l in src_layers}, {m: dy for m in dst_layers}, dev)
    prog = Progress(len(fit_ids), "pass 1")
    for i in range(0, len(fit_ids), a.batch):
        b = fit_ids[i:i + a.batch].to(dev)
        stats.add(capture(small, b, src_layers), capture(medium, b, dst_layers))
        prog.update(i + len(b))
    cen = stats.centered()
    print(f"   {cen['n']:,} token positions used")

    # ---- pass 2: solve ------------------------------------------------------------------------------------------------------------
    stage("pass 2/3: solving 50 maps x %d ridge strengths (exact)" % len(cs))
    maps = {}       # (dir, src_layer, dst_layer) -> {c: (W, mu_src, mu_dst)}   [src_layer / dst_layer refer to the model that is source / target]
    for l in src_layers:
        solver = RidgeSolver(cen["Sxx"][l])
        for m in dst_layers:
            Ws = solver.solve_many(cen["Sxy"][(l, m)], cs)
            maps[("s2m", l, m)] = {c: (W.float(), cen["mu_x"][l].float(), cen["mu_y"][m].float()) for c, W in Ws.items()}
    for m in dst_layers:
        solver = RidgeSolver(cen["Syy"][m])
        for l in src_layers:
            Ws = solver.solve_many(cen["Sxy"][(l, m)].T.contiguous(), cs)
            maps[("m2s", m, l)] = {c: (W.float(), cen["mu_y"][m].float(), cen["mu_x"][l].float()) for c, W in Ws.items()}
    del stats, cen
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- pass 3: held-out scoring -------------------------------------------------------------------------------------------------
    stage("pass 3/3: scoring every map on held-out text (ridge strength chosen on half A, reported on half B)")
    tstat = {("small", l): TargetStats(dx, dev) for l in src_layers}
    tstat.update({("medium", m): TargetStats(dy, dev) for m in dst_layers})
    scores = {(k, c): MapScore(dy if k[0] == "s2m" else dx, dev) for k in maps for c in cs}
    prog = Progress(len(held_ids), "pass 3")
    for bi, i in enumerate(range(0, len(held_ids), a.batch)):
        half = bi % 2
        b = held_ids[i:i + a.batch].to(dev)
        X, Y = capture(small, b, src_layers), capture(medium, b, dst_layers)
        for l, x in X.items():
            tstat[("small", l)].add(half, x)
        for m, y in Y.items():
            tstat[("medium", m)].add(half, y)
        for (d, p, q), by_c in maps.items():
            x, y = (X[p], Y[q]) if d == "s2m" else (Y[p], X[q])
            for c, (W, mu_s, mu_t) in by_c.items():
                scores[((d, p, q), c)].add(half, x, y, W, mu_s, mu_t)
        prog.update(i + len(b))

    rows, saved = [], {}
    for (d, p, q), by_c in maps.items():
        tgt = tstat[("medium", q)] if d == "s2m" else tstat[("small", q)]
        best_c = max(cs, key=lambda c: scores[((d, p, q), c)].metrics(0, tgt.sst(0))["perdim_r2"])
        met = scores[((d, p, q), best_c)].metrics(1, tgt.sst(1))
        rows.append({"dir": d, "src_layer": p, "dst_layer": q, "c": best_c, **met})
        W, mu_s, mu_t = by_c[best_c]
        saved[f"{d}_L{p}_L{q}"] = {"W": W.cpu(), "mu_src": mu_s.cpu(), "mu_dst": mu_t.cpu(), "c": best_c, **met}

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"small": SMALL, "medium": MEDIUM, "fit_tokens": len(fit_ids) * POS_PER_SEQ, "held_tokens": len(held_ids) * POS_PER_SEQ,
            "ridge_strengths": cs, "src_layers": src_layers, "dst_layers": dst_layers, "hook": "resid_pre",
            "naming": "key = <dir>_L<layer of the source model>_L<layer of the target model>; s2m = small to medium, m2s = medium to small"}
    torch.save({"meta": meta, "maps": saved}, os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt"))
    with open(os.path.join(OUT_DIR, f"maps_summary{suffix}.json"), "w", encoding="utf8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    with open(os.path.join(OUT_DIR, f"maps_summary{suffix}.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    sgd = None
    if a.sgd_check:
        best = max((r for r in rows if r["dir"] == "s2m"), key=lambda r: r["perdim_r2"])
        sgd = sgd_check(small, medium, fit_ids, held_ids, dev, best, cs, a)
        with open(os.path.join(OUT_DIR, f"maps_sgd_check{suffix}.json"), "w", encoding="utf8") as f:
            json.dump(sgd, f, indent=1)

    # ---- result block -----------------------------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("RESULT  step A1 (translators)" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    print(f"  fitted on {meta['fit_tokens']:,} tokens, scored on {meta['held_tokens']:,} held-out tokens (half B). Scale: 0 = no better than the average state, 1 = perfect.")
    for d, title in (("s2m", "SMALL -> MEDIUM"), ("m2s", "MEDIUM -> SMALL")):
        for metric, name in (("perdim_r2", "per-dimension R-squared (honest)"), ("raw_r2", "raw R-squared"), ("cosine", "cosine of the direction")):
            print(f"\n  {title}: {name}")
            print(grid(rows, d, metric, src_layers, dst_layers))
        top = sorted((r for r in rows if r["dir"] == d), key=lambda r: -r["perdim_r2"])[:3]
        print(f"\n  {title}: best 3 pairs by per-dimension R-squared")
        for r in top:
            print(f"     source layer {r['src_layer']:>2} -> target layer {r['dst_layer']:>2}:  per-dim {r['perdim_r2']:.3f}  raw {r['raw_r2']:.3f}  cosine {r['cosine']:.3f}  (ridge c={r['c']:g})")
    hi = sum(r["c"] == max(cs) for r in rows)
    lo = sum(r["c"] == min(cs) for r in rows)
    print(f"\n  ridge strength chosen: the strongest value in the grid for {hi} of {len(rows)} maps, the weakest for {lo}"
          + ("   <-- many at the strongest edge: the best value may lie beyond the grid; tell me and we widen it" if hi > len(rows) // 2 and not a.quick else ""))
    if sgd:
        print(f"\n  SGD CHECK small {sgd['pair'][0]} -> medium {sgd['pair'][1]} (per-dim R2, same {sgd['tokens']:,} fit tokens): "
              f"exact unregularised {sgd['exact_unregularised_perdim_r2']:.4f}, exact best ridge {sgd['exact_best_ridge_perdim_r2']:.4f}, "
              f"gradient descent {sgd['sgd_perdim_r2']:.4f}")
    if dev.startswith("cuda"):
        print(f"\n  peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB;  total time {time.time() - t_start:.0f} s")
    print(f"  saved: outputs/transfer/maps_gpt2_to_gpt2-medium{suffix}.pt, maps_summary{suffix}.json, maps_summary{suffix}.csv")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
