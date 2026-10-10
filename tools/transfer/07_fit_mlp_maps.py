"""
Neural translator versus linear translator (answers "do we need a neural network?" with the same checks as step A2).

    .venv\\Scripts\\python.exe tools/transfer/07_fit_mlp_maps.py --quick
    .venv\\Scripts\\python.exe tools/transfer/07_fit_mlp_maps.py          # about 15 minutes

Trains, on the same wikitext-2 text and the same held-out text as step A1, a residual MLP map (src/transfer/mlp_maps.py, starting at the linear
solution) for three layer pairs: small 8 -> medium 12, small 8 -> medium 16 (Export) and medium 16 -> small 8 (Import). Then, side by side with the
linear map, on text neither has seen:
  held-out per-dimension R-squared | stitching (the receiver runs on the translated state; loss recovered) | difference cosine at the changed
  word and at the last position, with the within-template chance 95th percentile (translated change = f(h_a) - f(h_b)).
Writes outputs/transfer/mlp_maps.pt (weights) and mlp_vs_linear.json.
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
import torch.nn.functional as F

from src.transfer.minimal_pairs import build_groups
from src.transfer.mlp_maps import ResidualMlpMap, fit, perdim_r2
from src.transfer.models import load_model, pick_device
from src.transfer.paired_acts import capture
from src.transfer.stitch import loss_with_replacement, states

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
PAIRS = [("s2m", 8, 12), ("s2m", 8, 16), ("m2s", 16, 8)]            # (direction, source layer, target layer)


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect(small, medium, ids, batch, dev):
    X8, Y12, Y16 = [], [], []
    for i in range(0, len(ids), batch):
        b = ids[i:i + batch].to(dev)
        X8.append(capture(small, b, [8])[8].cpu())
        y = capture(medium, b, [12, 16])
        Y12.append(y[12].cpu()); Y16.append(y[16].cpu())
    return torch.cat(X8), torch.cat(Y12), torch.cat(Y16)


def diff_fidelity(S, R, groups, l, m, fn, dev, reps=20, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    res = {}
    dh_all, dr_all, within = {"entity": [], "last": []}, {"entity": [], "last": []}, {"entity": [], "last": []}
    with torch.no_grad():
        for g in groups:
            tok = g["tokens"].to(dev)
            HS, HR = states(S, tok, l), states(R, tok, m)
            FS = fn(HS)
            I = torch.tensor([p[0] for p in g["pairs"]], device=dev)
            J = torch.tensor([p[1] for p in g["pairs"]], device=dev)
            for kind, pos in (("entity", g["diff_pos"]), ("last", g["last_pos"])):
                dh, dr = FS[I, pos] - FS[J, pos], HR[I, pos] - HR[J, pos]
                dh_all[kind].append(dh); dr_all[kind].append(dr)
                P = len(I)
                for _ in range(reps):
                    perm = torch.randperm(P, generator=g_).to(dev)
                    keep = perm != torch.arange(P, device=dev)
                    within[kind].append(F.cosine_similarity(dh[keep], dr[perm][keep], dim=1))
    for kind in ("entity", "last"):
        dh, dr = torch.cat(dh_all[kind]), torch.cat(dr_all[kind])
        real = F.cosine_similarity(dh, dr, dim=1)
        p95 = torch.quantile(torch.cat(within[kind]), 0.95).item()
        res[kind] = {"cos": real.mean().item(), "chance_p95": p95, "share_above": (real > p95).float().mean().item(), "size_ratio": (dh.norm(dim=1) / dr.norm(dim=1).clamp(min=1e-9)).mean().item()}
    return res


def stitch_recovered(S, R, ids, l, m, fn, dev, mu_floor, clean, batch=16):
    st = fl = n = 0.0
    k = 0
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            b = ids[i:i + batch].to(dev)
            repl = fn(states(S, b, l))
            s_, c = loss_with_replacement(R, b, m, repl); st += s_; k += c
            fl += loss_with_replacement(R, b, m, mu_floor.expand_as(repl))[0]
    stitched, floor = st / k, fl / k
    return {"stitched": stitched, "floor": floor, "clean": clean, "recovered": (floor - stitched) / max(floor - clean, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fit-tokens", type=int, default=400_000)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    suffix = "_quick" if a.quick else ""
    if a.quick:
        a.fit_tokens, a.epochs, a.hidden = 20_000, 3, 256
    dev = pick_device(a.device)
    t0 = time.time()
    data = torch.load(os.path.join(OUT_DIR, f"wikitext2_tokens{suffix}.pt"))
    maps = torch.load(os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt"))["maps"]
    fit_pool, held_pool = (data["train"], data["validation"]) if "train" in data else (data["validation"], data["validation"].flip(0))
    fit_ids = fit_pool[: math.ceil(a.fit_tokens / 127)]
    held_ids = held_pool[: math.ceil(10_000 / 127) if a.quick else 788]
    small, medium = load_model("gpt2", dev), load_model("gpt2-medium", dev)

    stage(f"collecting states: {len(fit_ids)} fitting sequences, {len(held_ids)} held-out sequences")
    X8, Y12, Y16 = collect(small, medium, fit_ids, 16, dev)
    hX8, hY12, hY16 = collect(small, medium, held_ids, 16, dev)
    print(f"   {len(X8):,} fitting tokens, {len(hX8):,} held-out tokens", flush=True)
    half = len(hX8) // 2
    data_by = {"x": {"s": X8, "m12": Y12, "m16": Y16}, "h": {"s": hX8, "m12": hY12, "m16": hY16}}
    groups = build_groups(small, per_template=8 if a.quick else 35)
    ids_stitch = held_pool[1000:1000 + (16 if a.quick else 100)] if "train" in data else held_pool[1000:1016]
    clean = {}
    for nm, mdl in (("small", small), ("medium", medium)):
        tot = n = 0
        for i in range(0, len(ids_stitch), 16):
            s_, c = loss_with_replacement(mdl, ids_stitch[i:i + 16].to(dev), 0, None); tot += s_; n += c
        clean[nm] = tot / n

    saved, report = {}, {}
    for d, l, m in PAIRS:
        key = f"{d}_L{l}_L{m}"
        stage(f"{key}: train the neural map")
        src_k, dst_k = ("s", "m12" if m == 12 else "m16") if d == "s2m" else ("m16", "s")
        X, Y = data_by["x"][src_k], data_by["x"][dst_k]
        hX, hY = data_by["h"][src_k].to(dev), data_by["h"][dst_k].to(dev)
        W = maps[key]["W"].to(dev)
        mu_x, mu_y = maps[key]["mu_src"].to(dev), maps[key]["mu_dst"].to(dev)
        sig_x, sig_y = X.std(0).to(dev), Y.std(0).to(dev)
        model = ResidualMlpMap(W, mu_x, mu_y, sig_x, sig_y, hidden=a.hidden).to(dev)
        with torch.no_grad():
            lin_b = perdim_r2((hX[half:] - mu_x) @ W + mu_y, hY[half:])
        state, best_a = fit(model, X, Y, hX[:half], hY[:half], epochs=a.epochs, log=lambda s: print(s, flush=True))
        model.eval()
        with torch.no_grad():
            mlp_b = perdim_r2(torch.cat([model(hX[half:][i:i + 8192]) for i in range(0, len(hX) - half, 8192)]), hY[half:])
        print(f"   held-out half B per-dimension R2: linear {lin_b:.4f}   neural {mlp_b:.4f}", flush=True)
        saved[key] = {"state": {k: v.cpu() for k, v in state.items()}, "dx": W.shape[0], "dy": W.shape[1], "hidden": a.hidden}

        S, R = (small, medium) if d == "s2m" else (medium, small)
        floor = (maps[key]["mu_dst"]).to(dev)
        rcv_clean = clean["medium"] if d == "s2m" else clean["small"]
        lin_fn = lambda h: (h - mu_x) @ W + mu_y
        mlp_fn = lambda h: model(h)
        row = {"r2_linear": lin_b, "r2_neural": mlp_b}
        for nm, fn in (("linear", lin_fn), ("neural", mlp_fn)):
            row[nm] = {"stitching": stitch_recovered(S, R, ids_stitch, l, m, fn, dev, floor, rcv_clean),
                       "differences": diff_fidelity(S, R, groups, l, m, fn, dev)}
        report[key] = row
        print(f"   stitching recovered: linear {row['linear']['stitching']['recovered']:.3f}, neural {row['neural']['stitching']['recovered']:.3f}", flush=True)
        for kind in ("entity", "last"):
            print(f"   difference cosine at the {kind:>6}: linear {row['linear']['differences'][kind]['cos']:.3f}, neural {row['neural']['differences'][kind]['cos']:.3f} "
                  f"(chance 95th pct {row['linear']['differences'][kind]['chance_p95']:.3f} / {row['neural']['differences'][kind]['chance_p95']:.3f})", flush=True)
        del model
        torch.cuda.empty_cache() if dev.startswith("cuda") else None

    torch.save(saved, os.path.join(OUT_DIR, f"mlp_maps{suffix}.pt"))
    json.dump(report, open(os.path.join(OUT_DIR, f"mlp_vs_linear{suffix}.json"), "w", encoding="utf8"), indent=1)
    print("\n" + "=" * 112)
    print("RESULT  neural translator versus linear translator" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    print(f"  {'pair':<12}{'R2 lin':>8}{'R2 nn':>8} | {'stitch lin':>11}{'nn':>7} | {'word cos lin':>13}{'nn':>7} | {'last cos lin':>13}{'nn':>7}{'chance p95':>11}")
    for key, r in report.items():
        print(f"  {key:<12}{r['r2_linear']:>8.3f}{r['r2_neural']:>8.3f} | {r['linear']['stitching']['recovered']:>11.3f}{r['neural']['stitching']['recovered']:>7.3f} | "
              f"{r['linear']['differences']['entity']['cos']:>13.3f}{r['neural']['differences']['entity']['cos']:>7.3f} | {r['linear']['differences']['last']['cos']:>13.3f}"
              f"{r['neural']['differences']['last']['cos']:>7.3f}{r['linear']['differences']['last']['chance_p95']:>11.3f}")
    print(f"\n  total time {time.time() - t0:.0f} s; weights in outputs/transfer/mlp_maps{suffix}.pt")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    sys.exit(main())
