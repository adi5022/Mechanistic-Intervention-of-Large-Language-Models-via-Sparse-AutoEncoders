"""
Step A2 of docs/cross_model_transfer/PLAN.md: do the translators actually work?  (Gate 1)

    .venv\\Scripts\\python.exe tools/transfer/02_check_maps.py --quick       # about a minute, a few pairs, small samples
    .venv\\Scripts\\python.exe tools/transfer/02_check_maps.py               # full: 8 layer pairs, 100 text sequences, about 210 sentence pairs
    add --feature-view to also test whether GPT-2 small's SAE feature directions survive a round trip through the maps

For each layer pair it asks two questions that a good R-squared cannot answer:

 1. STITCHING. Run the receiving model, but at the chosen layer replace its internal state with the TRANSLATED state of the other model
    (for the same text) and measure how well it still predicts the next word.
        clean  = receiver untouched      floor = its state replaced by its average state (information destroyed)
        wrong  = the translation of a DIFFERENT text (right kind of state, wrong content)
        recovered = (floor - stitched) / (floor - clean):  0 = information destroyed, 1 = as good as the real state.

 2. DIFFERENCES. Take two sentences that differ in one word ("The capital of France is" / "...Germany is"; 6 templates, about 210 pairs). Translate
    the first model's difference and compare it with the second model's real difference (cosine, 1 = same direction). Edits are sent as
    differences, so this is the number that matters most. Chance level comes from comparing a difference with a DIFFERENT pair's difference,
    once within the same template (hard) and once across all templates. Measured at the word's own position and at the last position.

Gate 1 (proposed in the plan; arguments below can change it, and it must not be changed after seeing results):
    recovered >= 0.8   AND   mean last-position difference cosine >= 0.3   AND   that mean is above the 95th percentile of the within-template chance level.

Pairs are written dir:source_layer:target_layer, where s2m = GPT-2 small to medium, m2s = medium to small (layers belong to the source and target model).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from src.transfer.minimal_pairs import build_groups
from src.transfer.models import load_model, pick_device
from src.transfer.stitch import loss_with_replacement, states, stitching

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
SMALL, MEDIUM = "gpt2", "gpt2-medium"
DEFAULT_PAIRS = "s2m:2:4,s2m:6:8,s2m:8:12,s2m:8:16,m2s:4:2,m2s:8:6,m2s:12:8,m2s:16:8"


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_loss(model, ids, batch, dev):
    tot, n = 0.0, 0
    for i in range(0, len(ids), batch):
        s, k = loss_with_replacement(model, ids[i:i + batch].to(dev), 0, None)
        tot += s; n += k
    return tot / n


def difference_fidelity(S, R, groups, l, m, W, dev, reps=20, seed=0):
    """Cosine between the translated difference of two sentences and the receiver's real difference, with chance levels."""
    g_ = torch.Generator().manual_seed(seed)
    res = {}
    kinds = ("entity", "last")
    dh_all, dr_all, within = {k: [] for k in kinds}, {k: [] for k in kinds}, {k: [] for k in kinds}
    for g in groups:
        tok = g["tokens"].to(dev)
        HS, HR = states(S, tok, l), states(R, tok, m)
        I = torch.tensor([p[0] for p in g["pairs"]], device=dev)
        J = torch.tensor([p[1] for p in g["pairs"]], device=dev)
        for kind, pos in (("entity", g["diff_pos"]), ("last", g["last_pos"])):
            dh = (HS[I, pos] - HS[J, pos]) @ W                      # translated difference (no bias: a difference has none)
            dr = HR[I, pos] - HR[J, pos]                            # the receiver's real difference
            dh_all[kind].append(dh); dr_all[kind].append(dr)
            P = len(I)
            for _ in range(reps):
                perm = torch.randperm(P, generator=g_).to(dev)
                keep = perm != torch.arange(P, device=dev)
                within[kind].append(F.cosine_similarity(dh[keep], dr[perm][keep], dim=1))
    for kind in kinds:
        dh, dr = torch.cat(dh_all[kind]), torch.cat(dr_all[kind])
        real = F.cosine_similarity(dh, dr, dim=1)
        w = torch.cat(within[kind])
        glob = []
        for _ in range(reps):
            perm = torch.randperm(len(dh), generator=g_).to(dev)
            keep = perm != torch.arange(len(dh), device=dev)
            glob.append(F.cosine_similarity(dh[keep], dr[perm][keep], dim=1))
        glob = torch.cat(glob)
        p95 = torch.quantile(w, 0.95).item()
        res[kind] = {
            "n_pairs": int(len(dh)),
            "mean_cos": real.mean().item(),
            "within_template_chance_mean": w.mean().item(), "within_template_chance_p95": p95,
            "across_templates_chance_mean": glob.mean().item(), "across_templates_chance_p95": torch.quantile(glob, 0.95).item(),
            "share_pairs_above_chance_p95": (real > p95).float().mean().item(),
            "relative_error": ((dh - dr).norm(dim=1) / dr.norm(dim=1).clamp(min=1e-9)).mean().item(),
            "scale_ratio": (dh.norm(dim=1) / dr.norm(dim=1).clamp(min=1e-9)).mean().item(),
        }
    return res


def feature_view(maps, dev):
    """Do GPT-2 small's SAE feature directions (layer 8) survive a round trip small -> medium -> small, compared with random directions?"""
    from src.sae_utils import load_sae_for_layer
    sae = load_sae_for_layer(8, dev)
    D = sae.W_dec.detach().float().to(dev)                          # [n_features, 768]
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(5000, D.shape[1], generator=g).to(dev)
    out = {}
    for key in maps:
        if not key.startswith("s2m_L8_L"):
            continue
        m = key.split("_L")[-1]
        back_key = f"m2s_L{m}_L8"
        if back_key not in maps:
            continue
        W1, W2 = maps[key]["W"].to(dev), maps[back_key]["W"].to(dev)
        rt = lambda X: F.cosine_similarity((X @ W1) @ W2, X, dim=1)
        cf, cr = rt(D), rt(rnd)
        q = lambda c: [torch.quantile(c, p).item() for p in (0.1, 0.5, 0.9)]
        out[f"small8_via_medium{m}"] = {"features_mean": cf.mean().item(), "features_p10_p50_p90": q(cf),
                                         "random_mean": cr.mean().item(), "random_p10_p50_p90": q(cr), "n_features": int(D.shape[0])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="comma-separated dir:source_layer:target_layer")
    ap.add_argument("--n-seq", type=int, default=100, help="text sequences for stitching")
    ap.add_argument("--per-template", type=int, default=35, help="sentence pairs per template")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--feature-view", action="store_true")
    ap.add_argument("--gate-recovered", type=float, default=0.8)
    ap.add_argument("--gate-cos", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    suffix = "_quick" if a.quick else ""
    if a.quick:
        a.n_seq, a.per_template = 16, 8
        a.pairs = "s2m:8:16,m2s:16:8"
    t0 = time.time()
    dev = pick_device(a.device)

    maps_path = os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt")
    tok_path = os.path.join(OUT_DIR, f"wikitext2_tokens{suffix}.pt")
    for p in (maps_path, tok_path):
        if not os.path.exists(p):
            print(f"missing {p}: run steps A0 and A1 first"); return 1
    blob = torch.load(maps_path)
    maps = blob["maps"]
    val = torch.load(tok_path)["validation"]
    ids = val[1000:1000 + a.n_seq]                                  # sequences A1 never touched (it used the first 788)
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]

    stage("loading models and building sentence pairs")
    models = {"s": load_model(SMALL, dev), "m": load_model(MEDIUM, dev)}
    groups = build_groups(models["s"], per_template=a.per_template)
    assert models["s"].tokenizer.get_vocab() == models["m"].tokenizer.get_vocab()
    print("   templates kept:", ", ".join(f"{g['name']} ({len(g['pairs'])} pairs)" for g in groups))
    clean = {"s": clean_loss(models["s"], ids, a.batch, dev), "m": clean_loss(models["m"], ids, a.batch, dev)}
    print(f"   clean next-word loss on the test text: small {clean['s']:.3f}, medium {clean['m']:.3f}")

    results = []
    for d, p, q in pairs:
        p, q = int(p), int(q)
        key = f"{d}_L{p}_L{q}"
        if key not in maps:
            print(f"   no map {key}, skipping"); continue
        S, R = (models["s"], models["m"]) if d == "s2m" else (models["m"], models["s"])
        rcv = "m" if d == "s2m" else "s"
        mp = maps[key]
        W, mu_s, mu_t = mp["W"].to(dev), mp["mu_src"].to(dev), mp["mu_dst"].to(dev)
        stage(f"pair {key}: A1 scores per-dim R2 {mp['perdim_r2']:.3f}, cosine {mp['cosine']:.3f}")
        st = stitching(S, R, ids, p, q, W, mu_s, mu_t, batch=a.batch, clean=clean[rcv])
        print(f"   stitching: clean {st['clean']:.3f}  stitched {st['stitched']:.3f}  floor {st['floor']:.3f}  wrong-text {st.get('wrong', float('nan')):.3f}  -> recovered {st['recovered']:.3f}")
        df = difference_fidelity(S, R, groups, p, q, W, dev)
        for kind in ("entity", "last"):
            x = df[kind]
            print(f"   differences at the {kind:>6} position: cosine {x['mean_cos']:.3f}   chance within template {x['within_template_chance_mean']:.3f} (95th pct {x['within_template_chance_p95']:.3f}), "
                  f"across templates {x['across_templates_chance_mean']:.3f};  {100 * x['share_pairs_above_chance_p95']:.0f}% of pairs above chance;  size ratio {x['scale_ratio']:.2f}")
        last = df["last"]
        gate = st["recovered"] >= a.gate_recovered and last["mean_cos"] >= a.gate_cos and last["mean_cos"] > last["within_template_chance_p95"]
        results.append({"key": key, "a1_perdim_r2": mp["perdim_r2"], "a1_cosine": mp["cosine"], "stitching": st, "differences": df, "gate1_pass": bool(gate)})

    fv = None
    if a.feature_view:
        stage("feature view: do SAE feature directions of small layer 8 survive a round trip through the maps?")
        fv = feature_view(maps, dev)
        for k, v in fv.items():
            print(f"   {k}: cosine after round trip, SAE features mean {v['features_mean']:.3f} (10/50/90th pct {v['features_p10_p50_p90'][0]:.2f}/{v['features_p10_p50_p90'][1]:.2f}/{v['features_p10_p50_p90'][2]:.2f})"
                  f"  versus random directions mean {v['random_mean']:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"a2_check{suffix}.json"), "w", encoding="utf8") as f:
        json.dump({"gate": {"recovered": a.gate_recovered, "cos": a.gate_cos}, "clean_loss": clean, "n_seq": a.n_seq,
                   "results": results, "feature_view": fv}, f, indent=1)

    print("\n" + "=" * 118)
    print("RESULT  step A2 (do the translators work?)" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    print(f"  Gate 1 (proposed): recovered >= {a.gate_recovered}  AND  last-position difference cosine >= {a.gate_cos} and above the within-template chance 95th percentile")
    print(f"  {'pair':<12}{'A1 R2':>7}{'clean':>7}{'stitch':>8}{'floor':>7}{'wrong':>7}{'recovered':>11} | {'diff cos (last)':>16}{'chance p95':>11}{'above':>7} | {'diff cos (word)':>16}{'chance p95':>11} | gate")
    for r in results:
        st, dl, de = r["stitching"], r["differences"]["last"], r["differences"]["entity"]
        print(f"  {r['key']:<12}{r['a1_perdim_r2']:>7.3f}{st['clean']:>7.2f}{st['stitched']:>8.2f}{st['floor']:>7.2f}{st.get('wrong', float('nan')):>7.2f}{st['recovered']:>11.3f} | "
              f"{dl['mean_cos']:>16.3f}{dl['within_template_chance_p95']:>11.3f}{100 * dl['share_pairs_above_chance_p95']:>6.0f}% | {de['mean_cos']:>16.3f}{de['within_template_chance_p95']:>11.3f} | {'PASS' if r['gate1_pass'] else 'no'}")
    n_pass = sum(r["gate1_pass"] for r in results)
    print(f"\n  {n_pass} of {len(results)} pairs pass the proposed gate.   total time {time.time() - t0:.0f} s")
    print("  loss columns are next-word cross-entropy (lower = better): clean = receiver untouched, floor = information destroyed, wrong = another text's translation")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    sys.exit(main())
