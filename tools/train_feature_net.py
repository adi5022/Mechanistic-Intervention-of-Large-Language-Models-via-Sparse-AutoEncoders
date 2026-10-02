"""
Train and evaluate the PER-FEATURE model (version 2; src/feature_models.py, docs/Research_Journal/23.md section 6.3).

Every candidate feature of a prompt gets its own multiplier from a small shared network (FeatureNet). Needs the same data as
version 1: outputs/strength_cache, outputs/strength_tables, data/counterfact_split.json. NO new data collection.

It compares, on the SAME one-shot proxy (all multipliers applied together, no step-by-step addition, no pool refill):
  reference            mute 0.6 / boost 0.5 on the strict-filter survivors (version 1 rule, all survivors at once)
  v1 learned pair      the single pair trained in version 1 (from --v1-export)
  v1 PromptNet         the per-prompt strengths trained in version 1 (from --v1-export)
  FeatureNet           per-feature multipliers from the new network (no extra filter: the filter is learned)
  FeatureNet + guard   the same, then every feature the strict rule would reject is dropped
  free per-feature     multipliers optimised on each evaluation prompt (upper bound for this loss; not a trained model)
and reports paired comparisons (exact McNemar test on rank 1) on the 300 test prompts and on all 450 held-out prompts.
Everything is the PROXY, not the real sweep.

    .venv\\Scripts\\python.exe tools/train_feature_net.py --quick       (a few minutes: smoke test)
    .venv\\Scripts\\python.exe tools/train_feature_net.py                (sizes 250, 500, 1000; 3 seeds)
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from argparse import Namespace
from datetime import datetime

import torch
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device  # noqa: E402
from src import strength_models as sm  # noqa: E402
from src import feature_models as fm  # noqa: E402
from src.hooks import make_scale_map_hook  # noqa: E402


def check_edit_equals_hook(model, sae, items, W_dec, dev):
    """A random per-feature edit applied through the cache path must equal the same edit through the real hook."""
    bt = sm.make_batch(items[:5], W_dec, dev)
    g = torch.Generator().manual_seed(0)
    a = (torch.rand(bt["B"], bt["K"], generator=g) * 2.0 - 0.6).to(dev)
    a = a * (torch.rand(bt["B"], bt["K"], generator=g) < 0.2).to(dev) * bt["kmask"]          # 20% of the candidates, offsets -0.6 to 1.4
    with torch.no_grad():
        fast = sm.forward_last(model, bt, fm.feature_delta(bt, a))
        worst, same_rank = 0.0, 0
        for i in range(bt["B"]):
            it = items[i]
            scale = {int(f): 1.0 + float(a[i, k]) for k, f in enumerate(it["ids"].tolist()) if abs(float(a[i, k])) > 0}
            model.reset_hooks()
            model.add_hook("blocks.8.hook_resid_pre", make_scale_map_hook(scale, sae))
            real = model(model.to_tokens(it["prompt"]))[0, -1]
            model.reset_hooks()
            worst = max(worst, float((real - fast[i]).abs().max()))
            rk = lambda l: int((torch.softmax(l, -1) > torch.softmax(l, -1)[it["target_id"]]).sum()) + 1
            same_rank += int(rk(real) == rk(fast[i]))
        clean = sm.clean_last_logits(model, bt)
        d_c = max(abs(float(torch.softmax(clean[i], -1)[bt["tgt"][i]]) - items[i]["baseline_prob"]) for i in range(bt["B"]))
    net = fm.make_feature_net(items, 32, 0.1, True, dev).eval()
    with torch.no_grad():
        a0 = net(bt)
    print(f"Check (a) random per-feature edit: cache path vs your real hook, 5 prompts: max logit diff {worst:.1e}, same target rank {same_rank}/5")
    print(f"Check (b) baseline probability from the saved state vs the cache: max diff {d_c:.1e}")
    print(f"Check (c) the untrained FeatureNet makes (almost) no edit: largest |a| = {float(a0.abs().max()):.4f}")
    if worst > 5e-3 or same_rank < 5 or d_c > 1e-3 or float(a0.abs().max()) > 0.05:
        sys.exit("A check failed; stopping.")


def mcnemar(res, x, y, ids):
    b = sum(1 for c in ids if res[x][c] == 1 and res[y][c] != 1)
    cc = sum(1 for c in ids if res[x][c] != 1 and res[y][c] == 1)
    both = sum(1 for c in ids if res[x][c] == 1 and res[y][c] == 1)
    p = binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0
    return b, cc, both, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs", "strength_cache"))
    ap.add_argument("--tables-dir", default=os.path.join(ROOT, "outputs", "strength_tables"))
    ap.add_argument("--split-file", default=os.path.join(ROOT, "data", "counterfact_split.json"))
    ap.add_argument("--v1-export", default=os.path.join(ROOT, "docs", "Research_Journal", "packs", "strength_models", "strengths_export.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sizes", default="250,500,1000")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lam-kl", type=float, default=3.0, help="side-effect weight (version 1 used 10; with 10 the per-feature network barely learned)")
    ap.add_argument("--lam-size-feat", type=float, default=0.005, help="weight on sum |a_k| w_k (prefer few, small changes)")
    ap.add_argument("--no-filter-inputs", action="store_true", help="hide the strict filter's verdict from the network's inputs")
    ap.add_argument("--oracle-steps", type=int, default=100)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.sizes, a.seeds, a.epochs, a.oracle_steps = "250", 1, 10, 30
    sizes = [int(x) for x in a.sizes.split(",")]
    out_dir = a.out_dir or os.path.join(ROOT, "outputs", "feature_models", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    dev = get_default_device()
    model = load_base_model(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sae = load_sae_for_layer(layer=8)
    W = sae.W_dec.detach()
    print(f"Device {dev} | torch {torch.__version__} | sizes {sizes} | seeds {a.seeds} | out {out_dir}", flush=True)
    split = json.load(open(a.split_file, encoding="utf-8"))
    t0 = time.time()
    load = lambda name, ids: [sm.load_item(i, name, a.cache_dir, a.tables_dir) for i in ids]
    train_all = load("train", split["train"][:max(sizes)])
    sets = {"val": load("val", split["val"]), "test_seen": load("test_seen", split["test_seen"]), "test_unseen": load("test_unseen", split["test_unseen"])}
    held = sets["val"] + sets["test_seen"] + sets["test_unseen"]
    print(f"Loaded {len(train_all)} training + {len(held)} evaluation prompts in {time.time() - t0:.0f} s", flush=True)
    check_edit_equals_hook(model, sae, sets["val"], W, dev)

    ids = [it["case_id"] for it in held]
    which = {it["case_id"]: it["set"] for it in held}
    per_prompt = {}            # method -> {case_id: rank}
    per_prompt_kl = {}
    summary = {}

    # ---- version 1 rules under the same one-shot proxy ----
    v1 = json.load(open(a.v1_export))
    a1 = Namespace(batch=a.batch, prefix_list=[0], train_prefix_list=[0], margin=a.margin, lam_kl=a.lam_kl, lam_size=0.02)
    def per_case(table):
        return lambda bt: (torch.tensor([table[str(c)]["mu"] for c in bt["case_ids"]], device=dev),
                           torch.tensor([table[str(c)]["beta"] for c in bt["case_ids"]], device=dev))
    const = lambda mu, be: (lambda bt: (torch.full((bt["B"],), mu, device=dev), torch.full((bt["B"],), be, device=dev)))
    v1_rules = {"reference 0.6/0.5": const(0.6, 0.5), "v1 learned pair": const(v1["constant"]["mu"], v1["constant"]["beta"]),
                "v1 PromptNet": per_case(v1["promptnet"]), "v1 per-prompt tuned": per_case(v1["oracle"])}
    print(f"\nVersion 1 rules under the one-shot proxy (learned pair = mute {v1['constant']['mu']:.2f} / boost {v1['constant']['beta']:.2f}):", flush=True)
    for name, fn in v1_rules.items():
        out = sm.evaluate(model, fn, held, a1, W, dev)
        per_prompt[name] = dict(zip(out["case_id"], out["rank"]))
        per_prompt_kl[name] = dict(zip(out["case_id"], out["kl"]))
        summary[name] = {s: sum(1 for c, r in per_prompt[name].items() if which[c] == s and r == 1) / len(sets[s]) for s in sets}
        print(f"   {name:<22} " + " ".join(f"{s} {100 * summary[name][s]:5.1f}%" for s in sets), flush=True)

    # ---- train the per-feature network ----
    results = {"args": vars(a), "runs": []}
    best = {}
    for size in sizes:
        for seed in range(a.seeds):
            t1 = time.time()
            print(f"\n[FeatureNet] size {size} seed {seed}", flush=True)
            net, hist = fm.train_feature_net(model, train_all[:size], sets["val"], a, W, dev, seed, log=lambda m: print(m, flush=True))
            row = {"size": size, "seed": seed, "epochs": len(hist), "seconds": round(time.time() - t1, 1), "history": hist}
            for guard in (False, True):
                for name, items in {"train": train_all[:size], **sets}.items():
                    r = fm.evaluate_feature(model, net, items, a, W, dev, guard=guard)
                    row[("guard_" if guard else "") + name] = fm.summarise_feature(r)
            results["runs"].append(row)
            v = row["val"]
            print(f"   done in {row['seconds']} s ({row['epochs']} epochs) | val {100 * v['rank1_rate']:.1f}% | KL(succ) {v['mean_kl_of_successes']:.3f} | "
                  f"features changed {v['mean_n_mute']:.0f} muted / {v['mean_n_boost']:.0f} boosted | with guard val {100 * row['guard_val']['rank1_rate']:.1f}%", flush=True)
            if size not in best or v["rank1_rate"] > best[size][0]["val"]["rank1_rate"]:
                best[size] = (row, net)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, default=float)

    row, net = best[max(sizes)]
    for guard, name in ((False, "FeatureNet"), (True, "FeatureNet + guard")):
        out = fm.evaluate_feature(model, net, held, a, W, dev, guard=guard)
        per_prompt[name] = dict(zip(out["case_id"], out["rank"]))
        per_prompt_kl[name] = dict(zip(out["case_id"], out["kl"]))
        summary[name] = {s: sum(1 for c, r in per_prompt[name].items() if which[c] == s and r == 1) / len(sets[s]) for s in sets}
    mult = {}                                   # exported multipliers of the selected network (non-trivial ones only)
    with torch.no_grad():
        for chunk in sm.iter_batches(held, a.batch):
            bt = sm.make_batch(chunk, W, dev)
            av = net(bt)
            for i, it in enumerate(chunk):
                k = it["ids"].numel()
                mult[str(it["case_id"])] = {str(f): round(float(x), 3) for f, x in zip(it["ids"].tolist(), av[i, :k].tolist()) if abs(x) > 0.02}
    json.dump({"multipliers_a": mult, "note": "multiplier = 1 + a; only |a| > 0.02 stored", "size": row["size"], "seed": row["seed"]},
              open(os.path.join(out_dir, "feature_multipliers.json"), "w"))
    torch.save({"state_dict": net.state_dict(), "args": vars(a), "size": row["size"], "seed": row["seed"]}, os.path.join(out_dir, "featurenet_best.pt"))

    if not a.skip_oracle:
        t1 = time.time()
        out = fm.oracle_feature(model, held, a, W, dev, steps=a.oracle_steps)
        per_prompt["free per-feature"] = dict(zip(out["case_id"], out["rank"]))
        per_prompt_kl["free per-feature"] = dict(zip(out["case_id"], out["kl"]))
        summary["free per-feature"] = {s: sum(1 for c, r in per_prompt["free per-feature"].items() if which[c] == s and r == 1) / len(sets[s]) for s in sets}
        print(f"\nFree per-feature (tuned on each evaluation prompt) done in {time.time() - t1:.0f} s", flush=True)

    # ---- summary ----
    test_ids = [c for c in ids if which[c] != "val"]
    print("\n==================== SUMMARY (one-shot proxy; rank-1 rate on held-out prompts) ====================")
    print(f"{'rule':<24}{'val':>8}{'test_seen':>11}{'test_unseen':>13}{'TEST(300)':>11}{'ALL(450)':>10}   KL(succ)")
    for name in per_prompt:
        r = per_prompt[name]
        n_t = sum(1 for c in test_ids if r[c] == 1)
        n_a = sum(1 for c in ids if r[c] == 1)
        ks = [per_prompt_kl[name][c] for c in ids if r[c] == 1]
        print(f"{name:<24}{100 * summary[name]['val']:>7.1f}%{100 * summary[name]['test_seen']:>10.1f}%{100 * summary[name]['test_unseen']:>12.1f}%"
              f"{100 * n_t / len(test_ids):>10.1f}%{100 * n_a / len(ids):>9.1f}%   {st.mean(ks) if ks else float('nan'):.3f}")
    print("\nFeatureNet by training size (mean over seeds, validation / test_seen / test_unseen rank-1 %, no guard):")
    for size in sizes:
        rs = [x for x in results["runs"] if x["size"] == size]
        m = lambda key, s: 100 * st.mean(x[key]["rank1_rate"] for x in rs)
        print(f"   size {size:>5}: train {m('train', 0):5.1f} | val {m('val', 0):5.1f} | test_seen {m('test_seen', 0):5.1f} | test_unseen {m('test_unseen', 0):5.1f}"
              f"   (with guard: val {m('guard_val', 0):5.1f} | test_seen {m('guard_test_seen', 0):5.1f} | test_unseen {m('guard_test_unseen', 0):5.1f})")
    print("\nPaired comparisons (exact McNemar on rank 1):")
    pairs = [("FeatureNet", "v1 learned pair"), ("FeatureNet", "reference 0.6/0.5"), ("FeatureNet", "v1 PromptNet"),
             ("FeatureNet + guard", "v1 learned pair"), ("FeatureNet + guard", "FeatureNet")]
    if "free per-feature" in per_prompt:
        pairs += [("free per-feature", "FeatureNet")]
    for title, sub in (("300 TEST prompts", test_ids), ("all 450 held-out prompts", ids)):
        print(f"  on the {title}:")
        for x, y in pairs:
            b, cc, both, p = mcnemar(per_prompt, x, y, sub)
            print(f"     {x:<20} vs {y:<20}: {x} only {b:>3} | {y} only {cc:>3} | both {both:>3} | p = {p:.4f}")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({**results, "summary": summary, "selected": {"size": row["size"], "seed": row["seed"]},
                   "per_prompt_rank": {k: {str(c): r for c, r in v.items()} for k, v in per_prompt.items()}}, f, default=float)
    print(f"\nWrote {out_dir}\\results.json, feature_multipliers.json, featurenet_best.pt   (total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
