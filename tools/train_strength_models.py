"""
Train and evaluate the learned-strength models (docs/Research_Journal/23.md, sections 6 to 8).

Needs (all built earlier): outputs/strength_cache (1,450 files), outputs/strength_tables (1,450 files),
data/counterfact_split.json.

What it does
  1. self-checks: (a) the reduced forward pass equals the full model; (b) the table-based edit equals your real hook;
     (c) the saved state reproduces the baseline; (d) the allowed features at 0.6 / 0.5 equal the REAL SWEEP's round-0
     safe pools (ids and order) on a few prompts. It stops if a check fails badly.
  2. reference: fixed 0.6 / 0.5 under the same prefix proxy.
  3. for each training size (learning curve) and seed: train ConstantPair (the control) and PromptNet, early stopping on the
     validation loss; evaluate on the training prompts used, validation, test-seen and test-unseen.
  4. oracle: strengths optimised on each validation / test prompt by gradient (upper bound).
  5. writes outputs/strength_models/<time>/results.json, strengths_export.json (per-prompt strengths for the real sweeps)
     and the selected PromptNet checkpoint, and prints a summary table.

THE PROXY (see src/strength_models.py): growing prefixes of the allowed features (1, 2, 3, 5, 8, 13, 21, 34, 55, all), like
the sweep's cumulative steps; a prompt succeeds if any prefix reaches rank 1; side effects are those of the first successful
prefix. The real sweep also refills its pools; numbers here guide model choice only, the real sweep on the test prompts decides.

    .venv\\Scripts\\python.exe tools/train_strength_models.py --quick        (a few minutes: smoke test)
    .venv\\Scripts\\python.exe tools/train_strength_models.py                (full: sizes 100 250 500 1000, 3 seeds; about an hour on the GTX 1660 Ti, estimate)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device  # noqa: E402
from src import strength_models as sm  # noqa: E402
from src.hooks import make_scale_map_hook  # noqa: E402


def check_forward_and_edit(model, sae, items, W_dec, dev, a):
    """(a) reduced forward == full forward, (b) table edit == real hook, (c) baseline reproduced."""
    bt = sm.make_batch(items[:8], W_dec, dev)
    mu = torch.full((bt["B"],), sm.REFERENCE[0], device=dev)
    beta = torch.full((bt["B"],), sm.REFERENCE[1], device=dev)
    with torch.no_grad():
        delta, pm, pb = sm.edit_delta(bt, mu, beta)
        fast = sm.forward_last(model, bt, delta)
        full = model(bt["resid"] + delta, start_at_layer=8, tokens=bt["toks"])[torch.arange(bt["B"], device=dev), bt["lens"] - 1]
        d_a = float((fast - full).abs().max())
        clean = sm.clean_last_logits(model, bt)
        d_c = max(abs(float(torch.softmax(clean[i], -1)[bt["tgt"][i]]) - items[i]["baseline_prob"]) for i in range(bt["B"]))
        d_b, same_rank = 0.0, 0
        for i in range(min(5, bt["B"])):
            it = items[i]
            scale = {}
            for k, fid in enumerate(it["ids"].tolist()):
                if bool(pm[i, k]):
                    scale[fid] = 1.0 - sm.REFERENCE[0]
                elif bool(pb[i, k]):
                    scale[fid] = 1.0 + sm.REFERENCE[1]
            model.reset_hooks()
            model.add_hook("blocks.8.hook_resid_pre", make_scale_map_hook(scale, sae))
            real = model(model.to_tokens(it["prompt"]))[0, -1]
            model.reset_hooks()
            d_b = max(d_b, float((real - fast[i]).abs().max()))
            rk = lambda l: int((torch.softmax(l, -1) > torch.softmax(l, -1)[it["target_id"]]).sum()) + 1
            same_rank += int(rk(real) == rk(fast[i]))
    print(f"Check (a) reduced forward vs full forward: max logit diff {d_a:.1e}")
    print(f"Check (b) table-based edit vs your real hook (0.6/0.5, 5 prompts): max logit diff {d_b:.1e}, same target rank {same_rank}/5")
    print(f"Check (c) baseline probability from the saved state vs the cache: max diff {d_c:.1e}")
    if d_a > 1e-3 or d_b > 5e-3 or d_c > 1e-3 or same_rank < 5:
        sys.exit("A forward/edit check failed; stopping.")


def survivor_ids(item, W_dec, dev):
    """Allowed features at (0.6, 0.5) in the sweep's order, from the strength tables."""
    bt = sm.make_batch([item], W_dec, dev)
    mu = torch.tensor([sm.REFERENCE[0]], device=dev)
    beta = torch.tensor([sm.REFERENCE[1]], device=dev)
    pm, pb = sm.survivors(bt, mu, beta)
    ids = item["ids"].tolist()
    mute = [ids[k] for k in bt["m_order"][0].tolist() if k < len(ids) and bool(pm[0, k])]
    boost = [ids[k] for k in bt["b_order"][0].tolist() if k < len(ids) and bool(pb[0, k])]
    return mute, boost


def check_against_sweep(model, sae, items, W_dec, dev, n):
    """Compare the table-derived round-0 safe pools at 0.6/0.5 with the REAL sweep's (ids and order)."""
    from src.hybrid_runner import run_hybrid_sweep
    cfg = {"mute_strength": 0.6, "boost_strength": 0.5, "top_n": 200, "safety_mode": "strict", "max_steps": 1,
           "collateral": False, "combination_check": False, "record_detail": "compact", "stop_on_rank1": True}
    exact, tot_diff, worst = 0, 0, 0.0
    for it in items[:n]:
        rec = run_hybrid_sweep(model, sae, "blocks.8.hook_resid_pre", 8, str(next(model.parameters()).device), it["prompt"],
                               it["target"], cfg, "all")
        r0 = rec["rounds"][0]
        real_m, real_b = r0["safe_mute_ids"], r0["safe_boost_ids"]
        my_m, my_b = survivor_ids(it, W_dec, dev)
        ids = it["ids"].tolist()
        pos = {f: k for k, f in enumerate(ids)}
        eff = lambda f: max(abs(float(it["md"][2, pos[f]])), abs(float(it["bd"][1, pos[f]])))     # effect at mu 0.6 / beta 0.5
        sym = (set(real_m) ^ set(my_m)) | (set(real_b) ^ set(my_b))
        diff = len(set(real_m) ^ set(my_m)) + len(set(real_b) ^ set(my_b))
        gap = max([eff(f) for f in sym] or [0.0])                         # size of the effect of features that differ
        for real, mine, vals in ((real_m, my_m, it["db"]), (real_b, my_b, it["dt"])):
            cr, cm = [f for f in real if f in set(mine)], [f for f in mine if f in set(real)]
            gap = max([gap] + [abs(float(vals[pos[x]]) - float(vals[pos[y]])) for x, y in zip(cr, cm) if x != y])   # order swaps between near-ties
        worst = max(worst, gap)
        exact += int(diff == 0 and gap == 0.0)
        tot_diff += diff
        print(f"   prompt {it['case_id']}: sweep mute/boost {len(real_m)}/{len(real_b)} | tables {len(my_m)}/{len(my_b)} | "
              f"features that differ: {diff} | largest effect among differing or swapped features: {gap:.1e}")
    print(f"Check (d) allowed features vs the real sweep's round 0: exactly identical on {exact}/{n} prompts; {tot_diff} differing features in total; "
          f"every difference involves effects of at most {worst:.1e} (numerical noise between machines)")
    if worst > 1e-4:
        sys.exit("Table-derived pools differ from the sweep's round 0 by more than numerical noise; stopping.")


def fmt(s):
    return (f"{100 * s['rank1_rate']:5.1f}% | KL(succ) {s['mean_kl_of_successes']:.3f} | mu {s['mean_mu']:.2f}+-{s['std_mu']:.2f} | "
            f"beta {s['mean_beta']:.2f}+-{s['std_beta']:.2f} | feats {s['mean_n_mute']:.0f}m/{s['mean_n_boost']:.0f}b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs", "strength_cache"))
    ap.add_argument("--tables-dir", default=os.path.join(ROOT, "outputs", "strength_tables"))
    ap.add_argument("--split-file", default=os.path.join(ROOT, "data", "counterfact_split.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sizes", default="100,250,500,1000")
    ap.add_argument("--seeds", type=int, default=3, help="random seeds per model and size (the plan said 5; 3 keeps the run to about an hour)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--const-lr", type=float, default=0.05, help="learning rate of the two-number control (needs a larger step than the network)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lam-kl", type=float, default=10.0)
    ap.add_argument("--lam-size", type=float, default=0.02)
    ap.add_argument("--train-prefixes", default="1,3,8,21,0", help="coarser prefix grid used while TRAINING (faster); evaluation uses --prefixes")
    ap.add_argument("--prefixes", default=",".join(str(x) for x in sm.DEFAULT_PREFIXES),
                    help="prefix sizes tried per prompt (first n allowed mute and boost features, like the sweep's cumulative steps); 0 = all")
    ap.add_argument("--oracle-steps", type=int, default=100)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--sweep-check", type=int, default=5, help="prompts to compare with the real sweep's round 0 (0 = skip)")
    ap.add_argument("--quick", action="store_true", help="smoke test: sizes 100,250; 1 seed; 10 epochs; 30 oracle steps")
    a = ap.parse_args()
    if a.quick:
        a.sizes, a.seeds, a.epochs, a.oracle_steps = "100,250", 1, 10, 30
    sizes = [int(x) for x in a.sizes.split(",")]
    a.prefix_list = [int(x) for x in a.prefixes.split(",")]
    a.train_prefix_list = [int(x) for x in a.train_prefixes.split(",")]
    out_dir = a.out_dir or os.path.join(ROOT, "outputs", "strength_models", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    dev = get_default_device()
    model = load_base_model(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sae = load_sae_for_layer(layer=8)
    W_dec = sae.W_dec.detach()
    print(f"Device {dev} | torch {torch.__version__} | sizes {sizes} | seeds {a.seeds} | out {out_dir}", flush=True)

    with open(a.split_file, encoding="utf-8") as f:
        split = json.load(f)
    t0 = time.time()
    load = lambda name, ids: [sm.load_item(i, name, a.cache_dir, a.tables_dir) for i in ids]
    train_all = load("train", split["train"][:max(sizes)])
    sets = {"val": load("val", split["val"]), "test_seen": load("test_seen", split["test_seen"]),
            "test_unseen": load("test_unseen", split["test_unseen"])}
    print(f"Loaded {len(train_all)} training + {sum(len(v) for v in sets.values())} evaluation prompts in {time.time() - t0:.0f} s", flush=True)

    check_forward_and_edit(model, sae, sets["val"], W_dec, dev, a)
    if a.sweep_check:
        check_against_sweep(model, sae, sets["val"], W_dec, dev, a.sweep_check)

    results = {"args": vars(a), "reference": {}, "oracle": {}, "runs": []}
    ref_fn = lambda bt: (torch.full((bt["B"],), sm.REFERENCE[0], device=dev), torch.full((bt["B"],), sm.REFERENCE[1], device=dev))
    print("\nReference (fixed 0.6 / 0.5, same prefix proxy):", flush=True)
    for name, items in {**sets, "train(first %d)" % max(sizes): train_all}.items():
        s = sm.summarise(sm.evaluate(model, ref_fn, items, a, W_dec, dev))
        results["reference"][name] = s
        print(f"   {name:<16} {fmt(s)}", flush=True)

    exports = {"reference": {"mu": sm.REFERENCE[0], "beta": sm.REFERENCE[1]}, "constant": {}, "promptnet": {}, "oracle": {}}
    selected = {}
    for size in sizes:
        train_items = train_all[:size]
        for kind in ("constant", "promptnet"):
            for seed in range(a.seeds):
                t1 = time.time()
                print(f"\n[{kind}] size {size} seed {seed}", flush=True)
                net, hist = sm.train_one(model, kind, train_items, sets["val"], a, W_dec, dev, seed, log=lambda m: print(m, flush=True))
                fn = lambda bt, net=net: net(bt["feat"])
                row = {"kind": kind, "size": size, "seed": seed, "epochs": len(hist), "seconds": round(time.time() - t1, 1), "history": hist}
                evals = {}
                for name, items in {"train": train_items, **sets}.items():
                    evals[name] = sm.evaluate(model, fn, items, a, W_dec, dev)
                    row[name] = sm.summarise(evals[name])
                if kind == "constant":
                    row["pair"] = {"mu": row["val"]["mean_mu"], "beta": row["val"]["mean_beta"]}
                results["runs"].append(row)
                print(f"   done in {row['seconds']} s ({row['epochs']} epochs) | val {fmt(row['val'])}", flush=True)
                key = (kind, size)
                if key not in selected or row["val"]["rank1_rate"] > selected[key][0]["val"]["rank1_rate"]:
                    selected[key] = (row, net, evals)
        with open(os.path.join(out_dir, "results.json"), "w") as f:       # save after every size
            json.dump(results, f, default=float)

    if not a.skip_oracle:
        print("\nOracle (strengths optimised on each evaluation prompt):", flush=True)
        orc = {}
        for name, items in sets.items():
            t1 = time.time()
            orc[name] = sm.oracle(model, items, a, W_dec, dev, steps=a.oracle_steps)
            results["oracle"][name] = sm.summarise(orc[name])
            print(f"   {name:<16} {fmt(results['oracle'][name])}   ({time.time() - t1:.0f} s)", flush=True)
            for c, mu, be in zip(orc[name]["case_id"], orc[name]["mu"], orc[name]["beta"]):
                exports["oracle"][str(c)] = {"mu": mu, "beta": be}

    # export the selected models (largest size, best validation rank-1 rate) for the real sweeps
    big = max(sizes)
    row_c, net_c, ev_c = selected[("constant", big)]
    row_p, net_p, ev_p = selected[("promptnet", big)]
    exports["constant"] = row_c["pair"]
    for name in sets:
        for c, mu, be in zip(ev_p[name]["case_id"], ev_p[name]["mu"], ev_p[name]["beta"]):
            exports["promptnet"][str(c)] = {"mu": mu, "beta": be}
    torch.save({"state_dict": net_p.state_dict(), "args": vars(a), "size": big, "seed": row_p["seed"]}, os.path.join(out_dir, "promptnet_best.pt"))
    results["selected"] = {"constant_seed": row_c["seed"], "promptnet_seed": row_p["seed"], "size": big}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, default=float)
    with open(os.path.join(out_dir, "strengths_export.json"), "w") as f:
        json.dump(exports, f)

    # ---- summary ----
    print("\n==================== SUMMARY (prefix proxy; rank-1 rate, mean over seeds) ====================")
    print(f"{'model':<18}{'size':>6}   {'train':>7} {'val':>7} {'test_seen':>10} {'test_unseen':>12}   {'KL(succ)':>8}  mu / beta (val, mean)")
    for name in ("val", "test_seen", "test_unseen"):
        pass
    r = results["reference"]
    print(f"{'reference 0.6/0.5':<18}{'-':>6}   {100 * r['train(first %d)' % max(sizes)]['rank1_rate']:>6.1f}% {100 * r['val']['rank1_rate']:>6.1f}% "
          f"{100 * r['test_seen']['rank1_rate']:>9.1f}% {100 * r['test_unseen']['rank1_rate']:>11.1f}%   {r['val']['mean_kl_of_successes']:>8.3f}  0.60 / 0.50")
    for kind in ("constant", "promptnet"):
        for size in sizes:
            rows = [x for x in results["runs"] if x["kind"] == kind and x["size"] == size]
            mean = lambda f: sum(f(x) for x in rows) / len(rows)
            print(f"{kind:<18}{size:>6}   {100 * mean(lambda x: x['train']['rank1_rate']):>6.1f}% {100 * mean(lambda x: x['val']['rank1_rate']):>6.1f}% "
                  f"{100 * mean(lambda x: x['test_seen']['rank1_rate']):>9.1f}% {100 * mean(lambda x: x['test_unseen']['rank1_rate']):>11.1f}%   "
                  f"{mean(lambda x: x['val']['mean_kl_of_successes']):>8.3f}  {mean(lambda x: x['val']['mean_mu']):.2f} / {mean(lambda x: x['val']['mean_beta']):.2f}"
                  f"   (std of mu across prompts {mean(lambda x: x['val']['std_mu']):.3f}, beta {mean(lambda x: x['val']['std_beta']):.3f})")
    if results["oracle"]:
        o = results["oracle"]
        print(f"{'oracle (per prompt)':<18}{'-':>6}   {'-':>7} {100 * o['val']['rank1_rate']:>6.1f}% {100 * o['test_seen']['rank1_rate']:>9.1f}% "
              f"{100 * o['test_unseen']['rank1_rate']:>11.1f}%   {o['val']['mean_kl_of_successes']:>8.3f}")
    print("\nstd of mu / beta across prompts near 0 means the network collapsed to a constant (then it equals the control).")
    print(f"\nWrote {out_dir}\\results.json, strengths_export.json, promptnet_best.pt   (total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
