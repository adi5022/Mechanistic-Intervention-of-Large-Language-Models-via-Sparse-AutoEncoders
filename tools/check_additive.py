"""
Self-checks for the additive edit (docs/Research_Journal/30.md). Run before trusting any additive result:

 1. REGRESSION   with the new options off, the gradient-descent edit reproduces the Entry 29 `gd` results (rank equal, KL close)
 2. NO-EDIT START at step 0 the additive edit leaves the rank unchanged and the KL near 0
 3. PATH EQUALITY a random multiplier + additive edit through the delta shortcut equals the same edit through the real hook
 4. SILENT ONLY  every added feature has activation exactly 0 at the last position, and the added amounts stay in [0, cap]

    .venv\\Scripts\\python.exe tools/check_additive.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from src.sae_utils import load_base_model, load_sae_for_layer
from src.editing import build_clean_context, get_target_token_id
from src.gradient_editing import run_gradient_descent_edit, _last_logits, silent_candidates
from src.hooks import make_scale_and_add_hook

PACK = os.path.join(ROOT, "docs", "Research_Journal", "packs", "sweep_vs_gradient", "runs.jsonl")


def main():
    model = load_base_model()
    layer = 8
    sae = load_sae_for_layer(layer=layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
    rows = [json.loads(l) for l in open(PACK, encoding="utf-8") if l.strip()]
    pick = []
    for band in ("2-5", "6-20", "21-100", "101-1000"):
        pick += [r for r in rows if r["band"] == band][:2]
    pick = pick[:6]
    ok_all = True

    def ctx_for(r):
        tid = get_target_token_id(model, " " + r["target"])
        return tid, build_clean_context(model, sae, r["prompt"], tid)

    print("1. REGRESSION (options off vs Entry 29 gd):")
    worst = 0.0
    for r in pick:
        tid, ctx = ctx_for(r)
        g = run_gradient_descent_edit(model, sae, ctx, tid, r["prompt"], layer, hook_name, top_n=200, positions="all", steps=100)
        d = abs(g["real_path"]["kl"] - r["gd"]["kl"])
        worst = max(worst, d)
        same = g["real_path"]["rank"] == r["gd"]["rank"]
        ok_all &= same
        print(f"   {r['prompt'][:40]!r:<44} rank {g['real_path']['rank']} vs {r['gd']['rank']} {'ok' if same else 'DIFFERENT'} | KL {g['real_path']['kl']:.4f} vs {r['gd']['kl']:.4f}")
    print(f"   largest KL difference {worst:.2e}")
    ok_all &= worst < 5e-3

    print("2. NO-EDIT START (additive on, 0 steps):")
    for r in pick[:3]:
        tid, ctx = ctx_for(r)
        g = run_gradient_descent_edit(model, sae, ctx, tid, r["prompt"], layer, hook_name, steps=0, additive=True)
        sm = g["add_summary"]
        good = (g["real_path"]["rank"] == ctx.clean_rank and g["train_path"]["kl"] < 1e-3 and g["real_path"]["kl"] < 1e-3
                and sm["n_ge_1pct_cap"] == 0 and sm["total_over_cap"] < 0.1)
        ok_all &= good
        print(f"   {r['prompt'][:40]!r:<44} rank {ctx.clean_rank} -> {g['real_path']['rank']}, KL {g['real_path']['kl']:.2e}, "
              f"largest start amount {sm['max_over_cap']:.1e} of cap, total {sm['total_over_cap']:.3f} caps {'ok' if good else 'FAIL'}")

    print("3. PATH EQUALITY (random edit, shortcut vs real hook):")
    torch.manual_seed(0)
    r = pick[2]
    tid, ctx = ctx_for(r)
    resid = ctx.resid_all.detach()
    ids, _ = silent_candidates(model, sae, resid, ctx.tokens, layer, tid, 50)
    from src.editing import get_top_active_features
    fids = [f for f, _ in get_top_active_features(model, sae, r["prompt"], top_n=60, clean_ctx=ctx, positions="all")]
    a = torch.rand(len(fids), device=resid.device) * 3 - 1                       # multipliers 0..3 (a in -1..2)
    c = torch.rand(ids.numel(), device=resid.device) * 5                         # amounts 0..5
    with torch.no_grad():
        acts = sae.encode(resid[0])[:, fids]
        delta = torch.einsum("pk,k,kd->pd", acts, a, sae.W_dec[fids]).unsqueeze(0)
        delta[0, -1] += c @ sae.W_dec[ids]
        short = _last_logits(model, resid + delta, ctx.tokens, layer)
    smap = {f: float(1 + x) for f, x in zip(fids, a.tolist())}
    amap = {int(i): float(x) for i, x in zip(ids.tolist(), c.tolist())}
    model.reset_hooks()
    with torch.no_grad():
        hooked = model.run_with_hooks(ctx.tokens, fwd_hooks=[(hook_name, make_scale_and_add_hook(smap, amap, sae))])[0, -1]
    model.reset_hooks()
    diff = float((short - hooked).abs().max().item())
    print(f"   max logit difference {diff:.2e}  {'ok' if diff < 1e-4 else 'FAIL'}")
    ok_all &= diff < 1e-4

    print("4. SILENT ONLY (additive on, 100 steps):")
    for r in pick[3:6]:
        tid, ctx = ctx_for(r)
        g = run_gradient_descent_edit(model, sae, ctx, tid, r["prompt"], layer, hook_name, steps=100, additive=True)
        with torch.no_grad():
            act_last = sae.encode(ctx.resid_all[0, -1:])[0]
        bad = [i for i in g["added"] if float(act_last[i]) != 0.0]
        in_range = all(0.0 <= v <= g["add_cap"] * 1.0001 for v in g["added"].values())
        good = not bad and in_range
        ok_all &= good
        rk = g["real_path"]["rank"]
        sm = g["add_summary"]
        print(f"   {r['prompt'][:40]!r:<44} start {ctx.clean_rank} -> {rk} (tuning path {g['train_path']['rank']}), KL {g['real_path']['kl']:.3f}; "
              f"added >=1% of cap: {sm['n_ge_1pct_cap']}, >=10%: {sm['n_ge_10pct_cap']} of {g['add_candidates']}, total {sm['total_over_cap']:.2f} caps; "
              f"active among added: {len(bad)}, in range: {in_range} {'ok' if good else 'FAIL'}")
        ok_all &= (g["train_path"]["rank"] == rk)

    print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
