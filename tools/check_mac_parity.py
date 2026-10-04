"""
Does THIS machine / device reproduce the results saved from the Windows + NVIDIA runs?  Run it first on a new machine (for example the Mac).

    python tools/check_mac_parity.py                  # auto device (cuda > mps > cpu)
    python tools/check_mac_parity.py --device cpu     # force a device
    python tools/check_mac_parity.py --full           # also runs tools/check_additive.py (5 self-checks, a few minutes)

What it checks (references are the committed Entry 29 records in docs/Research_Journal/packs/sweep_vs_gradient/runs.jsonl):
 1. BASELINE   the unedited model's rank of the true target on 16 prompts, against the saved start ranks
 2. GRADIENT   the gradient-descent edit on 4 prompts: does it still reach rank 1, and is the side-effect KL close to the saved value
 3. DEVICE     the edit applied through the real hook gives the same logits as the shortcut used while tuning (device-internal consistency)

Floating-point arithmetic differs a little between NVIDIA GPUs, Apple GPUs and CPUs. Small differences in probabilities are expected and
are NOT failures; a rank can differ by a place when two words are nearly tied. Exit code 0 = acceptable, 1 = a real mismatch.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from src.device_utils import describe_device, sync_device
from src.sae_utils import get_default_device, load_base_model, load_sae_for_layer
from src.editing import build_clean_context, get_target_token_id
from src.gradient_editing import run_gradient_descent_edit, _last_logits
from src.hooks import make_scale_and_add_hook

PACK = os.path.join(ROOT, "docs", "Research_Journal", "packs", "sweep_vs_gradient", "runs.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--full", action="store_true", help="also run tools/check_additive.py")
    a = ap.parse_args()
    device = a.device or get_default_device()
    print(f"device: {describe_device(device)}   torch {torch.__version__}")

    model = load_base_model(device=device)
    layer = 8
    sae = load_sae_for_layer(layer=layer, device=device)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")
    rows = [json.loads(l) for l in open(PACK, encoding="utf-8") if l.strip()]
    ok = True

    # 1. baseline ranks, 4 prompts from each difficulty band
    pick = []
    for band in ("2-5", "6-20", "21-100", "101-1000"):
        pick += [r for r in rows if r["band"] == band][:4]
    print(f"\n1. BASELINE: unedited rank of the true target on {len(pick)} prompts vs the saved start rank")
    exact, worst_gap, bad = 0, 0, []
    for r in pick:
        tid = get_target_token_id(model, " " + r["target"])
        ctx = build_clean_context(model, sae, r["prompt"], tid)
        gap = abs(ctx.clean_rank - r["start_rank"])
        worst_gap = max(worst_gap, gap)
        exact += gap == 0
        if gap:
            print(f"   differs: {r['prompt'][:50]!r} rank {ctx.clean_rank} vs saved {r['start_rank']}")
        if gap > max(1, int(0.02 * r["start_rank"])):          # a place near ties, or 2% for deep ranks
            bad.append(r)
    print(f"   {exact} of {len(pick)} identical; largest rank difference {worst_gap}")
    ok &= not bad
    print("   " + ("OK" if not bad else f"MISMATCH on {len(bad)} prompt(s) beyond the tolerance (a place, or 2%)"))

    # 2. gradient descent on 4 prompts
    print("\n2. GRADIENT: the multiplier edit on 4 prompts (one per band) vs the saved Entry 29 result")
    small = []
    for band in ("2-5", "6-20", "21-100", "101-1000"):
        small += [r for r in rows if r["band"] == band and r["gd"]["rank"] == 1][:1]
    for r in small:
        tid = get_target_token_id(model, " " + r["target"])
        ctx = build_clean_context(model, sae, r["prompt"], tid)
        sync_device(device)
        t0 = time.perf_counter()
        g = run_gradient_descent_edit(model, sae, ctx, tid, r["prompt"], layer, hook_name, top_n=200, positions="all", steps=100)
        sync_device(device)
        dt = time.perf_counter() - t0
        rk, kl = g["real_path"]["rank"], g["real_path"]["kl"]
        good = rk == 1 and abs(kl - r["gd"]["kl"]) < 0.05
        ok &= good
        print(f"   {r['prompt'][:42]!r:<46} rank {r['start_rank']} -> {rk} (saved {r['gd']['rank']}) | KL {kl:.4f} (saved {r['gd']['kl']:.4f}) | {dt:.1f}s {'ok' if good else 'MISMATCH'}")

    # 3. real hook == shortcut, on this device
    print("\n3. DEVICE: tuned edit through the real hook equals the shortcut used while tuning")
    r = small[0]
    tid = get_target_token_id(model, " " + r["target"])
    ctx = build_clean_context(model, sae, r["prompt"], tid)
    g = run_gradient_descent_edit(model, sae, ctx, tid, r["prompt"], layer, hook_name, top_n=200, positions="all", steps=30)
    same = g["train_path"]["rank"] == g["real_path"]["rank"]
    ok &= same
    print(f"   tuning-path rank {g['train_path']['rank']} vs real-hook rank {g['real_path']['rank']}: {'ok' if same else 'DIFFERENT'}")

    if a.full:
        print("\nrunning tools/check_additive.py ...")
        env = {**os.environ, "FEATURESCALPEL_DEVICE": device}
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_additive.py")], env=env, cwd=ROOT).returncode
        ok &= rc == 0

    print("\n" + ("PARITY OK: this device reproduces the saved results within tolerance." if ok else
                  "PARITY PROBLEM: see the lines above. If the device is mps, try FEATURESCALPEL_DEVICE=cpu and tell the maintainer."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
