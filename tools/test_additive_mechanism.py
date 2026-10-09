"""
Why does a strong additive edit make unrelated prompts continue with the target word?

Two checks on the edit that gradient descent finds for one (prompt, target) pair:

  1. DIRECTION. The additive part of the edit is a single vector added to the layer-8 residual at the last position:
     v = sum_k amount_k * W_dec[k]. If the edit is a "say this word" push and not a fact edit, v should point along that
     token's unembedding column. Reported: cosine(v, W_U[:, target]), the percentile of that cosine among all vocabulary
     tokens, and the rank of the target in v @ W_U (a logit-lens reading; the final LayerNorm is ignored, so it is a crude
     reading, not the model's own). The same numbers are given for the multiplier part of the edit (its change at the last
     position) for comparison. A random-word control uses the same pipeline.

  2. DOSE-RESPONSE. The tuned edit is re-applied, scaled by a factor, to unrelated prompts. For each factor: the share of
     unrelated prompts whose top-1 becomes the target, and the mean KL(clean || edited). Additive arm: the amounts are
     multiplied by the factor. Multiplier arm: each multiplier m becomes max(0, 1 + factor * (m - 1)).

    .venv\\Scripts\\python.exe tools/test_additive_mechanism.py --prompt "The capital of France is" --target " India"
    .venv\\Scripts\\python.exe tools/test_additive_mechanism.py --control " powder"

Exploratory: one prompt, one target and one control word per run. The numbers describe that edit, not the method in general.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.editing import build_clean_context, get_target_token_id
from src.gradient_editing import run_gradient_descent_edit, next_token_shift

NEUTRAL = [
    "This boy is", "The weather today is", "She opened the door and", "My favourite food is",
    "The meeting will start at", "He was born in", "The best way to learn is", "I think that the answer is",
    "The company announced that", "In the morning we", "The old man said", "Yesterday I went to the",
]
FACTORS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0, 50.0, 200.0]


def direction_report(model, vec, tid):
    """Cosine of `vec` with the target's unembedding column, its percentile among all tokens, and the target's rank in vec @ W_U."""
    WU = model.W_U.detach().float()                                  # [d_model, vocab]
    v = vec.detach().float()
    cos = (v @ WU) / (v.norm() * WU.norm(dim=0) + 1e-9)              # [vocab]
    lens = v @ WU
    return {"cosine": float(cos[tid]), "cosine_percentile": float((cos < cos[tid]).float().mean() * 100),
            "lens_rank": int((lens > lens[tid]).sum().item()) + 1, "vec_norm": float(v.norm())}


def edit_vectors(sae, ctx, g):
    """The additive vector and the multiplier change at the prompt's last position, both in the residual space."""
    resid = ctx.resid_all.detach()
    with torch.no_grad():
        fids = g["fids"]
        acts_last = sae.encode(resid[0])[-1, fids]
        a = torch.tensor(g["a"], device=resid.device)
        mult_delta = (a * acts_last) @ sae.W_dec[fids].detach()
        add_vec = None
        if g["added"]:
            ids = torch.as_tensor(list(g["added"].keys()), device=resid.device)
            amts = torch.as_tensor(list(g["added"].values()), device=resid.device, dtype=resid.dtype)
            add_vec = amts @ sae.W_dec[ids].detach().to(resid.dtype)
    return add_vec, mult_delta


def dose_response(model, sae, hook_name, g, tid, arm, neutral):
    target_str = model.to_string([tid])
    rows = []
    for f in FACTORS:
        if arm == "add":
            smap = {k: v for k, v in zip(g["fids"], [1.0 + x for x in g["a"]])}
            amap = {k: v * f for k, v in g["added"].items()}
        else:
            smap = {k: max(0.0, 1.0 + f * x) for k, x in zip(g["fids"], g["a"])}
            amap = {}
        kls, flips, tops = [], 0, []
        for p in neutral:
            r = next_token_shift(model, sae, hook_name, p, smap, amap)
            kls.append(r["kl"])
            flips += int(r["edited_top1"] == target_str)        # string compare: to_tokens(prepend_bos=False) would flip the shared BOS default
            tops.append(r["edited_top1"])
        rows.append({"factor": f, "share_top1_is_target": flips / len(neutral), "mean_kl": sum(kls) / len(kls),
                     "example_top1": tops[:4]})
        print(f"  [{arm}] x{f:<6} target-top1 on {flips}/{len(neutral)} unrelated prompts, mean KL {rows[-1]['mean_kl']:.3f}", flush=True)
    return rows


def run_one(model, sae, hook_name, layer, prompt, target, neutral):
    tid = get_target_token_id(model, target)
    ctx = build_clean_context(model, sae, prompt, tid)
    out = {"prompt": prompt, "target": target, "start_rank": ctx.clean_rank}
    print(f"\n=== {prompt!r} -> {target!r} (start rank {ctx.clean_rank})", flush=True)
    for arm, kw in (("add", dict(additive=True)), ("mult", dict(additive=False))):
        t0 = time.time()
        g = run_gradient_descent_edit(model, sae, ctx, tid, prompt, layer, hook_name, top_n=200, positions="all", steps=100, **kw)
        rp = g["real_path"]
        print(f" [{arm}] tuned in {time.time() - t0:.1f}s: final rank {rp['rank']}, KL {rp['kl']:.3f}, edit size {100 * g['edit_size_frac_norm']:.1f}% of residual norm", flush=True)
        add_vec, mult_delta = edit_vectors(sae, ctx, g)
        d = {"final_rank": rp["rank"], "kl": rp["kl"], "edit_pct": 100 * g["edit_size_frac_norm"],
             "direction_multiplier_part": direction_report(model, mult_delta, tid)}
        if arm == "add" and add_vec is not None:
            d["direction_additive_part"] = direction_report(model, add_vec, tid)
            print(f"   additive vector: cosine with target unembedding {d['direction_additive_part']['cosine']:.3f} "
                  f"(percentile {d['direction_additive_part']['cosine_percentile']:.1f}), lens rank {d['direction_additive_part']['lens_rank']}", flush=True)
        print(f"   multiplier change: cosine {d['direction_multiplier_part']['cosine']:.3f} "
              f"(percentile {d['direction_multiplier_part']['cosine_percentile']:.1f}), lens rank {d['direction_multiplier_part']['lens_rank']}", flush=True)
        d["dose_response"] = dose_response(model, sae, hook_name, g, tid, arm, neutral)
        out[arm] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--target", default=" India")
    ap.add_argument("--control", default=" powder", help="random-word control target (same prompt); '' to skip")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "Research_Journal", "packs", "additive_control",
                                                  "mechanism_check.json"))
    a = ap.parse_args()

    model = load_base_model()
    sae = load_sae_for_layer(layer=a.layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{a.layer}.hook_resid_pre")
    res = {"date": time.strftime("%Y-%m-%d"), "device": get_default_device(), "layer": a.layer,
           "neutral_prompts": NEUTRAL, "factors": FACTORS, "runs": [run_one(model, sae, hook_name, a.layer, a.prompt, a.target, NEUTRAL)]}
    if a.control:
        res["runs"].append(run_one(model, sae, hook_name, a.layer, a.prompt, a.control, NEUTRAL))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
