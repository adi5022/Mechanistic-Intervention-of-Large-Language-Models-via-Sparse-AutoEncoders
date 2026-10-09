"""
Step D PILOT (docs/cross_model_transfer/PLAN.md, phase D): export a real gradient-descent edit from GPT-2 small to GPT-2 medium.
A small look, not a result: about 30 CounterFact records, no dev/test split, no gate. Its job is to show what the edit looks like and
whether the translated change moves the other model at all, so that the real step D can be designed on facts.

    .venv\\Scripts\\python.exe tools/transfer/04_export_pilot.py --n 6        # smoke test
    .venv\\Scripts\\python.exe tools/transfer/04_export_pilot.py              # 30 records

For each record ("The mother tongue of Danielle Darrieux is" -> counterfactual target " English"):
  1. In GPT-2 small, run the project's gradient-descent edit (src/gradient_editing.py, unchanged: 200 candidate features, 100 steps) so that
     the counterfactual target becomes the top answer.
  2. Read the change it makes to small's layer-8 state at every position:  delta = sum_k a_k * act_k(position) * W_dec[k].
  3. Translate it with the A1 map (small layer 8 -> medium layer 12 or 16), multiply by a dose and add it to medium while it reads the same prompt.
  4. Look at where the target word ends up in medium's ranking.
Comparisons (all the same size at each position): a random vector; the translated edit of ANOTHER record (aligned at the end of the prompt);
a push along the target word's own output direction in medium (the "just say the word" control).
Records are taken from CounterFact with single-token targets, excluding the 100 ROME dev records (random.Random(0).sample) and any record where
either model already answers with the target. Writes outputs/transfer/d_pilot.json.
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.editing import build_clean_context, get_target_token_id, tokens_without_bos
from src.gradient_editing import run_gradient_descent_edit
from src.sae_utils import load_sae_for_layer
from src.transfer.models import load_model, pick_device
from src.transfer.swap import run_injected

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
HOOK = "blocks.8.hook_resid_pre"
ARMS = ["translated", "random", "wrong_record", "word_push"]


def rank_of(logits, tid):
    return int((logits > logits[tid]).sum().item()) + 1


def last_logits_of(model, tokens):
    with torch.no_grad():
        return model(tokens)[0, -1].float()


def pick_records(small, medium, n, dev):
    ds = json.load(open(os.path.join(ROOT, "datasets", "counterfact.json"), encoding="utf-8"))
    rome_dev = set(random.Random(0).sample(range(len(ds)), 100))
    order = random.Random(1).sample(range(len(ds)), len(ds))
    picked = []
    for i in order:
        if i in rome_dev:
            continue
        rw = ds[i]["requested_rewrite"]
        tgt = rw["target_new"]["str"]
        if tokens_without_bos(small, " " + tgt).numel() != 1:
            continue
        prompt = rw["prompt"].format(rw["subject"])
        tid = get_target_token_id(small, " " + tgt)
        toks = small.to_tokens(prompt)
        if last_logits_of(small, toks).argmax().item() == tid or last_logits_of(medium, toks).argmax().item() == tid:
            continue
        picked.append({"index": i, "case_id": ds[i]["case_id"], "prompt": prompt, "subject": rw["subject"], "target": tgt, "tid": tid})
        if len(picked) >= n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--keys", default="s2m_L8_L12,s2m_L8_L16")
    ap.add_argument("--alphas", default="1,2,3,4")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    keys = a.keys.split(",")
    dev = pick_device(a.device)
    t0 = time.time()
    maps = torch.load(os.path.join(OUT_DIR, "maps_gpt2_to_gpt2-medium.pt"))["maps"]
    small, medium = load_model("gpt2", dev), load_model("gpt2-medium", dev)
    sae = load_sae_for_layer(8, dev)
    recs = pick_records(small, medium, a.n, dev)
    print(f"{len(recs)} records; maps {keys}; doses {alphas}", flush=True)

    # ---- step 1 and 2: the edit in small, and the change it makes -------------------------------------------------------------
    edits = []
    for r in recs:
        ctx = build_clean_context(small, sae, r["prompt"], r["tid"])
        out = run_gradient_descent_edit(small, sae, ctx, r["tid"], r["prompt"], 8, HOOK, top_n=200, steps=a.steps)
        fids = out["fids"]
        with torch.no_grad():
            acts = sae.encode(ctx.resid_all[0])[:, fids]
            delta = torch.einsum("pk,k,kd->pd", acts, torch.tensor(out["a"], device=dev), sae.W_dec[fids].detach())      # [seq, 768]
        norms = delta.norm(dim=-1)
        resid_norm = ctx.resid_all[0].norm(dim=-1)
        edits.append({"rec": r, "delta": delta, "tokens": ctx.tokens, "small_start_rank": ctx.clean_rank, "small_edit_rank": out["real_path"]["rank"],
                      "kl": out["real_path"]["kl"], "edit_size_last": out["edit_size_frac_norm"],
                      "share_last": float(norms[-1] ** 2 / (norms ** 2).sum().clamp(min=1e-12)),
                      "share_other_than_bos_last": float((norms[1:-1] ** 2).sum() / (norms ** 2).sum().clamp(min=1e-12)),
                      "peak_position_from_end": int(len(norms) - 1 - norms.argmax().item()),
                      "total_rel_size": float((norms[1:] ** 2).sum().sqrt() / (resid_norm[1:] ** 2).sum().sqrt())})
        print(f"   edit {len(edits):>2}/{len(recs)} {r['prompt']!r} -> {r['target']!r}: small rank {ctx.clean_rank} -> {out['real_path']['rank']}, KL {out['real_path']['kl']:.2f}, "
              f"share of the change at the last position {edits[-1]['share_last']:.0%}", flush=True)

    # ---- steps 3 and 4: send it to medium -----------------------------------------------------------------------------------------
    results = {}
    g = torch.Generator().manual_seed(0)
    for key in keys:
        q = int(key.split("_L")[2])
        W = maps[key]["W"].to(dev)
        res = {arm: {al: [] for al in alphas} for arm in ARMS}
        base_ranks = []
        for i, e in enumerate(edits):
            r, tok = e["rec"], e["tokens"]
            Dt = e["delta"] @ W                                                          # [seq, dR]
            L = Dt.shape[0]
            lg0 = last_logits_of(medium, tok)
            base_ranks.append(rank_of(lg0, r["tid"]))
            other = edits[(i + 7) % len(edits)]["delta"] @ W                             # another record's change, aligned at the end of the prompt
            wrong = torch.zeros_like(Dt)
            m = min(L, other.shape[0])
            wrong[-m:] = other[-m:]
            rnd = torch.randn(Dt.shape, generator=g).to(dev)
            rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp(min=1e-9) * Dt.norm(dim=-1, keepdim=True)
            push = torch.zeros_like(Dt)
            direction = medium.W_U[:, r["tid"]]
            push[-1] = direction / direction.norm() * Dt[-1].norm()
            arms = {"translated": Dt, "random": rnd, "wrong_record": wrong, "word_push": push}
            for arm, D in arms.items():
                for al in alphas:
                    lg = run_injected(medium, tok, (al * D)[None], q)[0]
                    res[arm][al].append((rank_of(lg, r["tid"]), int(lg.argmax().item() == r["tid"])))
        med = lambda v: sorted(v)[len(v) // 2]
        results[key] = {"medium_baseline_median_rank": med(base_ranks),
                        "arms": {arm: {str(al): {"top1": sum(t for _, t in res[arm][al]), "n": len(edits), "median_rank": med([rk for rk, _ in res[arm][al]])} for al in alphas} for arm in ARMS}}

    ok = [e for e in edits if e["small_edit_rank"] == 1]
    summary = {"n_records": len(edits), "small_edit_reaches_rank1": len(ok),
               "small_start_median_rank": sorted(e["small_start_rank"] for e in edits)[len(edits) // 2],
               "median_kl": sorted(e["kl"] for e in edits)[len(edits) // 2],
               "median_edit_size_last_position": sorted(e["edit_size_last"] for e in edits)[len(edits) // 2],
               "mean_share_of_change_at_last_position": sum(e["share_last"] for e in edits) / len(edits),
               "mean_share_of_change_at_positions_between": sum(e["share_other_than_bos_last"] for e in edits) / len(edits)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "d_pilot.json"), "w", encoding="utf8") as f:
        json.dump({"summary": summary, "maps": results, "records": [{k: v for k, v in e.items() if k in ("small_start_rank", "small_edit_rank", "kl", "edit_size_last", "share_last", "peak_position_from_end")} | {"prompt": e["rec"]["prompt"], "target": e["rec"]["target"], "case_id": e["rec"]["case_id"]} for e in edits]}, f, indent=1)

    print("\n" + "=" * 110)
    print("RESULT  step D pilot (a look, not a result: about 30 records, no split, no gate)")
    print(f"  records {summary['n_records']}; the gradient-descent edit reaches rank 1 in small on {summary['small_edit_reaches_rank1']}; small's start rank of the target (median) {summary['small_start_median_rank']}; "
          f"same-prompt KL (median) {summary['median_kl']:.2f}; edit size at the last position (median) {summary['median_edit_size_last_position']:.0%} of the residual norm")
    print(f"  where the edit's change sits (share of its squared size): last position {summary['mean_share_of_change_at_last_position']:.0%}, positions between the start token and the last {summary['mean_share_of_change_at_positions_between']:.0%}")
    for key, v in results.items():
        print(f"\n  {key}: medium's own median rank of the target before anything = {v['medium_baseline_median_rank']}")
        print(f"  {'arm':<14}" + "".join(f"{'dose ' + str(al):>22}" for al in alphas) + "     (cells: target is medium's top answer / median rank of the target)")
        for arm in ARMS:
            print(f"  {arm:<14}" + "".join(f"{str(v['arms'][arm][str(al)]['top1']) + '/' + str(v['arms'][arm][str(al)]['n']) + '  rank ' + str(v['arms'][arm][str(al)]['median_rank']):>22}" for al in alphas))
    print(f"\n  total time {time.time() - t0:.0f} s")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
