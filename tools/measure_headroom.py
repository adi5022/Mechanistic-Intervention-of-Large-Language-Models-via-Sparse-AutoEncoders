"""
Headroom measurement for the learned-strength study (docs/Research_Journal/23.md, Phase 2b).

Question: for each prompt, IF we could choose the strengths perfectly, could the target reach rank 1 by scaling SAE
features, and how does that compare with an edit that is not restricted to SAE features?

Uses ONLY the per-prompt cache (outputs/strength_cache/*.pt, built by tools/build_strength_cache.py). Nothing is
re-derived from the full model: each cache file holds GPT-2's layer-8 state at every token and the activations of the
200 candidate features, so any edit can be tried by (1) adding  sum_k (m_k - 1) * act_k * decoder_k  to the saved state
and (2) running ONLY blocks 8 to 11. One try costs about 50 ms and a batch of 32 prompts costs the same as one.

Methods measured (each: how many prompts reach rank 1, final rank, side effect KL)
  sae_reach     per-feature multipliers m_k in [0, 1 + beta_max] optimised freely, rank term only
                (the ceiling for SAE-feature scaling with this 200-candidate pool; bounded by beta_max)
  sae_gentle    same, but with penalties for moving other predictions and for large changes
                (what a sensible edit would do; shows the price of keeping side effects small)
  sae_naive_k   crude fixed proxy: top-k mute candidates x0.4 and top-k boost candidates x1.5, NO safety filter, no
                search. NOT the real sweep; only a rough floor so the other numbers have a reference.
  push_last_f   SAE-free: free residual push at layer 8 on the LAST token only, size = f x its residual norm
  push_all_f    SAE-free: free push on every non-BOS token, total size = f x their residual norm
The SAE-free pushes are optimised by gradient and may be adversarial; they are an upper bound for that push size,
not a usable method.

Run (about 30 minutes for all 1,450 prompts on the GTX 1660 Ti; --limit-per-set 10 for a quick test):
    .venv\\Scripts\\python.exe tools/measure_headroom.py
Output: outputs/headroom/headroom_<time>.json (per prompt) and a printed summary.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device  # noqa: E402

PAD_ID = 50256
BANDS = [(2, 5), (6, 20), (21, 100), (101, 1000)]


def band_of(r):
    return next((f"{lo}-{hi}" for lo, hi in BANDS if lo <= r <= hi), "other")


def load_entries(cache_dir, split, sets, limit_per_set):
    out = []
    for name in sets:
        ids = split[name][:1000] if name == "train" else split[name]
        if limit_per_set:
            ids = ids[:limit_per_set]
        for cid in ids:
            out.append((name, torch.load(os.path.join(cache_dir, f"{cid}.pt"), weights_only=False)))
    return out


def make_batch(items, sae, dev):
    """Pad a list of (set, cache entry) into tensors. Right padding is harmless: GPT-2 is causal and we read each
    prompt's logits at its own last real token."""
    B = len(items)
    P = max(e["tokens"].shape[0] for _, e in items)
    K = max(e["cand_ids"].numel() for _, e in items)
    resid = torch.zeros(B, P, 768)
    toks = torch.full((B, P), PAD_ID, dtype=torch.long)
    acts = torch.zeros(B, P, K)
    wdec = torch.zeros(B, K, 768)
    dt, db, amax = torch.zeros(B, K), torch.zeros(B, K), torch.zeros(B, K)
    lens, tgt, blk, kmask = [], [], [], torch.zeros(B, K, dtype=torch.bool)
    for b, (_, e) in enumerate(items):
        n, k = e["tokens"].shape[0], e["cand_ids"].numel()
        resid[b, :n], toks[b, :n] = e["resid8"], e["tokens"]
        acts[b, :n, :k] = e["cand_acts"]
        wdec[b, :k] = sae.W_dec[e["cand_ids"].to(dev)].detach().cpu()
        dt[b, :k], db[b, :k], amax[b, :k] = e["dt"], e["db"], e["cand_act_max"]
        kmask[b, :k] = True
        lens.append(n); tgt.append(e["target_id"]); blk.append(e["blocker_id"])
    to = lambda x: x.to(dev)
    return {"resid": to(resid), "toks": to(toks), "acts": to(acts), "wdec": to(wdec), "dt": to(dt), "db": to(db),
            "amax": to(amax), "kmask": to(kmask), "lens": torch.tensor(lens, device=dev),
            "tgt": torch.tensor(tgt, device=dev), "blk": torch.tensor(blk, device=dev), "B": B, "P": P, "K": K}


def forward_from_8(model, bt, delta):
    """Logits at each prompt's last real token after adding `delta` to the saved layer-8 state (blocks 8 to 11 only)."""
    logits = model(bt["resid"] + delta, start_at_layer=8, tokens=bt["toks"])
    return logits[torch.arange(bt["B"], device=logits.device), bt["lens"] - 1]


def score(last, clean_last, bt):
    """rank, prob, KL(clean || edited) over all tokens except the target and the blocker (renormalised)."""
    lp = F.log_softmax(last, dim=-1)
    tl = lp[torch.arange(bt["B"]), bt["tgt"]]
    rank = (lp > tl.unsqueeze(1)).sum(dim=1) + 1
    mask = torch.zeros_like(last, dtype=torch.bool)
    ar = torch.arange(bt["B"], device=last.device)
    mask[ar, bt["tgt"]] = True
    mask[ar, bt["blk"]] = True
    lq = F.log_softmax(clean_last.masked_fill(mask, float("-inf")), dim=-1)
    lpe = F.log_softmax(last.masked_fill(mask, float("-inf")), dim=-1)
    term = torch.where(mask, torch.zeros_like(lq), lq.exp() * (lq - lpe))
    margin = tl - last.masked_fill(F.one_hot(bt["tgt"], last.shape[1]).bool(), float("-inf")).max(dim=1).values
    return rank, tl.exp(), term.sum(dim=1), margin


class Best:
    """Per-prompt best iterate: lowest rank, then highest probability."""
    def __init__(self, B, dev):
        self.rank = torch.full((B,), 10 ** 9, device=dev)
        self.prob = torch.zeros(B, device=dev)
        self.kl = torch.zeros(B, device=dev)
        self.push = torch.zeros(B, device=dev)

    def update(self, rank, prob, kl, push):
        better = (rank < self.rank) | ((rank == self.rank) & (prob > self.prob))
        for dst, src in ((self.rank, rank), (self.prob, prob), (self.kl, kl), (self.push, push)):
            dst.copy_(torch.where(better, src, dst))


def rel_push(bt, delta):
    """size of the edit on the last token relative to that token's residual norm"""
    ar = torch.arange(bt["B"], device=delta.device)
    d = delta[ar, bt["lens"] - 1].norm(dim=1)
    r = bt["resid"][ar, bt["lens"] - 1].norm(dim=1)
    return d / r.clamp_min(1e-9)


def sae_delta(bt, m):
    return torch.einsum("bpk,bk,bkd->bpd", bt["acts"], (m - 1.0) * bt["kmask"], bt["wdec"])


def optimise_sae(model, bt, clean_last, steps, beta_max, margin, lam_kl, lam_size, lr=0.1):
    """Free per-feature multipliers m = 1 + a, a in [-1, beta_max], starting at m = 1 (no edit)."""
    dev = bt["resid"].device
    z = torch.full((bt["B"], bt["K"]), -float(torch.log(torch.tensor(beta_max))), device=dev).requires_grad_(True)   # sigmoid(z) = 1/(1+beta_max)  ->  a = 0
    opt = torch.optim.Adam([z], lr=lr)
    best = Best(bt["B"], dev)
    size_w = bt["amax"] / bt["amax"].sum(dim=1, keepdim=True).clamp_min(1e-9)
    for _ in range(steps + 1):
        a = -1.0 + (1.0 + beta_max) * torch.sigmoid(z)
        m = 1.0 + a * bt["kmask"]
        delta = sae_delta(bt, m)
        last = forward_from_8(model, bt, delta)
        rank, prob, kl, mg = score(last, clean_last, bt)
        best.update(rank.detach(), prob.detach(), kl.detach(), rel_push(bt, delta).detach())
        loss = F.relu(margin - mg) + lam_kl * kl + lam_size * (a.abs() * size_w).sum(dim=1)
        z.grad, = torch.autograd.grad(loss.sum(), z)
        opt.step()
    return best


def optimise_push(model, bt, clean_last, frac, positions, steps, margin, lr=0.15):
    """SAE-free: delta = budget * u * mask with ||u|| <= 1; budget = frac x residual norm (last token / non-BOS tokens)."""
    dev = bt["resid"].device
    B, P = bt["B"], bt["P"]
    pos = torch.arange(P, device=dev).unsqueeze(0)
    real = pos < bt["lens"].unsqueeze(1)
    if positions == "last":
        mask = (pos == (bt["lens"] - 1).unsqueeze(1)).float().unsqueeze(-1)
        base = bt["resid"][torch.arange(B, device=dev), bt["lens"] - 1].norm(dim=1)
    else:
        mask = ((pos >= 1) & real).float().unsqueeze(-1)
        base = (bt["resid"] * mask).flatten(1).norm(dim=1)
    budget = (frac * base).view(B, 1, 1)
    u = torch.zeros(B, P, 768, device=dev, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    best = Best(B, dev)
    for _ in range(steps + 1):
        delta = budget * u * mask
        last = forward_from_8(model, bt, delta)
        rank, prob, kl, mg = score(last, clean_last, bt)
        best.update(rank.detach(), prob.detach(), kl.detach(), rel_push(bt, delta).detach())
        u.grad, = torch.autograd.grad(F.relu(margin - mg).sum(), u)
        opt.step()
        with torch.no_grad():
            u.mul_(mask)
            u.mul_(torch.clamp(1.0 / u.flatten(1).norm(dim=1).clamp_min(1e-12), max=1.0).view(B, 1, 1))
    return best


def naive(model, bt, clean_last, k, mute_scale=0.4, boost_scale=1.5):
    dev = bt["resid"].device
    m = torch.ones(bt["B"], bt["K"], device=dev)
    big = 1e9
    for key, scale in (("db", mute_scale), ("dt", boost_scale)):
        v = bt[key].masked_fill(~bt["kmask"], big)
        top = torch.topk(v, min(k, bt["K"]), dim=1, largest=False)
        sel = torch.zeros_like(m, dtype=torch.bool).scatter_(1, top.indices, True) & (v < 0)
        m = torch.where(sel, torch.full_like(m, scale), m)
    delta = sae_delta(bt, m)
    with torch.no_grad():
        last = forward_from_8(model, bt, delta)
        rank, prob, kl, _ = score(last, clean_last, bt)
    best = Best(bt["B"], dev)
    best.update(rank, prob, kl, rel_push(bt, delta))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "outputs", "strength_cache"))
    ap.add_argument("--split-file", default=os.path.join(ROOT, "data", "counterfact_split.json"))
    ap.add_argument("--sets", default="val,test_seen,test_unseen,train")
    ap.add_argument("--limit-per-set", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--beta-max", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lam-kl", type=float, default=20.0)
    ap.add_argument("--lam-size", type=float, default=0.01)
    ap.add_argument("--fractions", default="0.05,0.1,0.2")
    ap.add_argument("--naive-k", default="5,20,50")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "headroom"))
    a = ap.parse_args()
    fracs = [float(x) for x in a.fractions.split(",")]
    ks = [int(x) for x in a.naive_k.split(",")]

    dev = get_default_device()
    model = load_base_model(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sae = load_sae_for_layer(layer=8)
    with open(a.split_file, encoding="utf-8") as f:
        split = json.load(f)
    entries = load_entries(a.cache_dir, split, [s.strip() for s in a.sets.split(",") if s.strip()], a.limit_per_set)
    # batch prompts of similar length together (less padding)
    entries.sort(key=lambda x: x[1]["tokens"].shape[0])
    batches = [entries[i:i + a.batch] for i in range(0, len(entries), a.batch)]
    print(f"{len(entries)} prompts in {len(batches)} batches | device {dev} | steps {a.steps} | beta_max {a.beta_max}", flush=True)

    methods = ["sae_reach", "sae_gentle"] + [f"sae_naive_{k}" for k in ks] + \
              [f"push_last_{f}" for f in fracs] + [f"push_all_{f}" for f in fracs]
    records, t0 = [], time.time()
    for bi, items in enumerate(batches):
        bt = make_batch(items, sae, dev)
        with torch.no_grad():
            clean_last = forward_from_8(model, bt, torch.zeros_like(bt["resid"]))
        # correctness checks on the first batch: the saved state reproduces the cached baseline, and padding is harmless
        if bi == 0:
            rank0, prob0, _, _ = score(clean_last, clean_last, bt)
            err = max(abs(float(prob0[i]) - items[i][1]["baseline_prob"]) for i in range(bt["B"]))
            with torch.no_grad():
                e0 = items[0][1]
                single = model(e0["resid8"].unsqueeze(0).to(dev), start_at_layer=8, tokens=e0["tokens"].unsqueeze(0).to(dev))[0, -1]
                pad_err = float((single - clean_last[0]).abs().max())
            print(f"Check: baseline probability from the saved layer-8 state vs the cache: max diff {err:.1e}; "
                  f"padded batch vs single prompt logits: max diff {pad_err:.1e}", flush=True)
            if err > 1e-3 or pad_err > 1e-2:
                sys.exit("Saved-state forward does not reproduce the baseline; stopping.")
        res = {"sae_reach": optimise_sae(model, bt, clean_last, a.steps, a.beta_max, a.margin, 0.0, 1e-4),
               "sae_gentle": optimise_sae(model, bt, clean_last, a.steps, a.beta_max, a.margin, a.lam_kl, a.lam_size)}
        for k in ks:
            res[f"sae_naive_{k}"] = naive(model, bt, clean_last, k)
        for f in fracs:
            res[f"push_last_{f}"] = optimise_push(model, bt, clean_last, f, "last", a.steps, a.margin)
            res[f"push_all_{f}"] = optimise_push(model, bt, clean_last, f, "all", a.steps, a.margin)
        for i, (name, e) in enumerate(items):
            rec = {"case_id": e["case_id"], "set": name, "relation_id": e["relation_id"], "baseline_rank": e["baseline_rank"],
                   "band": band_of(e["baseline_rank"]), "n_candidates": int(e["cand_ids"].numel())}
            for mth in methods:
                b = res[mth]
                rec[mth] = {"rank": int(b.rank[i]), "prob": float(b.prob[i]), "kl": float(b.kl[i]), "push": float(b.push[i])}
            records.append(rec)
        el = time.time() - t0
        print(f"[batch {bi + 1}/{len(batches)}] elapsed {el / 60:.1f} min | ETA {(len(batches) - bi - 1) * el / (bi + 1) / 60:.1f} min", flush=True)

    # ---- summary ----
    print("\n================ HEADROOM SUMMARY: prompts reaching rank 1 ================")
    groups = defaultdict(list)
    for r in records:
        groups["ALL"].append(r)
        groups[r["set"]].append(r)
    order = ["ALL"] + [s for s in ["train", "val", "test_seen", "test_unseen"] if s in groups]
    head = f"{'method':<20}" + "".join(f"{g:>16}" for g in order)
    print(head)
    for mth in methods:
        row = f"{mth:<20}"
        for g in order:
            rs = groups[g]
            n1 = sum(1 for r in rs if r[mth]["rank"] == 1)
            row += f"{n1:>7}/{len(rs):<4}{100 * n1 / len(rs):>4.0f}%"
        print(row)
    print("\nMean side-effect KL (nats, over all tokens except target and blocker) and mean push size on the last token:")
    for mth in methods:
        rs = records
        print(f"  {mth:<20} KL {sum(r[mth]['kl'] for r in rs) / len(rs):.4f}   push {100 * sum(r[mth]['push'] for r in rs) / len(rs):.1f}% of residual norm")
    print("\nsae_reach by starting-rank band (ALL prompts):")
    for lo, hi in BANDS:
        rs = [r for r in records if r["band"] == f"{lo}-{hi}"]
        if rs:
            print(f"  rank {lo}-{hi}: {sum(1 for r in rs if r['sae_reach']['rank'] == 1)}/{len(rs)}"
                  f"   sae_gentle {sum(1 for r in rs if r['sae_gentle']['rank'] == 1)}/{len(rs)}"
                  f"   push_all_{fracs[-1]} {sum(1 for r in rs if r[f'push_all_{fracs[-1]}']['rank'] == 1)}/{len(rs)}")
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"headroom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(a), "methods": methods, "seconds": round(time.time() - t0, 1), "records": records}, f)
    print(f"\nWrote {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
