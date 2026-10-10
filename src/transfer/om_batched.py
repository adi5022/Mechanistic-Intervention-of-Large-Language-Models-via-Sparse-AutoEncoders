"""
Step D8 building blocks: a batched, faster version of the D7 training (src/transfer/output_matching.py trains one text at a time).

Same maths as D7: loss = KL(goal || log_softmax(medium's last-position logits after adding dose * (delta @ W) at layer q)), goal built once per text.
What changes is only the execution: all texts are padded to one length and sent through medium's layers q..end together. A causal model never lets an
earlier position see a later (padded) one, so the padding does not change the numbers; the logits of each text are taken at its own last real position.
Optionally the blocks run in half precision (the final norm and unembedding stay in float32).
Nothing in the older modules is touched.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transfer.export_eval import make_recipe, plain_logits, small_change
from src.transfer.stitch import states


@torch.no_grad()
def build_eval_item(small, medium, sae, text, recipe, q, dev, tid):
    tok = small.to_tokens(text)
    _, dS = small_change(small, sae, tok, make_recipe(recipe, dev))
    lg = plain_logits(medium, tok)
    return {"tok": tok[0], "dS": dS, "hm": states(medium, tok, q)[0], "tid": tid, "base_rank": int((lg > lg[tid]).sum().item()) + 1}


def pack(items, dev, with_goal=False):
    """List of per-text dicts (tok [L], dS [L,768], hm [L,1024], optional goal [V] half, tid, base_rank) -> padded tensors on the device."""
    n, lmax = len(items), max(it["dS"].shape[0] for it in items)
    S = {"dS": torch.zeros(n, lmax, items[0]["dS"].shape[1], device=dev), "hm": torch.zeros(n, lmax, items[0]["hm"].shape[1], device=dev),
         "tok": torch.zeros(n, lmax, dtype=torch.long, device=dev), "lens": torch.tensor([it["dS"].shape[0] for it in items], device=dev)}
    for i, it in enumerate(items):
        L = it["dS"].shape[0]
        S["dS"][i, :L], S["hm"][i, :L], S["tok"][i, :L] = it["dS"].to(dev), it["hm"].to(dev), it["tok"].to(dev)
    if with_goal:
        S["goal"] = torch.stack([it["goal"] for it in items]).to(dev)
    if "tid" in items[0]:
        S["tid"] = torch.tensor([it["tid"] for it in items], device=dev)
        S["base_rank"] = torch.tensor([it["base_rank"] for it in items], device=dev, dtype=torch.float32)
    return S


def batch_logits(medium, S, idx, W, dose, q, amp=False):
    """Last-position logits [B, V] for texts idx with dose * (dS @ W) added to medium's layer-q state."""
    lens = S["lens"][idx]
    lb = int(lens.max())
    resid = S["hm"][idx, :lb] + dose * (S["dS"][idx, :lb] @ W)
    tok = S["tok"][idx, :lb]
    if amp:
        with torch.autocast("cuda", dtype=torch.float16):
            res = medium(resid, start_at_layer=q, stop_at_layer=medium.cfg.n_layers, tokens=tok)
        res = res.float()
    else:
        res = medium(resid, start_at_layer=q, stop_at_layer=medium.cfg.n_layers, tokens=tok)
    last = res[torch.arange(len(idx), device=res.device), lens - 1]
    x = medium.ln_final(last[:, None, :])[:, 0]
    return x @ medium.W_U + medium.b_U


def batch_kl(medium, S, idx, W, dose, q, amp=False):
    """Per-text KL(goal || medium after injection), shape [B]."""
    logp = F.log_softmax(batch_logits(medium, S, idx, W, dose, q, amp).float(), dim=-1)
    g = S["goal"][idx].float()
    return (g.exp() * (g - logp)).sum(-1)


@torch.no_grad()
def eval_ranks(medium, S, W, dose, q, bs=64, amp=False):
    """Rank of each text's target (1 = top answer) after injection; tensor [N]."""
    out = []
    for s in range(0, len(S["lens"]), bs):
        idx = torch.arange(s, min(s + bs, len(S["lens"])), device=S["lens"].device)
        lg = batch_logits(medium, S, idx, W, dose, q, amp)
        out.append((lg > lg.gather(1, S["tid"][idx][:, None])).sum(-1) + 1)
    return torch.cat(out)


def score(S, ranks, subset=None):
    """(top-1 rate, mean log-rank gain over the unchanged rank, median rank) on all texts or on the index tensor `subset`."""
    r, b = (ranks, S["base_rank"]) if subset is None else (ranks[subset], S["base_rank"][subset])
    r = r.float()
    # median: the upper of the two middle values for an even count, like the other tools (med() in tools/transfer/06 and 12); torch's median is the lower one
    return float((r == 1).float().mean()), float((b.log() - r.log()).mean()), float(r.sort().values[len(r) // 2])


def train_map_horizons(medium, W0, train, dev_set, q, epochs=40, horizons=(15, 40), lr=2e-4, reg=0.1, batch=32, seed=0, amp=False):
    """Same training as train_map (identical updates), but keeps, for each horizon h, the epoch <= h with the best dev mean log-rank gain at dose 1.
    One long run therefore also gives the shorter run: with the same seed the first h epochs are the same as a run that stops at h.
    Returns {h: (best W, best epoch)} and the per-epoch history."""
    W = nn.Parameter(W0.clone())
    opt = torch.optim.Adam([W], lr=lr)
    w0n = (W0 ** 2).sum()
    g = torch.Generator().manual_seed(seed)
    n = len(train["lens"])
    best = {h: (-1e9, W0.clone(), 0) for h in horizons}
    hist = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).to(W0.device)
        tot, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = batch_kl(medium, train, idx, W, 1.0, q, amp).mean()
            opt.zero_grad()
            (loss + reg * ((W - W0) ** 2).sum() / w0n).backward()
            opt.step()
            tot, nb = tot + float(loss.item()), nb + 1
        top1, gain, mr = score(dev_set, eval_ranks(medium, dev_set, W.detach(), 1.0, q, amp=amp))
        hist.append({"epoch": ep, "train_kl": tot / nb, "dev_top1": top1, "dev_gain": gain, "dev_median_rank": mr})
        for h in horizons:
            if ep <= h and gain > best[h][0]:
                best[h] = (gain, W.detach().clone(), ep)
    return {h: (best[h][1], best[h][2]) for h in horizons}, hist


def train_map(medium, W0, train, dev_set, q, epochs=15, lr=2e-4, reg=0.1, batch=32, seed=0, amp=False, log=None):
    """Train a linear map starting at W0 on the packed training set; keep the epoch with the best dev mean log-rank gain (dose 1).
    Returns (best W, best epoch, per-epoch history)."""
    W = nn.Parameter(W0.clone())
    opt = torch.optim.Adam([W], lr=lr)
    w0n = (W0 ** 2).sum()
    g = torch.Generator().manual_seed(seed)
    n = len(train["lens"])
    best_gain, best_W, best_ep, hist = -1e9, W0.clone(), 0, []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).to(W0.device)
        tot, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = batch_kl(medium, train, idx, W, 1.0, q, amp).mean()
            opt.zero_grad()
            (loss + reg * ((W - W0) ** 2).sum() / w0n).backward()
            opt.step()
            tot, nb = tot + float(loss.item()), nb + 1
        top1, gain, mr = score(dev_set, eval_ranks(medium, dev_set, W.detach(), 1.0, q, amp=amp))
        hist.append({"epoch": ep, "train_kl": tot / nb, "dev_top1": top1, "dev_gain": gain, "dev_median_rank": mr})
        if gain > best_gain:
            best_gain, best_W, best_ep = gain, W.detach().clone(), ep
        if log:
            log(ep, hist[-1], best_ep == ep)
    return best_W, best_ep, hist
