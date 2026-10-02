"""
Models and training machinery for the learned-strength study (docs/Research_Journal/23.md, sections 4 to 7).

WHAT IS LEARNED (version 1): for each prompt, one MUTE strength mu and one BOOST strength beta for the Hybrid editor.
    PromptNet     a small network: 12 summary numbers about the prompt -> (mu, beta)
    ConstantPair  the control: two shared numbers, no input (a "learned fixed pair")
    oracle        the upper bound: (mu, beta) optimised on that very prompt by gradient

HOW AN EDIT IS TRIED (no search, no labels): everything comes from the saved cache and strength tables.
  1. Which features are allowed at (mu, beta)?  The strict safety filter's verdict is looked up in the strength table
     at the nearest grid strength on each side (a candidate passes if the target's probability does not drop by more
     than 1e-6 and its rank does not worsen), then candidates passing on both sides are kept on the side where the target
     gains more, exactly as the sweep's round 0 does (src/hybrid_runner.py: build_pools).
  2. The edit is  delta = sum_k (m_k - 1) * activation_k * decoder_k  with m_k = 1 - mu (kept mute), 1 + beta (kept boost)
     or 1 (otherwise). It is added to the saved layer-8 state and ONLY blocks 8 to 11 are run (about 50 ms; a batch of
     32 prompts costs the same). The selection in step 1 is not differentiable; the edit in step 2 is, so a gradient
     tells mu and beta which way to move.
  3. Loss = rank hinge (target must beat the strongest other token by a margin)
          + lam_kl * KL(clean || edited) over all tokens except the target and the blocker
          + lam_size * (mu + beta / 2)                      (prefer the weakest edit that works)

THE PROXY (prefix-cumulative, mimics the sweep's round 0): the sweep adds features step by step (step k applies the first
k allowed mute features and the first k allowed boost features, in ranked order) and stops at the first step where the target
is rank 1. We evaluate a grid of prefix sizes (1, 2, 3, 5, 8, 13, 21, 34, 55 and "all") in one batched pass. A prompt counts
as a success if ANY prefix reaches rank 1; the reported side effect (KL) and feature counts are those of the FIRST successful
prefix (like the sweep), else of the closest one. Training minimises the best prefix's loss. This ignores the sweep's later
rounds (pool refill) and checks only the grid of prefix sizes, so it can undercount the sweep's successes; final claims must
come from real sweeps. (An earlier version applied ALL surviving features at once, about 190 features; its side effects were
far larger than the sweep's and the network collapsed to the weakest strengths.)

Strength range: the tables only cover mute 0.2 to 1.0 and boost 0.25 to 2.0, so the network's outputs are limited to those.
"""
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.strength_cache import SUMMARY_NAMES

PAD_ID = 50256
STRICT_PROB_TOL = 1e-6
MU_RANGE = (0.2, 1.0)
BETA_RANGE = (0.25, 2.0)
REFERENCE = (0.6, 0.5)
DEFAULT_PREFIXES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 0]          # 0 = all surviving features


# ----------------------------------------------------------------------------------------- data
def load_item(case_id, set_name, cache_dir, tables_dir):
    """Cache entry + strength table of one prompt, with the strict-filter masks precomputed."""
    c = torch.load(os.path.join(cache_dir, f"{case_id}.pt"), weights_only=False)
    t = torch.load(os.path.join(tables_dir, f"{case_id}.pt"), weights_only=False)
    assert torch.equal(c["cand_ids"], t["cand_ids"]), f"case {case_id}: table and cache candidate lists differ"
    cp, cr = t["clean_prob"], t["clean_rank"]
    ok_m = ((t["mute_prob"] - cp) >= -STRICT_PROB_TOL) & (t["mute_rank"] <= cr)          # [n_mute_grid, K]
    ok_b = ((t["boost_prob"] - cp) >= -STRICT_PROB_TOL) & (t["boost_rank"] <= cr)
    return {
        "case_id": int(case_id), "set": set_name, "relation_id": c["relation_id"], "baseline_rank": int(c["baseline_rank"]),
        "baseline_prob": float(c["baseline_prob"]), "prompt": c["prompt"], "target": c["target"],
        "target_id": int(c["target_id"]), "blocker_id": int(c["blocker_id"]),
        "resid": c["resid8"].float(), "toks": c["tokens"].long(), "acts": c["cand_acts"].float(), "ids": c["cand_ids"].long(),
        "ok_m": ok_m, "ok_b": ok_b, "md": (t["mute_prob"] - cp).float(), "bd": (t["boost_prob"] - cp).float(),
        "db": c["db"].float(), "dt": c["dt"].float(),
        "m_pos": torch.argsort(torch.argsort(c["db"], stable=True)), "b_pos": torch.argsort(torch.argsort(c["dt"], stable=True)),
        "feat": torch.tensor([c["summary"][n] for n in SUMMARY_NAMES], dtype=torch.float32),
        "mute_grid": list(t["mute_grid"]), "boost_grid": list(t["boost_grid"]), "n": int(c["tokens"].shape[0]),
    }


def make_batch(items, W_dec, dev):
    """Pad prompts into tensors (right padding is harmless: GPT-2 is causal; logits are read at each prompt's last token)."""
    B = len(items)
    P = max(it["n"] for it in items)
    K = max(it["ids"].numel() for it in items)
    G_m, G_b = items[0]["ok_m"].shape[0], items[0]["ok_b"].shape[0]
    big = 10 ** 6
    resid = torch.zeros(B, P, 768)
    toks = torch.full((B, P), PAD_ID, dtype=torch.long)
    acts = torch.zeros(B, P, K)
    ids = torch.zeros(B, K, dtype=torch.long)
    ok_m, ok_b = torch.zeros(B, G_m, K, dtype=torch.bool), torch.zeros(B, G_b, K, dtype=torch.bool)
    md, bd = torch.zeros(B, G_m, K), torch.zeros(B, G_b, K)
    m_pos, b_pos = torch.full((B, K), big, dtype=torch.long), torch.full((B, K), big, dtype=torch.long)
    for b, it in enumerate(items):
        n, k = it["n"], it["ids"].numel()
        resid[b, :n], toks[b, :n], acts[b, :n, :k], ids[b, :k] = it["resid"], it["toks"], it["acts"], it["ids"]
        ok_m[b, :, :k], ok_b[b, :, :k], md[b, :, :k], bd[b, :, :k] = it["ok_m"], it["ok_b"], it["md"], it["bd"]
        m_pos[b, :k], b_pos[b, :k] = it["m_pos"], it["b_pos"]
    to = lambda x: x.to(dev)
    return {"B": B, "P": P, "K": K, "resid": to(resid), "toks": to(toks), "acts": to(acts), "ok_m": to(ok_m), "ok_b": to(ok_b),
            "md": to(md), "bd": to(bd), "lens": torch.tensor([it["n"] for it in items], device=dev),
            "tgt": torch.tensor([it["target_id"] for it in items], device=dev),
            "blk": torch.tensor([it["blocker_id"] for it in items], device=dev),
            "feat": to(torch.stack([it["feat"] for it in items])),
            "m_order": to(torch.argsort(m_pos, dim=1)), "b_order": to(torch.argsort(b_pos, dim=1)),
            "mute_grid": torch.tensor(items[0]["mute_grid"], device=dev), "boost_grid": torch.tensor(items[0]["boost_grid"], device=dev),
            "wd": W_dec[to(ids)].detach(),                                  # [B, K, 768] decoder rows of the candidates
            "case_ids": [it["case_id"] for it in items], "baseline_rank": [it["baseline_rank"] for it in items]}


def iter_batches(items, batch_size, rng=None):
    """Batches of prompts with similar length. With an rng the grouping is jittered and shuffled every epoch."""
    keyed = sorted(items, key=lambda it: it["n"] + (rng.uniform(0, 2.5) if rng else 0.0))
    batches = [keyed[i:i + batch_size] for i in range(0, len(keyed), batch_size)]
    if rng:
        rng.shuffle(batches)
    return batches


# ------------------------------------------------------------------------------- edit + forward
def cap_mask(mask, order, cap):
    """Keep only the first `cap` surviving candidates of each row, in the sweep's order (order = argsort of rank positions)."""
    if not cap:
        return mask
    srt = mask.gather(1, order)
    keep = srt & (srt.long().cumsum(dim=1) <= cap)
    return torch.zeros_like(mask).scatter(1, order, keep)


def survivors(bt, mu, beta, cap=0):
    """Features allowed on each side at strengths (mu, beta): strict-filter lookup at the nearest grid strength, then the
    sweep's overlap rule (a candidate passing on both sides stays where the target gains more). `cap` keeps only the first
    `cap` allowed features per side in the sweep's order (a prefix); 0 keeps all."""
    ar = torch.arange(bt["B"], device=bt["resid"].device)
    gm = (mu.detach().unsqueeze(1) - bt["mute_grid"].unsqueeze(0)).abs().argmin(dim=1)
    gb = (beta.detach().unsqueeze(1) - bt["boost_grid"].unsqueeze(0)).abs().argmin(dim=1)
    pm, pb = bt["ok_m"][ar, gm], bt["ok_b"][ar, gb]
    md, bd = bt["md"][ar, gm], bt["bd"][ar, gb]
    both = pm & pb
    keep_boost = bd > md
    pm = pm & ~(both & keep_boost)
    pb = pb & ~(both & ~keep_boost)
    return cap_mask(pm, bt["m_order"], cap), cap_mask(pb, bt["b_order"], cap)


def edit_delta(bt, mu, beta, cap=0):
    pm, pb = survivors(bt, mu, beta, cap)
    mult_minus_1 = -mu.unsqueeze(1) * pm.float() + beta.unsqueeze(1) * pb.float()          # [B, K]
    delta = torch.einsum("bpk,bk,bkd->bpd", bt["acts"], mult_minus_1, bt["wd"])
    return delta, pm, pb


def forward_last(model, bt, delta):
    """Logits at each prompt's last real token after adding `delta` to the saved layer-8 state. Runs blocks 8 to 11 only and
    applies the final norm and unembedding to the last position only (saves memory)."""
    res = model(bt["resid"] + delta, start_at_layer=8, stop_at_layer=model.cfg.n_layers, tokens=bt["toks"])
    last = res[torch.arange(bt["B"], device=res.device), bt["lens"] - 1]
    x = model.ln_final(last.unsqueeze(1))[:, 0]
    return x @ model.W_U + model.b_U


def clean_last_logits(model, bt):
    with torch.no_grad():
        return forward_last(model, bt, torch.zeros_like(bt["resid"]))


def score(last, clean_last, bt):
    """rank, probability, KL(clean || edited) over all tokens except target and blocker (renormalised), margin."""
    ar = torch.arange(bt["B"], device=last.device)
    lp = F.log_softmax(last, dim=-1)
    tl = lp[ar, bt["tgt"]]
    rank = (lp > tl.unsqueeze(1)).sum(dim=1) + 1
    mask = torch.zeros_like(last, dtype=torch.bool)
    mask[ar, bt["tgt"]] = True
    mask[ar, bt["blk"]] = True
    lq = F.log_softmax(clean_last.masked_fill(mask, float("-inf")), dim=-1)
    lpe = F.log_softmax(last.masked_fill(mask, float("-inf")), dim=-1)
    kl = torch.where(mask, torch.zeros_like(lq), lq.exp() * (lq - lpe)).sum(dim=1)
    tgt_logit = last[ar, bt["tgt"]]
    margin = tgt_logit - last.masked_fill(F.one_hot(bt["tgt"], last.shape[1]).bool(), float("-inf")).max(dim=1).values
    return rank, tl.exp(), kl, margin


def per_prompt_loss(margin, kl, mu, beta, a):
    """Rank hinge everywhere; the side-effect penalty only where the target already leads by the margin (that is where the
    sweep would stop). Penalising side effects on prompts that fail anyway would just push every strength down."""
    lead = (margin >= a.margin).float().detach()
    return F.relu(a.margin - margin) + a.lam_kl * kl * lead + a.lam_size * (mu + beta / 2.0)


# ------------------------------------------------------------------------------------- models
def _logit(value, lo, hi):
    p = (value - lo) / (hi - lo)
    return math.log(p / (1 - p))


Z0 = [_logit(REFERENCE[0], *MU_RANGE), _logit(REFERENCE[1], *BETA_RANGE)]       # raw outputs that mean exactly (0.6, 0.5)


def to_strengths(z):
    mu = MU_RANGE[0] + (MU_RANGE[1] - MU_RANGE[0]) * torch.sigmoid(z[:, 0])
    beta = BETA_RANGE[0] + (BETA_RANGE[1] - BETA_RANGE[0]) * torch.sigmoid(z[:, 1])
    return mu, beta


class PromptNet(nn.Module):
    """12 summary numbers -> (mu, beta). Starts at the reference (0.6, 0.5) for every prompt."""
    def __init__(self, mean, std, hidden=32, dropout=0.1):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clamp_min(1e-6))
        self.net = nn.Sequential(nn.Linear(len(SUMMARY_NAMES), hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(Z0))

    def forward(self, feat):
        x = ((feat - self.mean) / self.std).clamp(-5, 5)
        return to_strengths(self.net(x))


class ConstantPair(nn.Module):
    """The control: one (mu, beta) shared by every prompt."""
    def __init__(self):
        super().__init__()
        self.z = nn.Parameter(torch.tensor(Z0))

    def forward(self, feat):
        return to_strengths(self.z.unsqueeze(0).expand(feat.shape[0], 2))


def make_net(kind, train_items, hidden, dropout, dev):
    if kind == "constant":
        return ConstantPair().to(dev)
    feats = torch.stack([it["feat"] for it in train_items])
    return PromptNet(feats.mean(dim=0), feats.std(dim=0), hidden, dropout).to(dev)


# ----------------------------------------------------------------------- training, evaluation
def rep(bt, R):
    """The batch repeated R times along the batch dimension (one copy per prefix size)."""
    return {"B": bt["B"] * R, "resid": bt["resid"].repeat(R, 1, 1), "toks": bt["toks"].repeat(R, 1), "lens": bt["lens"].repeat(R),
            "tgt": bt["tgt"].repeat(R), "blk": bt["blk"].repeat(R)}


def prefix_pass(model, bt, mu, beta, a, clean, prefixes=None):
    """Try every prefix size in one batched pass. Returns tensors of shape [R, B]."""
    prefixes = prefixes or a.prefix_list
    R = len(prefixes)
    deltas, nm, nb = [], [], []
    for n in prefixes:
        d, pm, pb = edit_delta(bt, mu, beta, n)
        deltas.append(d)
        nm.append(pm.sum(1))
        nb.append(pb.sum(1))
    big = rep(bt, R)
    last = forward_last(model, big, torch.cat(deltas, 0))
    rank, prob, kl, margin = score(last, clean.repeat(R, 1), big)
    shape = lambda x: x.view(R, bt["B"])
    return shape(rank), shape(prob), shape(kl), shape(margin), torch.stack(nm), torch.stack(nb)


def select(rank, prob, kl, margin, nm, nb, mu, beta, a):
    """Per prompt: the first prefix reaching rank 1 (as the sweep stops), else the one closest to it.
    Also returns the training loss = the best prefix's loss."""
    loss_rb = per_prompt_loss(margin, kl, mu.unsqueeze(0), beta.unsqueeze(0), a)            # [R, B]
    succ = rank == 1
    any_s = succ.any(dim=0)
    first = succ.int().argmax(dim=0)
    closest = (rank.float() * 1e4 + loss_rb.detach()).argmin(dim=0)
    idx = torch.where(any_s, first, closest).unsqueeze(0)
    g = lambda x: x.gather(0, idx)[0]
    return {"rank": g(rank), "prob": g(prob), "kl": g(kl), "n_mute": g(nm), "n_boost": g(nb), "loss_sel": g(loss_rb),
            "train_loss": loss_rb.min(dim=0).values, "success": any_s}


def run_batch_loss(model, net, bt, a):
    mu, beta = net(bt["feat"])
    clean = clean_last_logits(model, bt)
    sel = select(*prefix_pass(model, bt, mu, beta, a, clean, a.train_prefix_list), mu, beta, a)       # coarser grid while training
    return sel["train_loss"], sel, mu, beta


def _new_out():
    return {k: [] for k in ("case_id", "rank", "prob", "kl", "mu", "beta", "n_mute", "n_boost", "loss", "baseline_rank", "relation_id")}


def evaluate(model, strength_fn, items, a, W_dec, dev):
    """No-gradient evaluation of a strength rule on a list of prompts (prefix-cumulative proxy)."""
    out = _new_out()
    rel = {it["case_id"]: it["relation_id"] for it in items}
    with torch.no_grad():
        for chunk in iter_batches(items, a.batch):
            bt = make_batch(chunk, W_dec, dev)
            mu, beta = strength_fn(bt)
            clean = clean_last_logits(model, bt)
            sel = select(*prefix_pass(model, bt, mu, beta, a, clean), mu, beta, a)
            for k, v in (("rank", sel["rank"]), ("prob", sel["prob"]), ("kl", sel["kl"]), ("mu", mu), ("beta", beta),
                         ("loss", sel["train_loss"]), ("n_mute", sel["n_mute"]), ("n_boost", sel["n_boost"])):
                out[k] += v.tolist()
            out["case_id"] += bt["case_ids"]
            out["baseline_rank"] += bt["baseline_rank"]
            out["relation_id"] += [rel[c] for c in bt["case_ids"]]
    return out


def summarise(res):
    n = len(res["rank"])
    ok = [r == 1 for r in res["rank"]]
    fails = sorted(r for r in res["rank"] if r > 1)
    mean = lambda x: sum(x) / max(len(x), 1)
    std = lambda x: (sum((v - mean(x)) ** 2 for v in x) / max(len(x), 1)) ** 0.5
    return {"n": n, "rank1": sum(ok), "rank1_rate": sum(ok) / max(n, 1),
            "rank1_and_kl_le_0.1": sum(1 for r, k in zip(res["rank"], res["kl"]) if r == 1 and k <= 0.1) / max(n, 1),
            "median_rank_of_failures": (fails[len(fails) // 2] if fails else 0),
            "mean_kl": mean(res["kl"]), "mean_kl_of_successes": mean([k for r, k in zip(res["rank"], res["kl"]) if r == 1]),
            "mean_mu": mean(res["mu"]), "std_mu": std(res["mu"]), "mean_beta": mean(res["beta"]), "std_beta": std(res["beta"]),
            "mean_n_mute": mean(res["n_mute"]), "mean_n_boost": mean(res["n_boost"]), "mean_loss": mean(res["loss"])}


def train_one(model, kind, train_items, val_items, a, W_dec, dev, seed, log=None):
    """Train a PromptNet or ConstantPair; early stopping on the validation loss. Returns (net, history)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = make_net(kind, train_items, a.hidden, a.dropout, dev)
    # the control has only two numbers, so it needs a much larger step size than the network to be optimised fairly
    opt = torch.optim.AdamW(net.parameters(), lr=(a.const_lr if kind == "constant" else a.lr), weight_decay=(0.0 if kind == "constant" else a.wd))
    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(a.epochs):
        net.train()
        tot, cnt = 0.0, 0
        for chunk in iter_batches(train_items, a.batch, rng):
            bt = make_batch(chunk, W_dec, dev)
            loss_i, *_ = run_batch_loss(model, net, bt, a)
            loss = loss_i.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot, cnt = tot + float(loss) * len(chunk), cnt + len(chunk)
        net.eval()
        val = summarise(evaluate(model, lambda bt: net(bt["feat"]), val_items, a, W_dec, dev))
        hist.append({"epoch": epoch + 1, "train_loss": tot / cnt, "val_loss": val["mean_loss"], "val_rank1": val["rank1_rate"]})
        if log and (epoch % 5 == 0 or epoch == a.epochs - 1):
            log(f"      epoch {epoch + 1:>3}: train loss {tot / cnt:.3f} | val loss {val['mean_loss']:.3f} | val rank-1 {100 * val['rank1_rate']:.0f}%")
        if val["mean_loss"] < best - 1e-4:
            best, bad = val["mean_loss"], 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


def oracle(model, items, a, W_dec, dev, steps=100, lr=0.1):
    """Per-prompt optimisation of (mu, beta) by gradient on that very prompt: the upper bound a strength-picking network
    could aim for. Keeps, per prompt, the best iterate: a successful iterate with the lowest loss, else the lowest loss."""
    out = _new_out()
    rel = {it["case_id"]: it["relation_id"] for it in items}
    for chunk in iter_batches(items, a.batch):
        bt = make_batch(chunk, W_dec, dev)
        B = bt["B"]
        z = torch.tensor(Z0, device=dev).repeat(B, 1).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        clean = clean_last_logits(model, bt)
        best_key = torch.full((B,), float("inf"), device=dev)
        store = {k: torch.zeros(B, device=dev) for k in ("rank", "prob", "kl", "mu", "beta", "n_mute", "n_boost", "loss")}
        for _ in range(steps + 1):
            mu, beta = to_strengths(z)
            sel = select(*prefix_pass(model, bt, mu, beta, a, clean), mu, beta, a)
            key = (~sel["success"]).float() * 1e6 + sel["train_loss"].detach()
            better = key < best_key
            best_key = torch.where(better, key, best_key)
            for k, v in (("rank", sel["rank"].float()), ("prob", sel["prob"]), ("kl", sel["kl"]), ("mu", mu), ("beta", beta),
                         ("n_mute", sel["n_mute"].float()), ("n_boost", sel["n_boost"].float()), ("loss", sel["train_loss"])):
                store[k] = torch.where(better, v.detach().float(), store[k])
            opt.zero_grad()
            sel["train_loss"].sum().backward()
            opt.step()
        for k in store:
            out[k] += (store[k].round().long().tolist() if k == "rank" else store[k].tolist())
        out["case_id"] += bt["case_ids"]
        out["baseline_rank"] += bt["baseline_rank"]
        out["relation_id"] += [rel[c] for c in bt["case_ids"]]
    return out
