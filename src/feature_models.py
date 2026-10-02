"""
Version 2 of the learned-strength study (docs/Research_Journal/23.md, section 6.3): PER-FEATURE multipliers.

Version 1 (src/strength_models.py) chose ONE mute strength and ONE boost strength per prompt, applied to whole pools of
features. Here a small network gives EVERY candidate feature its own multiplier:

        multiplier m_k = 1 + a_k,   a_k in [-1, beta_max]
        a_k < 0  mutes feature k by |a_k|      a_k > 0  boosts it by a_k      a_k = 0  leaves it alone

FeatureNet is a shared scoring network applied to each candidate separately (the same weights for all 200 candidates).
For candidate k of a prompt it reads 15 descriptors of the candidate (activation, positions, removal effects on the
target and the blocker, direct-logit alignment with the target and blocker, decoder norm, list ranks, and the strict
filter's verdict at 0.6 / 0.5; see CDESC_NAMES in src/strength_models.py) plus the 12 prompt-level summary numbers.
The strict safety filter is not a separate step: a feature that hurts the target makes the loss worse, so its multiplier
is pushed back toward 1 (a "learned filter"). `strict_guard` can additionally drop features the strict rule would reject.

THE EDIT IS ONE-SHOT: all multipliers are applied together (no step-by-step addition, no refill), by adding
delta = sum_k a_k * activation_k * decoder_k to the saved layer-8 state and running blocks 8 to 11. This is a different
mechanism from the sweep, so it is evaluated against the version-1 rules under the SAME one-shot proxy (prefix list [0]).

Loss per prompt: rank hinge (target must beat the strongest other token by a margin)
               + lam_kl * KL(clean || edited) over tokens except target and blocker, only where the target already leads
               + lam_size * sum_k |a_k| w_k   (w_k = activation share of candidate k: prefer few, small changes)
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import strength_models as sm
from src.strength_models import CDESC_NAMES, SUMMARY_NAMES, iter_batches, make_batch, clean_last_logits, forward_last, score

BETA_MAX = 2.0
FILTER_INPUTS = [CDESC_NAMES.index(n) for n in ("passes_strict_mute_0.6", "passes_strict_boost_0.5",
                                                "slog_target_change_mute_0.6", "slog_target_change_boost_0.5")]


def to_a(z, beta_max=BETA_MAX):
    """raw network output -> a in [-1, beta_max]"""
    return -1.0 + (1.0 + beta_max) * torch.sigmoid(z)


Z_ZERO = -math.log(BETA_MAX)             # sigmoid(Z_ZERO) = 1 / (1 + beta_max)  ->  a = 0 (no edit)


class FeatureNet(nn.Module):
    """Shared scorer: (candidate descriptors + prompt summary) -> multiplier offset a_k. Starts at a = 0 for every candidate."""
    def __init__(self, c_mean, c_std, p_mean, p_std, hidden=32, dropout=0.1, use_filter_inputs=True, beta_max=BETA_MAX):
        super().__init__()
        self.beta_max = beta_max
        self.register_buffer("c_mean", c_mean.clone())
        self.register_buffer("c_std", c_std.clamp_min(1e-6))
        self.register_buffer("p_mean", p_mean.clone())
        self.register_buffer("p_std", p_std.clamp_min(1e-6))
        keep = torch.ones(len(CDESC_NAMES))
        if not use_filter_inputs:
            keep[FILTER_INPUTS] = 0.0
        self.register_buffer("keep", keep)
        d_in = len(CDESC_NAMES) + len(SUMMARY_NAMES)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        with torch.no_grad():
            self.net[-1].bias.fill_(Z_ZERO)

    def forward(self, bt):
        c = ((bt["cdesc"] - self.c_mean) / self.c_std).clamp(-5, 5) * self.keep               # [B, K, Dc]
        p = ((bt["feat"] - self.p_mean) / self.p_std).clamp(-5, 5)                           # [B, 12]
        x = torch.cat([c, p.unsqueeze(1).expand(-1, c.shape[1], -1)], dim=-1)
        a = to_a(self.net(x).squeeze(-1), self.beta_max)                                     # [B, K]
        return a * bt["kmask"]


def make_feature_net(train_items, hidden, dropout, use_filter_inputs, dev):
    allc = torch.cat([it["cdesc"] for it in train_items], dim=0)
    feats = torch.stack([it["feat"] for it in train_items])
    return FeatureNet(allc.mean(0), allc.std(0), feats.mean(0), feats.std(0), hidden, dropout, use_filter_inputs).to(dev)


def strict_guard(bt, a):
    """Drop (set to 0) every feature the strict safety filter would reject at the strength this edit gives it
    (nearest grid strength in the strength tables, mute side if a < 0, boost side if a > 0)."""
    mu_k = (-a).clamp(min=0)
    be_k = a.clamp(min=0)
    gm = (mu_k.unsqueeze(-1) - bt["mute_grid"].view(1, 1, -1)).abs().argmin(-1)              # [B, K]
    gb = (be_k.unsqueeze(-1) - bt["boost_grid"].view(1, 1, -1)).abs().argmin(-1)
    ok_m = bt["ok_m"].gather(1, gm.unsqueeze(1)).squeeze(1)
    ok_b = bt["ok_b"].gather(1, gb.unsqueeze(1)).squeeze(1)
    keep = torch.where(a < 0, ok_m, torch.where(a > 0, ok_b, torch.ones_like(ok_m)))
    return a * keep.float()


def feature_delta(bt, a):
    return torch.einsum("bpk,bk,bkd->bpd", bt["acts"], a, bt["wd"])


def run_edit(model, bt, a, clean):
    last = forward_last(model, bt, feature_delta(bt, a))
    return score(last, clean, bt)                                                             # rank, prob, kl, margin


def feature_loss(margin, kl, a, bt, args):
    w = bt["amax"] / bt["amax"].sum(dim=1, keepdim=True).clamp_min(1e-9)
    lead = (margin >= args.margin).float().detach()
    return F.relu(args.margin - margin) + args.lam_kl * kl * lead + args.lam_size_feat * (a.abs() * w).sum(dim=1)


def _out():
    return {k: [] for k in ("case_id", "rank", "prob", "kl", "n_mute", "n_boost", "mean_abs_a", "loss", "baseline_rank", "relation_id")}


def evaluate_feature(model, a_fn, items, args, W_dec, dev, guard=False):
    """One-shot evaluation of a per-feature rule `a_fn(bt) -> a [B, K]` (no gradient)."""
    out = _out()
    rel = {it["case_id"]: it["relation_id"] for it in items}
    with torch.no_grad():
        for chunk in iter_batches(items, args.batch):
            bt = make_batch(chunk, W_dec, dev)
            a = a_fn(bt)
            if guard:
                a = strict_guard(bt, a)
            clean = clean_last_logits(model, bt)
            rank, prob, kl, margin = run_edit(model, bt, a, clean)
            loss_i = feature_loss(margin, kl, a, bt, args)
            for k, v in (("rank", rank), ("prob", prob), ("kl", kl), ("loss", loss_i),
                         ("n_mute", (a < -0.05).sum(1)), ("n_boost", (a > 0.05).sum(1)),
                         ("mean_abs_a", (a.abs() * bt["kmask"]).sum(1) / bt["kmask"].sum(1))):
                out[k] += v.tolist()
            out["case_id"] += bt["case_ids"]
            out["baseline_rank"] += bt["baseline_rank"]
            out["relation_id"] += [rel[c] for c in bt["case_ids"]]
    return out


def summarise_feature(res):
    n = len(res["rank"])
    ok = [r == 1 for r in res["rank"]]
    mean = lambda x: sum(x) / max(len(x), 1)
    return {"n": n, "rank1": sum(ok), "rank1_rate": sum(ok) / max(n, 1),
            "mean_kl_of_successes": mean([k for r, k in zip(res["rank"], res["kl"]) if r == 1]), "mean_kl_all": mean(res["kl"]),
            "mean_n_mute": mean(res["n_mute"]), "mean_n_boost": mean(res["n_boost"]), "mean_abs_a": mean(res["mean_abs_a"]),
            "mean_loss": mean(res["loss"])}


def train_feature_net(model, train_items, val_items, args, W_dec, dev, seed, log=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = make_feature_net(train_items, args.hidden, args.dropout, not args.no_filter_inputs, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        net.train()
        tot, cnt = 0.0, 0
        for chunk in iter_batches(train_items, args.batch, rng):
            bt = make_batch(chunk, W_dec, dev)
            clean = clean_last_logits(model, bt)
            a = net(bt)
            rank, prob, kl, margin = run_edit(model, bt, a, clean)
            loss = feature_loss(margin, kl, a, bt, args).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot, cnt = tot + float(loss) * len(chunk), cnt + len(chunk)
        net.eval()
        val = summarise_feature(evaluate_feature(model, net, val_items, args, W_dec, dev))
        hist.append({"epoch": epoch + 1, "train_loss": tot / cnt, "val_loss": val["mean_loss"], "val_rank1": val["rank1_rate"]})
        if log and (epoch % 5 == 0 or epoch == args.epochs - 1):
            log(f"      epoch {epoch + 1:>3}: train loss {tot / cnt:.3f} | val loss {val['mean_loss']:.3f} | val rank-1 {100 * val['rank1_rate']:.0f}%")
        if val["mean_loss"] < best - 1e-4:
            best, bad = val["mean_loss"], 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


def oracle_feature(model, items, args, W_dec, dev, steps=100, lr=0.1):
    """Per-prompt free optimisation of every candidate's multiplier on that very prompt (upper bound for this loss; not a
    trained model). Keeps the best iterate: a successful one with the lowest loss, else the lowest loss."""
    out = _out()
    rel = {it["case_id"]: it["relation_id"] for it in items}
    for chunk in iter_batches(items, args.batch):
        bt = make_batch(chunk, W_dec, dev)
        z = torch.full((bt["B"], bt["K"]), Z_ZERO, device=dev).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        clean = clean_last_logits(model, bt)
        best_key = torch.full((bt["B"],), float("inf"), device=dev)
        store = {k: torch.zeros(bt["B"], device=dev) for k in ("rank", "prob", "kl", "n_mute", "n_boost", "mean_abs_a", "loss")}
        for _ in range(steps + 1):
            a = to_a(z) * bt["kmask"]
            rank, prob, kl, margin = run_edit(model, bt, a, clean)
            loss_i = feature_loss(margin, kl, a, bt, args)
            key = (rank > 1).float() * 1e6 + loss_i.detach()
            better = key < best_key
            best_key = torch.where(better, key, best_key)
            vals = {"rank": rank.float(), "prob": prob, "kl": kl, "n_mute": (a < -0.05).sum(1).float(), "n_boost": (a > 0.05).sum(1).float(),
                    "mean_abs_a": (a.abs() * bt["kmask"]).sum(1) / bt["kmask"].sum(1), "loss": loss_i}
            for k, v in vals.items():
                store[k] = torch.where(better, v.detach().float(), store[k])
            opt.zero_grad()
            loss_i.sum().backward()
            opt.step()
        for k in store:
            out[k] += (store[k].round().long().tolist() if k == "rank" else store[k].tolist())
        out["case_id"] += bt["case_ids"]
        out["baseline_rank"] += bt["baseline_rank"]
        out["relation_id"] += [rel[c] for c in bt["case_ids"]]
    return out
