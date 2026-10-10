"""
Step D14 building blocks: maps from small's change (768) to medium's layer-16 change (1024) that have less freedom to memorise than the full linear map of D7 to D13.

All maps start at the ridge map W0 (step A1) and only the change of an edit is passed through them, so every map must send a zero change to zero (no biases).
  LowRankMap       x W0 + (x A) B, A [768, r], B [r, 1024] starting at zero: only r directions of the map can move.
  ResidualMlpDelta x W + s_y * down(dropout(gelu(up(x / s_x)))): the ridge map (trainable, with a penalty in the trainer) plus a small non-linear correction that starts at zero.
The trainer is the D13 trainer (src/transfer/om_batched.train_map_hinge with lam = 0) written for a module: same batches for the same seed, same output-matching KL loss, same epoch rule
(best dev mean log-rank gain at dose 1). To reuse the evaluation functions of the earlier steps unchanged, the translated change of a packed set is computed once (`translated`) and the
evaluation functions are then called with an identity matrix as the map. Nothing in the older modules is touched.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transfer.om_batched import batch_logits, eval_ranks, score


class LowRankMap(nn.Module):
    def __init__(self, W0, rank=16):
        super().__init__()
        self.register_buffer("W0", W0.clone())
        self.A = nn.Parameter(torch.randn(W0.shape[0], rank, device=W0.device) / W0.shape[0] ** 0.5)
        self.B = nn.Parameter(torch.zeros(rank, W0.shape[1], device=W0.device))

    def forward(self, x):
        return x @ self.W0 + (x @ self.A) @ self.B

    def matrix(self):
        return (self.W0 + self.A @ self.B).detach()


class ResidualMlpDelta(nn.Module):
    def __init__(self, W0, s_x, s_y, hidden=256, p=0.2):
        super().__init__()
        self.register_buffer("W0", W0.clone())
        self.W = nn.Parameter(W0.clone())
        self.up = nn.Linear(W0.shape[0], hidden, bias=False).to(W0.device)
        self.down = nn.Linear(hidden, W0.shape[1], bias=False).to(W0.device)
        nn.init.zeros_(self.down.weight)
        self.drop = nn.Dropout(p)
        self.s_x, self.s_y = float(s_x), float(s_y)

    def forward(self, x):
        return x @ self.W + self.s_y * self.down(self.drop(F.gelu(self.up(x / self.s_x))))


@torch.no_grad()
def typical_sizes(S, W0, n_texts=400):
    """(s_x, s_y): root-mean-square entry of the change dS before and after the ridge map, over the non-zero positions of the first texts of a packed set."""
    dS = S["dS"][:n_texts]
    rows = dS[dS.abs().sum(-1) > 0]
    return float(rows.pow(2).mean().sqrt()), float((rows @ W0).pow(2).mean().sqrt())


@torch.no_grad()
def translated(S, g, bs=256):
    """Copy of a packed set whose dS is the translated change g(dS) [N, L, 1024] (the module in eval mode); use it with `identity` as the map."""
    g.eval()
    out = dict(S)
    out["dS"] = torch.cat([g(S["dS"][s:s + bs]) for s in range(0, len(S["lens"]), bs)])
    return out


def identity(dim, dev):
    return torch.eye(dim, device=dev)


def train_module(medium, g, groups, train, dev_set, q, penalty=None, epochs=15, batch=32, seed=0, log=None):
    """The D13 training for a module g. `groups` are the optimiser parameter groups (the optimiser is AdamW when a group has a weight decay, else Adam; both with the given lrs),
    `penalty` an optional function g -> scalar added to the loss. Returns (state dict of the best epoch by dev mean log-rank gain at dose 1, that epoch, history)."""
    dev_ = train["hm"].device
    I = identity(medium.cfg.d_model, dev_)
    opt = torch.optim.AdamW(groups) if any(gr.get("weight_decay", 0) > 0 for gr in groups) else torch.optim.Adam(groups)
    gen = torch.Generator().manual_seed(seed)
    n = len(train["lens"])
    best_gain, best_state, best_ep, hist = -1e9, copy.deepcopy(g.state_dict()), 0, []
    for ep in range(1, epochs + 1):
        g.train()
        perm = torch.randperm(n, generator=gen).to(dev_)
        tot_kl, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            lens = train["lens"][idx]
            lb = int(lens.max())
            Sb = {"hm": train["hm"][idx, :lb], "dS": g(train["dS"][idx, :lb]), "tok": train["tok"][idx, :lb], "lens": lens}
            lg = batch_logits(medium, Sb, torch.arange(len(idx), device=dev_), I, 1.0, q).float()
            logp = F.log_softmax(lg, dim=-1)
            gl = train["goal"][idx].float()
            kl = (gl.exp() * (gl - logp)).sum(-1).mean()
            loss = kl if penalty is None else kl + penalty(g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_kl, nb = tot_kl + float(kl.item()), nb + 1
        top1, gain, mr = score(dev_set, eval_ranks(medium, translated(dev_set, g), I, 1.0, q))
        hist.append({"epoch": ep, "train_kl": tot_kl / nb, "dev_top1": top1, "dev_gain": gain, "dev_median_rank": mr})
        if gain > best_gain:
            best_gain, best_state, best_ep = gain, copy.deepcopy(g.state_dict()), ep
        if log:
            log(ep, hist[-1], best_ep == ep)
    return best_state, best_ep, hist
