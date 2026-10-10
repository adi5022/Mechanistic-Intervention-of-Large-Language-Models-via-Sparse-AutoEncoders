"""
Step D10 building blocks: a batched relay judge for a trained map.

For every test record the recipe of the original small-tuned edit (and, as controls, another record's recipe and the edit toward a rank-matched random word)
is re-applied in small to the tuned prompt, to the 2 rewordings and 5 neighbours (all three arms) and to the 12 unrelated prompts (real arm only), translated
by the map times the dose and added to medium's layer 16. Medium's answers are read at the last position:
  tuned prompt  top-1 and rank of the target (the random-word arm is judged on its own word), mean log-rank gain over the unchanged rank
  rewordings    PS = share where the new answer beats the true one
  neighbours    NS = share where the true answer still beats the new one
  unrelated     KL(unchanged || injected) and the share of top answers that flip
Same measures as D3, D5 and D7 (tools/transfer/06, 10 and 12); new code only.
"""
import math

import torch
import torch.nn.functional as F

from src.transfer.export_eval import make_recipe, plain_logits, small_change
from src.transfer.om_batched import batch_logits, pack
from src.transfer.stitch import states
from src.transfer.swap import NEUTRAL

ARMS = ("real", "wrong", "rand")
KINDS = ("tuned", "para", "neigh", "unrel")


@torch.no_grad()
def build_judge_items(small, medium, sae, test, edits, q, dev, log=None):
    """List of per-text dicts on the CPU: tok, dS, hm, tid (the word judged), base_rank (its rank in unchanged medium), true, rec, arm, kind."""
    items, n = [], len(test)
    for i, r in enumerate(test):
        e = edits[r["case_id"]]
        wrong = edits[test[(i + 7) % n]["case_id"]]
        arms = [(0, e["real"], r["tid_new"]), (1, wrong["real"], r["tid_new"])] + ([(2, e["random"], e["random"]["tid"])] if "random" in e else [])
        texts = [(0, r["prompt"])] + [(1, t) for t in r["paraphrases"]] + [(2, t) for t in r["neighbours"]]
        clean_cache = {}
        for arm, recipe, new in arms:
            rec = make_recipe(recipe, dev)
            for kind, text in texts + ([(3, t) for t in NEUTRAL] if arm == 0 else []):
                tok = small.to_tokens(text)
                if text not in clean_cache:
                    lg = plain_logits(medium, tok)
                    clean_cache[text] = (lg, states(medium, tok, q)[0])
                lg, hm = clean_cache[text]
                _, dS = small_change(small, sae, tok, rec)
                items.append({"tok": tok[0].cpu(), "dS": dS.cpu(), "hm": hm.cpu(), "tid": int(new), "true": int(r["tid_true"]),
                              "base_rank": int((lg > lg[new]).sum().item()) + 1, "rec": i, "arm": arm, "kind": kind})
        if log and (i + 1) % 25 == 0:
            log(i + 1, n)
    return items


def pack_judge(items, dev):
    S = pack(items, dev)
    for k in ("true", "rec", "arm", "kind"):
        S[k] = torch.tensor([it[k] for it in items], device=dev)
    return S


@torch.no_grad()
def judge_map(medium, S, W, dose, q, bs=64):
    """Per-text outputs for the packed judge set: rank of the judged word after the injection and unchanged, es (judged word beats the true word), KL and flips."""
    n = len(S["lens"])
    rank, rank0, es, kl, flip = [], [], [], [], []
    for s in range(0, n, bs):
        idx = torch.arange(s, min(s + bs, n), device=S["lens"].device)
        lg = batch_logits(medium, S, idx, W, dose, q)
        cl = batch_logits(medium, S, idx, W, 0.0, q)
        new, true = S["tid"][idx][:, None], S["true"][idx][:, None]
        rank.append((lg > lg.gather(1, new)).sum(-1) + 1)
        rank0.append((cl > cl.gather(1, new)).sum(-1) + 1)
        lp, lc = F.log_softmax(lg, -1), F.log_softmax(cl, -1)
        es.append((lp.gather(1, new) > lp.gather(1, true))[:, 0])
        kl.append((lc.exp() * (lc - lp)).sum(-1))
        flip.append(lg.argmax(-1) != cl.argmax(-1))
    cat = lambda xs: torch.cat(xs).cpu()
    return {"rank": cat(rank).float(), "rank0": cat(rank0).float(), "es": cat(es).float(), "kl": cat(kl), "flip": cat(flip).float()}


def summarize(S, res, rec_mask):
    """Rates for the records in rec_mask (bool tensor over test records): per arm top-1, median rank, mean log-rank gain, PS, NS, unrelated KL and flips."""
    arm, kind, rec = S["arm"].cpu(), S["kind"].cpu(), S["rec"].cpu()
    keep = rec_mask[rec]
    out = {}
    for ai, name in enumerate(ARMS):
        sel = lambda k: (arm == ai) & (kind == k) & keep
        t = sel(0)
        if t.sum() == 0:
            continue
        r, r0 = res["rank"][t], res["rank0"][t]
        d = {"n": int(t.sum()), "top1": float((r == 1).float().mean()), "median_rank": float(r.sort().values[len(r) // 2]), "gain": float((r0.log() - r.log()).mean())}
        p, nb, u = sel(1), sel(2), sel(3)
        d["PS"] = float(res["es"][p].mean()) if p.sum() else float("nan")
        d["NS"] = 1.0 - float(res["es"][nb].mean()) if nb.sum() else float("nan")
        if u.sum():
            d["unrelated_KL"], d["unrelated_flips"] = float(res["kl"][u].mean()), float(res["flip"][u].mean())
        out[name] = d
    return out


def per_record_gain(S, res, arm_index=0):
    """Mean log-rank gain at the tuned prompt per test record (real arm by default): tensor [n_records], nan where missing."""
    arm, kind, rec = S["arm"].cpu(), S["kind"].cpu(), S["rec"].cpu()
    t = (arm == arm_index) & (kind == 0)
    n = int(rec.max()) + 1
    out = torch.full((n,), float("nan"))
    out[rec[t]] = res["rank0"][t].log() - res["rank"][t].log()
    return out
