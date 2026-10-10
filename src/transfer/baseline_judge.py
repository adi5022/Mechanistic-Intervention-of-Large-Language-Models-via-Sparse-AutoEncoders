"""
Step S3 building block: the relay judge of src/transfer/om_judge.py (judge_map) plus the unchanged level of the judged word, so that every arm, the random-word arm included, can be compared
with its OWN unchanged level (S2 showed that comparing a random word with the real target's unchanged level is not fair). Same maths as judge_map; new code only.
"""
import torch
import torch.nn.functional as F

from src.transfer.om_batched import batch_logits


@torch.no_grad()
def judge_with_baseline(medium, S, W, dose, q, bs=64):
    """Per packed text: rank of the judged word after the injection (rank) and unchanged (rank0), es / es0 = the judged word beats the true word with / without the injection (float 0 or 1)."""
    n = len(S["lens"])
    rank, rank0, es, es0 = [], [], [], []
    for s in range(0, n, bs):
        idx = torch.arange(s, min(s + bs, n), device=S["lens"].device)
        lg = batch_logits(medium, S, idx, W, dose, q)
        cl = batch_logits(medium, S, idx, W, 0.0, q)
        new, true = S["tid"][idx][:, None], S["true"][idx][:, None]
        rank.append((lg > lg.gather(1, new)).sum(-1) + 1)
        rank0.append((cl > cl.gather(1, new)).sum(-1) + 1)
        lp, lc = F.log_softmax(lg, -1), F.log_softmax(cl, -1)
        es.append((lp.gather(1, new) > lp.gather(1, true))[:, 0])
        es0.append((lc.gather(1, new) > lc.gather(1, true))[:, 0])
    cat = lambda xs: torch.cat(xs).cpu()
    return {"rank": cat(rank).float(), "rank0": cat(rank0).float(), "es": cat(es).float(), "es0": cat(es0).float()}


def per_record(S, res, arm_index, kind_index, key, n_records):
    """Mean of res[key] over the texts of each test record for one arm and one kind of text: tensor [n_records], nan where the record has no such text."""
    arm, kind, rec = S["arm"].cpu(), S["kind"].cpu(), S["rec"].cpu()
    out = torch.full((n_records,), float("nan"))
    for i in range(n_records):
        m = (rec == i) & (arm == arm_index) & (kind == kind_index)
        if m.any():
            out[i] = res[key][m].mean()
    return out
