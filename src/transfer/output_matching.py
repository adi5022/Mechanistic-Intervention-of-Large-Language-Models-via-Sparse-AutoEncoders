"""
Step D7: a translator trained with an output-matching loss.

The edit is tuned in GPT-2 small as usual (multipliers on SAE features of layer 8). Its change to small's layer-8 state, delta, is translated with a
linear map W (768 -> 1024) and added, times a dose, to GPT-2 medium's layer-q state. W is trained so that medium's answers change the way small's
answers change under the edit. Both models share one vocabulary, so the two output changes can be compared token by token:

    goal   = log_softmax( medium_clean_logits + (small_edited_logits - small_clean_logits) )
    loss   = KL( goal || log_softmax(medium logits after adding dose * delta @ W) )                   (zero at no edit)

Each training item is one text (the tuned prompt, a rewording or a neighbour prompt of a training record) with that record's small-tuned recipe
re-applied to it. Nothing here touches the evaluation records. Old modules are imported, not edited.
"""
import torch
import torch.nn.functional as F

from src.gradient_editing import _last_logits
from src.transfer.export_eval import make_recipe, plain_logits, small_change
from src.transfer.stitch import states
from src.transfer.swap import run_injected


@torch.no_grad()
def build_item(small, medium, sae, text, recipe, q, dev):
    """Everything the training loop needs for one text, computed once."""
    tok = small.to_tokens(text)
    rec = make_recipe(recipe, dev)
    _, dS = small_change(small, sae, tok, rec)                                    # start-token row zeroed: this is what goes through the map
    _, dS_full = small_change(small, sae, tok, rec, zero_bos=False)               # small's own edited run keeps the whole change
    zA0 = plain_logits(small, tok)
    zA1 = run_injected(small, tok, dS_full[None], 8)[0]
    zB0 = plain_logits(medium, tok)
    return {"tok": tok, "dS": dS, "hm": states(medium, tok, q), "goal": F.log_softmax(zB0 + zA1 - zA0, dim=-1).half()}


def item_loss(medium, item, W, dose, q):
    inj = dose * (item["dS"] @ W)
    last = _last_logits(medium, item["hm"] + inj[None], item["tok"], q)
    logp = F.log_softmax(last, dim=-1)
    g = item["goal"].float()
    return (g.exp() * (g - logp)).sum()


@torch.no_grad()
def dev_scores(medium, items, W, dose, q):
    """items: dicts with tok, dS, hm, tid, base_rank. Returns (top-1 rate, mean log-rank gain, median rank)."""
    import math
    ranks = []
    for it in items:
        last = _last_logits(medium, it["hm"] + (dose * (it["dS"] @ W))[None], it["tok"], q)
        ranks.append(int((last > last[it["tid"]]).sum().item()) + 1)
    top1 = sum(r == 1 for r in ranks) / len(ranks)
    gain = sum(math.log(it["base_rank"]) - math.log(r) for it, r in zip(items, ranks)) / len(ranks)
    return top1, gain, sorted(ranks)[len(ranks) // 2]
