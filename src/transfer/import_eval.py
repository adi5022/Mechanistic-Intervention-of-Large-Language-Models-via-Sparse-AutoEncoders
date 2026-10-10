"""
Steps E2 to E4 building blocks: Import (medium's state written into small).

Replacement ("the stitched model"): at every position but the first, small's layer-qS state is replaced by medium's layer-qM state translated into small's
coordinates, T(h_M) = (h_M - mu_src) W + mu_dst; small's remaining layers and its output then run as normal. Secondary arms add a (filtered) difference
instead. The answer is never given to small; the true answer's id is only used to judge (and in the reported word-push control). New code only.
"""
import torch

from src.editing import get_target_token_id


def make_facts(small, ds, rome_dev, pilot):
    """All CounterFact facts as dicts: prompt, true answer id, token length (start token included), paraphrases, and whether the fact is excluded from the decision counts."""
    facts = []
    for i, e in enumerate(ds):
        rw = e["requested_rewrite"]
        true = rw["target_true"]["str"]
        prompt = rw["prompt"].format(rw["subject"])
        facts.append({"i": i, "case_id": e["case_id"], "prompt": prompt, "subject": rw["subject"], "true": true, "tid": get_target_token_id(small, " " + true),
                      "n_tok": len(small.tokenizer(prompt)["input_ids"]), "paraphrases": e["paraphrase_prompts"][:2],
                      "excluded": i in rome_dev or e["case_id"] in pilot})
    return facts


def length_batches(idx, lens, bs):
    """Batches of indices whose prompts have the same token length (so no padding is needed)."""
    groups = {}
    for j in idx:
        groups.setdefault(lens[j], []).append(j)
    for _, js in sorted(groups.items()):
        for s in range(0, len(js), bs):
            yield js[s:s + bs]


@torch.no_grad()
def states_batch(model, toks, layer):
    name = f"blocks.{layer}.hook_resid_pre"
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n == name, stop_at_layer=layer + 1)
    return cache[name]


def translate_state(h, entry, W=None):
    """T(h) = (h - mu_src) W + mu_dst (W can be replaced, for the random-map control)."""
    return (h - entry["mu_src"]) @ (entry["W"] if W is None else W) + entry["mu_dst"]


@torch.no_grad()
def small_last_logits(small, toks, layer, replace=None, add=None):
    """Last-position logits of small [B, V]; with `replace` [B, L, 768] the layer state is replaced at every position but the first, with `add` it is added to there."""
    if replace is None and add is None:
        return small(toks)[:, -1].float()
    name = f"blocks.{layer}.hook_resid_pre"

    def hook(resid, hook):
        out = resid.clone()
        if replace is not None:
            out[:, 1:] = replace[:, 1:].to(out.dtype)
        if add is not None:
            out[:, 1:] = out[:, 1:] + add[:, 1:].to(out.dtype)
        return out

    return small.run_with_hooks(toks, fwd_hooks=[(name, hook)])[:, -1].float()


def top_and_rank(lg, tid):
    """(top-1 token id, rank of the true answer) for logits [B, V] and true ids [B]."""
    t = lg.gather(1, tid[:, None])
    return lg.argmax(-1), (lg > t).sum(-1) + 1


@torch.no_grad()
def sae_filtered_delta(sae, T, hS, k):
    """Decoded difference of the k features per position with the largest |f(T) - f(h_S)|: [B, L, 768]."""
    df = sae.encode(T) - sae.encode(hS)
    idx = df.abs().topk(k, dim=-1).indices
    mask = torch.zeros_like(df).scatter_(-1, idx, 1.0)
    return (df * mask) @ sae.W_dec
