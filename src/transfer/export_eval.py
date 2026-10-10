"""
Helpers for step D3/D4 (controlled export of a gradient-descent edit from GPT-2 small to GPT-2 medium).

A "recipe" is what the edit tuned in small is: feature ids and a_k (multiplier m_k = 1 + a_k). Applied to ANY prompt, the change it makes to
small's layer-8 state is  delta[p] = sum_k a_k * act_k[p] * W_dec[k]  at every position p (act = the SAE activations of that prompt). That
re-application is what "relay mode" means: a reworded prompt gets the same recipe applied to its own activations.
The change is translated into medium's coordinates (linear map, or a neural map: translated change = f(h + delta) - f(h)), multiplied by a
dose, and added to medium's residual stream at the chosen layer while it reads the same prompt. The start-of-text position is left alone
(its state is unusual and the map was fitted without it) except in one ablation.
"""
import torch
import torch.nn.functional as F

from src.transfer.stitch import states
from src.transfer.swap import run_injected


def make_recipe(edit, dev):
    return torch.tensor(edit["fids"], device=dev), torch.tensor(edit["a"], device=dev, dtype=torch.float32)


@torch.no_grad()
def small_change(small, sae, tokens, recipe, zero_bos=True):
    """(h, delta): small's layer-8 state [L, 768] and the change the recipe makes to it, at every position."""
    h = states(small, tokens, 8)[0]
    fids, a = recipe
    acts = sae.encode(h)[:, fids]
    delta = torch.einsum("pk,k,kd->pd", acts, a, sae.W_dec[fids].detach())
    if zero_bos:
        delta = delta.clone()
        delta[0] = 0
    return h, delta


class LinearTranslator:
    name = "linear"

    def __init__(self, W):
        self.W = W

    @torch.no_grad()
    def __call__(self, h, delta, keep_bos=False):
        out = delta @ self.W
        if not keep_bos:
            out[0] = 0
        return out


class MlpTranslator:
    """Neural map f from small's state to medium's state; the translated change is f(h + delta) - f(h)."""
    name = "mlp"

    def __init__(self, f):
        self.f = f

    @torch.no_grad()
    def __call__(self, h, delta, keep_bos=False):
        out = self.f(h + delta) - self.f(h)
        if not keep_bos:
            out[0] = 0
        return out


def make_arms(Dt, Dwrong, push_dir, gen):
    """The translated change and its controls, all the same size per position as Dt (the wrong recipe keeps its own size)."""
    rnd = torch.randn(Dt.shape, generator=gen).to(Dt.device)
    rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp(min=1e-9) * Dt.norm(dim=-1, keepdim=True)
    push = torch.zeros_like(Dt)
    push[-1] = push_dir / push_dir.norm() * Dt[-1].norm()
    return {"translated": Dt, "random": rnd, "wrong_recipe": Dwrong, "word_push": push}


def norm_match_factors(h_small, dS, h_med, Dt):
    """Per-position factor so that the injected change has, relative to medium's residual norm, the same size as the original edit has
    relative to small's residual norm. Chosen without looking at any outcome."""
    share = dS.norm(dim=-1) / h_small.norm(dim=-1).clamp(min=1e-9)
    target = share * h_med.norm(dim=-1)
    n = Dt.norm(dim=-1)
    s = target / n.clamp(min=1e-9)
    s[n < 1e-9] = 0
    s[0] = 0
    return s


def evaluate(model, tokens, deltas, layer, tid_new, tid_true, clean_logits=None):
    """Inject each of `deltas` [P, L, d] in turn (batched) and read the last position. Returns per-row arrays:
    rank_new (1 = top), top1 (new target is the top answer), es (log P(new) > log P(true)), ld (log P(new) - log P(true)), and with
    clean_logits: kl (clean || injected) and flip (the top answer changed)."""
    P = len(deltas)
    lg = run_injected(model, tokens.repeat(P, 1), deltas, layer)
    return summarise_logits(lg, tid_new, tid_true, clean_logits)


def summarise_logits(lg, tid_new, tid_true, clean_logits=None):
    lp = F.log_softmax(lg, dim=-1)
    out = {
        "rank_new": (lg > lg[:, tid_new:tid_new + 1]).sum(-1).float() + 1,
        "top1": (lg.argmax(-1) == tid_new).float(),
        "ld": lp[:, tid_new] - lp[:, tid_true],
    }
    out["es"] = (out["ld"] > 0).float()
    if clean_logits is not None:
        lc = F.log_softmax(clean_logits, dim=-1)
        out["kl"] = (lc.exp() * (lc - lp)).sum(-1)
        out["flip"] = (lg.argmax(-1) != clean_logits.argmax(-1)).float()
    return {k: v.cpu() for k, v in out.items()}


@torch.no_grad()
def plain_logits(model, tokens):
    return model(tokens)[0, -1].float()
