"""
Step S1 building blocks: the D6 ceiling test with a RANDOM basis instead of small's SAE decoder directions.

D6 tunes one multiplier a_k per candidate SAE feature; the change to small's layer-8 state is  delta[p] = sum_k a_k act_k[p] W_dec[k]  (activations act_k from the SAE, fixed).
Here W_dec[k] is replaced by one fixed random direction per feature (a Gaussian row rescaled to the norm of the decoder row), the same for every prompt; the activations (where and how much
each feature fires) are unchanged. Everything else (map, dose, loss, settings, judging) is as in D6. New code only; the loss helpers of src/gradient_editing.py are imported.
"""
import torch
import torch.nn.functional as F

from src.gradient_editing import Z_ZERO, _last_logits, _score, _to_a
from src.transfer.stitch import states


def random_basis(sae, seed=0):
    """[n_features, d_model]: one Gaussian row per feature, rescaled to the norm of the SAE decoder row."""
    g = torch.Generator().manual_seed(seed)
    W = sae.W_dec.detach()
    R = torch.randn(W.shape, generator=g).to(W.device)
    return R * (W.norm(dim=-1, keepdim=True) / R.norm(dim=-1, keepdim=True))


@torch.no_grad()
def change_with_basis(small, sae, tokens, recipe, basis, dev, zero_bos=True):
    """(h, delta): small's layer-8 state and the change the recipe makes with the given basis, at every position."""
    h = states(small, tokens, 8)[0]
    fid = torch.as_tensor(recipe["fids"], device=dev)
    a = torch.tensor(recipe["a"], device=dev, dtype=torch.float32)
    delta = torch.einsum("pk,k,kd->pd", sae.encode(h)[:, fid], a, basis[fid])
    if zero_bos:
        delta = delta.clone()
        delta[0] = 0
    return h, delta


def tune_with_basis(small, medium, sae, W, fids, tokens, target_id, q, dose, basis, steps=100, lr=0.1, lam_kl=3.0, lam_size=0.005, margin_target=0.3):
    """Same as src/transfer/b_aware_edit.tune_through_translator (the loss is measured on medium after the translated change is added) but the decoder directions are `basis[fids]`."""
    dev = tokens.device
    with torch.no_grad():
        h = states(small, tokens, 8)[0]
        fid_t = torch.as_tensor(fids, device=dev)
        acts = sae.encode(h)[:, fid_t]
        wd = basis[fid_t].detach()
        amax = acts.max(dim=0).values
        w = amax / amax.sum().clamp_min(1e-9)
        hm = states(medium, tokens, q)
        clean_last = _last_logits(medium, hm, tokens, q)
        blocker_id = int(torch.argmax(clean_last).item())
        keep = torch.ones(h.shape[0], 1, device=dev)
        keep[0] = 0
    z = torch.full((len(fids),), Z_ZERO, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    best_key, best_a, best_step, best_rank, best_kl, best_size = float("inf"), None, 0, None, None, None
    for step in range(steps + 1):
        a = _to_a(z)
        delta = torch.einsum("pk,k,kd->pd", acts, a, wd) * keep
        inj = dose * (delta @ W)
        last = _last_logits(medium, hm + inj[None], tokens, q)
        rank, prob, kl, margin = _score(last, clean_last, target_id, blocker_id)
        lead = float(margin.item() >= margin_target)
        loss = F.relu(margin_target - margin) + lam_kl * kl * lead + lam_size * (a.abs() * w).sum()
        key = (1e6 if rank > 1 else 0.0) + float(loss.item())
        if key < best_key:
            best_key, best_a, best_step, best_rank, best_kl = key, a.detach().clone(), step, rank, float(kl.item())
            best_size = float(inj[-1].norm().item() / hm[0, -1].norm().item())
        if step == steps:
            break
        (g,) = torch.autograd.grad(loss, [z])
        z.grad = g
        opt.step()
    return {"fids": list(fids), "a": best_a.tolist(), "best_step": best_step, "train_rank": best_rank, "train_kl": best_kl, "size_frac": best_size}
