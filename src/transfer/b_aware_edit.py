"""
Step D6: tune an edit through the translator (the ceiling test).

Same edit family as src/gradient_editing.py: one multiplier m_k = 1 + a_k (a_k in [-1, 2]) per candidate SAE feature of GPT-2 small's layer 8,
the same loss (rank hinge with a margin, KL once the target leads, size penalty) and the same settings. The difference is where the loss is
measured: small's change delta = sum_k a_k act_k W_dec[k] is translated with the linear map W (small 8 -> medium q), scaled by a fixed dose, added
to MEDIUM's layer-q state, and the hinge and KL are taken on MEDIUM's last-position logits. Gradients run through the map and medium's layers q..end
to the multipliers only. Nothing is changed in the old modules; their helpers are imported.
"""
import torch
import torch.nn.functional as F

from src.gradient_editing import Z_ZERO, _last_logits, _score, _to_a
from src.transfer.stitch import states


def tune_through_translator(small, medium, sae, W, fids, tokens, target_id, q, dose, steps=100, lr=0.1, lam_kl=3.0, lam_size=0.005, margin_target=0.3):
    """Returns {'fids', 'a', 'best_step', 'train_rank', 'train_kl', 'size_frac'} where size_frac = ||injected change|| / ||medium state|| at the last position."""
    dev = tokens.device
    with torch.no_grad():
        h = states(small, tokens, 8)[0]                                           # [L, 768] small's layer-8 state
        fid_t = torch.as_tensor(fids, device=dev)
        acts = sae.encode(h)[:, fid_t]                                            # [L, K]
        wd = sae.W_dec[fid_t].detach()                                            # [K, 768]
        amax = acts.max(dim=0).values
        w = amax / amax.sum().clamp_min(1e-9)
        hm = states(medium, tokens, q)                                            # [1, L, 1024]
        clean_last = _last_logits(medium, hm, tokens, q)
        blocker_id = int(torch.argmax(clean_last).item())
        keep = torch.ones(h.shape[0], 1, device=dev)
        keep[0] = 0                                                               # the start-token row is left alone, as in every export run

    z = torch.full((len(fids),), Z_ZERO, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    best_key, best_a, best_step, best_rank, best_kl = float("inf"), None, 0, None, None
    for step in range(steps + 1):
        a = _to_a(z)
        delta = torch.einsum("pk,k,kd->pd", acts, a, wd) * keep                   # [L, 768]
        inj = dose * (delta @ W)                                                  # [L, 1024]
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
