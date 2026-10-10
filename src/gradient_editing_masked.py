"""
Position-masked version of the gradient-descent multiplier edit (new file; src/gradient_editing.py is NOT changed).

Same method, same loss and settings as `run_gradient_descent_edit` (one multiplier per candidate feature, rank hinge + KL once the target leads +
size term, Adam, best iterate kept), except that the change is only applied at the positions where `pos_mask` is 1. Used by the fact-or-word-push
test (Phase C): an edit that works when it may only touch the SUBJECT's tokens acts on how the subject is represented (where ROME finds facts);
an edit that works when it may only touch the LAST position acts on the read-out.

The edit is judged through the same shortcut path the original uses while tuning (blocks `layer`..11 run from the edited residual state), which the
project verified against the real hook to 8.8e-06 (handoff 2026-10-05, section on the shortcut path).
"""
import torch

from src.editing import get_top_active_features
from src.gradient_editing import Z_ZERO, _last_logits, _score, _to_a
import torch.nn.functional as F


def subject_positions(model, text, subject):
    """Token positions (counting the start token as 0) of `subject` inside `text`, or None if it cannot be located cleanly."""
    i = text.find(subject)
    if i < 0:
        return None
    prefix, lead = text[:i], ""
    if prefix.endswith(" "):
        prefix, lead = prefix[:-1], " "
    start = model.to_tokens(prefix).shape[1] if prefix else 1
    n = len(model.tokenizer(lead + subject, add_special_tokens=False)["input_ids"])
    ids = model.to_tokens(text)[0]
    pos = list(range(start, start + n))
    if pos[-1] >= len(ids) - 0 or model.tokenizer.decode(ids[pos].tolist()).strip() != subject.strip():
        return None
    return pos


def run_masked_edit(model, sae, clean_ctx, target_id, prompt, layer, pos_mask, top_n=200, positions="all", steps=100, lr=0.1,
                    lam_kl=3.0, lam_size=0.005, margin_target=0.3):
    """Returns {"fids", "a", "rank", "kl", "best_step"}; rank and kl are the edit's effect through the shortcut path."""
    device = clean_ctx.resid_all.device
    cands = get_top_active_features(model, sae, prompt, top_n=top_n, clean_ctx=clean_ctx, positions=positions)
    fids = [f for f, _ in cands]
    K = len(fids)
    if K == 0:
        raise ValueError("No active candidate features for this prompt and candidate source.")
    resid = clean_ctx.resid_all.detach()
    tokens = clean_ctx.tokens
    blocker_id = int(torch.argmax(clean_ctx.clean_probs).item())
    mask = pos_mask.to(device).float()[:, None]
    with torch.no_grad():
        acts = sae.encode(resid[0])[:, fids] * mask
        wd = sae.W_dec[fids].detach()
        amax = acts.max(dim=0).values
        w = amax / amax.sum().clamp_min(1e-9)
        clean_last = _last_logits(model, resid, tokens, layer)
    z = torch.full((K,), Z_ZERO, device=device).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [z], "lr": lr}])
    best_key, best_a, best_step, best_rank, best_kl = float("inf"), None, 0, None, None
    for step in range(steps + 1):
        a = _to_a(z)
        delta = torch.einsum("pk,k,kd->pd", acts, a, wd).unsqueeze(0)
        last = _last_logits(model, resid + delta, tokens, layer)
        rank, prob, kl, margin = _score(last, clean_last, target_id, blocker_id)
        lead = float(margin.item() >= margin_target)
        loss = F.relu(margin_target - margin) + lam_kl * kl * lead + lam_size * (a.abs() * w).sum()
        key = (1e6 if rank > 1 else 0.0) + float(loss.item())
        if key < best_key:
            best_key, best_a, best_step, best_rank, best_kl = key, a.detach().clone(), step, rank, float(kl.item())
        if step == steps:
            break
        g = torch.autograd.grad(loss, [z])[0]
        z.grad = g
        opt.step()
    return {"fids": fids, "a": best_a.tolist(), "rank": best_rank, "kl": best_kl, "best_step": best_step}
