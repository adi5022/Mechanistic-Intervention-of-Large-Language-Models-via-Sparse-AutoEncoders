"""
Activation editing/steering, intervention logic, and the Causal Feature Selector.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from src.hooks import make_ablation_hook, make_scale_map_hook, with_extra_scale

HOOK_NAME = "blocks.8.hook_resid_pre"

def get_target_token_id(model, target_str: str) -> int:
    """
    Safely resolves a target string (e.g. ' Paris') to its token ID in the model's vocabulary.
    """
    token_ids = model.to_tokens(target_str, prepend_bos=False).squeeze()
    if token_ids.numel() > 1:
        return token_ids[-1].item()
    return token_ids.item()

@dataclass
class CleanContext:
    """
    Caches everything derivable from a single clean (unablated) forward pass
    on a given prompt, so downstream functions don't have to recompute it.
    """
    tokens: torch.Tensor          # model.to_tokens(prompt) output
    clean_probs: torch.Tensor     # softmax over vocab at final position, [vocab]
    resid_last: torch.Tensor      # residual stream at the SAE hook point, final token position
    clean_target_prob: float      # clean_probs[target_token_id].item(), if a target is known
    clean_rank: int                # rank of target_token_id in clean_probs, if a target is known
    resid_all: torch.Tensor = None # residual stream at the SAE hook point, ALL positions [1, seq, d_model]


def build_clean_context(model, sae, prompt: str, target_token_id: int = None) -> CleanContext:
    """
    Runs the model ONCE with run_with_cache, extracting both the full
    vocabulary distribution and the residual stream activation needed for
    SAE encoding. If target_token_id is provided, also computes clean
    probability and rank for that token.
    """
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)

    # Defensive reset: model/sae are shared, cached resources (e.g. Streamlit's
    # @st.cache_resource). If a prior run's hooks are still attached — e.g. a rerun was
    # triggered mid-forward-pass by a widget interaction — run_with_cache can silently
    # cache under different hook names, causing a KeyError below. Resetting immediately
    # before the pass, and retrying once on failure, recovers from that race instead of
    # crashing the whole app on an otherwise-transient glitch.
    model.reset_hooks()
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        clean_probs = F.softmax(logits[0, -1, :], dim=-1)
        try:
            resid_all = cache[hook_name]
            resid_last = resid_all[:, -1, :]  # [1, d_model]
        except KeyError:
            model.reset_hooks()
            logits, cache = model.run_with_cache(tokens)
            clean_probs = F.softmax(logits[0, -1, :], dim=-1)
            resid_all = cache[hook_name]
            resid_last = resid_all[:, -1, :]

    clean_target_prob = None
    clean_rank = None
    if target_token_id is not None:
        clean_target_prob = clean_probs[target_token_id].item()
        sorted_indices = torch.argsort(clean_probs, descending=True)
        clean_rank = (sorted_indices == target_token_id).nonzero().item() + 1

    return CleanContext(
        tokens=tokens,
        clean_probs=clean_probs,
        resid_last=resid_last,
        clean_target_prob=clean_target_prob,
        clean_rank=clean_rank,
        resid_all=resid_all,
    )

def build_steered_context(model, sae, clean_ctx: CleanContext, base_scale_map: dict,
                          target_token_id: int = None) -> CleanContext:
    """
    Same shape as CleanContext, but its distribution/rank/prob come from a forward pass with
    the already-applied features in base_scale_map active (the "steered baseline").
    resid_last is reused from clean_ctx: the SAE hook point sits upstream of every
    intervention, so the active-feature set at that point does not change with steering.
    """
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    model.reset_hooks()
    with torch.no_grad():
        logits = model.run_with_hooks(
            clean_ctx.tokens, fwd_hooks=[(hook_name, make_scale_map_hook(base_scale_map, sae))]
        )
        probs = F.softmax(logits[0, -1, :], dim=-1)
    model.reset_hooks()

    target_prob = None
    rank = None
    if target_token_id is not None:
        target_prob = probs[target_token_id].item()
        rank = (torch.argsort(probs, descending=True) == target_token_id).nonzero().item() + 1

    return CleanContext(
        tokens=clean_ctx.tokens,
        clean_probs=probs,
        resid_last=clean_ctx.resid_last,
        clean_target_prob=target_prob,
        clean_rank=rank,
        resid_all=clean_ctx.resid_all,
    )


def get_top_active_features(
    model, sae, prompt: str, top_n: int = 20, clean_ctx: CleanContext = None,
    exclude_ids=None, positions: str = "last"
) -> List[Tuple[int, float]]:
    """
    Runs the model on the prompt, encodes the residual stream with the SAE, and returns the top_n
    feature indices and their raw activations.

    positions="last": only the final prompt token (original behaviour).
    positions="all":  every prompt token except BOS; a feature's score is its maximum activation
                      over those positions, so features that fire on earlier words (e.g. "sky")
                      become candidates too.
    """
    if positions == "all":
        if clean_ctx is not None and clean_ctx.resid_all is not None:
            resid_all = clean_ctx.resid_all
        else:
            tokens = model.to_tokens(prompt)
            hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
            model.reset_hooks()
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            resid_all = cache[hook_name]
        with torch.no_grad():
            acts = sae.encode(resid_all[0])          # [seq, n_features]
            if acts.shape[0] > 1:
                acts = acts[1:]                      # drop BOS
            best = acts.max(dim=0).values            # [n_features]
            if exclude_ids:
                best = best.clone()
                best[torch.as_tensor(list(exclude_ids), device=best.device)] = 0.0
        values, indices = torch.topk(best, min(top_n, best.numel()))
        return [(idx.item(), val.item()) for idx, val in zip(indices, values) if val.item() > 0.0]

    if clean_ctx is not None:
        last_token_act = clean_ctx.resid_last
    else:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        
        # Get last token activation from residual stream
        hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
        resid_act = cache[hook_name]  # [batch, seq_len, d_model]
        last_token_act = resid_act[:, -1, :]  # [batch, d_model]
    
    # Encode to SAE features
    with torch.no_grad():
        feature_acts = sae.encode(last_token_act)  # [batch, n_features]
        last_token_features = feature_acts[0]  # [n_features]
        if exclude_ids:
            last_token_features = last_token_features.clone()
            last_token_features[torch.as_tensor(list(exclude_ids), device=last_token_features.device)] = 0.0
        
    values, indices = torch.topk(last_token_features, top_n)
    
    return [(idx.item(), val.item()) for idx, val in zip(indices, values) if val.item() > 0.0]

def run_causal_selector(
    model, sae, prompt: str, target_str: str, top_n: int = 20
) -> List[Dict]:
    """
    Ranks the top-N active features by their causal effect on the target token's probability
    when fully ablated (strength=1.0).
    """
    target_token_id = get_target_token_id(model, target_str)
    tokens = model.to_tokens(prompt)
    
    # Get baseline (clean) probabilities
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        clean_target_prob = clean_probs[target_token_id].item()
    
    # Get top active features
    active_features = get_top_active_features(model, sae, prompt, top_n=top_n)
    
    results = []
    for feature_id, activation in active_features:
        hook_fn = make_ablation_hook(feature_id, sae, strength=1.0)
        hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
        
        with torch.no_grad():
            # Run model with feature fully ablated
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_target_prob = ablated_probs[target_token_id].item()
            
            # Compute top-1 token after ablation
            top1_id = torch.argmax(ablated_probs).item()
            top1_token = model.to_string([top1_id])
            
            # Causal effect: change in target token probability
            prob_delta = ablated_target_prob - clean_target_prob
            
            # KL divergence: D_KL(clean || ablated) to measure distribution shift
            kl = torch.sum(clean_probs * torch.log((clean_probs + 1e-10) / (ablated_probs + 1e-10))).item()
            
        results.append({
            "feature_id": feature_id,
            "activation": activation,
            "clean_prob": clean_target_prob,
            "ablated_prob": ablated_target_prob,
            "prob_delta": prob_delta,
            "kl_divergence": kl,
            "top1_prediction": top1_token
        })
        
    # Rank by magnitude of causal effect (negative delta means ablated prob dropped, i.e., feature was positive for prediction)
    # Sorting by prob_delta ascending (largest drop first)
    results.sort(key=lambda x: x["prob_delta"])
    return results

def _single_scale_hook(sae, base_scale_map, feature_id: int, scale: float):
    """Original single-feature hook when nothing is applied; otherwise the same feature scaled on top of the applied set."""
    if not base_scale_map:
        return make_ablation_hook(feature_id, sae, strength=1.0 - scale)
    return make_scale_map_hook(with_extra_scale(base_scale_map, feature_id, scale), sae)


def get_top_competitor_features(
    model, sae, prompt: str, current_top_token_id: int, top_n: int = 20,
    clean_ctx: CleanContext = None, use_batched: bool = False,
    exclude_ids=None, base_scale_map: dict | None = None, positions: str = "last"
) -> list[tuple[int, float]]:
    """
    Ranks active features by how much their ablation (strength=1.0) decreases the probability of current_top_token_id.
    Returns: list of (feature_id, prob_delta) sorted by prob_delta ascending (largest drop first).
    """
    if clean_ctx is not None:
        tokens = clean_ctx.tokens
        clean_competitor_prob = clean_ctx.clean_probs[current_top_token_id].item()
    else:
        tokens = model.to_tokens(prompt)
        # 1. Get baseline probability of the competitor
        with torch.no_grad():
            clean_logits = model(tokens)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            clean_competitor_prob = clean_probs[current_top_token_id].item()
        
    # 2. Get top active features
    active_features = get_top_active_features(model, sae, prompt, top_n=top_n, clean_ctx=clean_ctx, exclude_ids=exclude_ids, positions=positions)
    if not active_features:
        return []

    if use_batched:
        from src.batched_eval import batched_ablation_probs
        hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
        feature_ids = [fid for fid, _ in active_features]
        
        ablated_probs_tensor = batched_ablation_probs(
            model=model,
            sae=sae,
            tokens=tokens,
            feature_ids=feature_ids,
            scale=0.0,
            token_ids_of_interest=[current_top_token_id],
            hook_name=hook_name,
            base_scale_map=base_scale_map
        )
        ablated_competitor_probs = ablated_probs_tensor[:, 0].tolist()
        results = [
            (fid, ab_prob - clean_competitor_prob)
            for fid, ab_prob in zip(feature_ids, ablated_competitor_probs)
        ]
        results.sort(key=lambda x: x[1])
        return results

    results = []
    for feature_id, activation in active_features:
        hook_fn = _single_scale_hook(sae, base_scale_map, feature_id, 0.0)
        hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
        
        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_competitor_prob = ablated_probs[current_top_token_id].item()
            
            # Causal effect on the competitor
            prob_delta = ablated_competitor_prob - clean_competitor_prob
            
        results.append((feature_id, prob_delta))
        
    # Sort by prob_delta ascending (biggest drop / most negative delta first)
    results.sort(key=lambda x: x[1])
    return results

def get_top_target_features(
    model, sae, prompt: str, target_token_id: int, top_n: int = 30,
    clean_ctx: CleanContext = None, use_batched: bool = False,
    exclude_ids=None, base_scale_map: dict | None = None, positions: str = "last"
) -> list[tuple[int, float]]:
    """
    Ranks active features by how much their ablation (strength=1.0) decreases the probability of target_token_id.
    Returns: list of (feature_id, prob_delta) sorted by prob_delta ascending (largest target drop first).
    """
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    if clean_ctx is not None:
        tokens = clean_ctx.tokens
        clean_target_prob = clean_ctx.clean_probs[target_token_id].item()
    else:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            clean_logits = model(tokens)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            clean_target_prob = clean_probs[target_token_id].item()
        
    active_features = get_top_active_features(model, sae, prompt, top_n=top_n, clean_ctx=clean_ctx, exclude_ids=exclude_ids, positions=positions)
    if not active_features:
        return []

    if use_batched:
        from src.batched_eval import batched_ablation_probs
        feature_ids = [fid for fid, _ in active_features]
        
        ablated_probs_tensor = batched_ablation_probs(
            model=model,
            sae=sae,
            tokens=tokens,
            feature_ids=feature_ids,
            scale=0.0,
            token_ids_of_interest=[target_token_id],
            hook_name=hook_name,
            base_scale_map=base_scale_map
        )
        ablated_target_probs = ablated_probs_tensor[:, 0].tolist()
        results = [
            (fid, ab_prob - clean_target_prob)
            for fid, ab_prob in zip(feature_ids, ablated_target_probs)
        ]
        results.sort(key=lambda x: x[1])
        return results

    results = []
    for feature_id, activation in active_features:
        hook_fn = _single_scale_hook(sae, base_scale_map, feature_id, 0.0)
        
        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_target_prob = ablated_probs[target_token_id].item()
            
            prob_delta = ablated_target_prob - clean_target_prob
            
        results.append((feature_id, prob_delta))
        
    results.sort(key=lambda x: x[1])
    return results

def check_target_safe(
    model, sae, prompt: str, feature_id: int, target_token_id: int, strength: float = 0.3,
    clean_target_prob: float = None, clean_rank: int = None,
    base_scale_map: dict | None = None
) -> tuple[bool, float]:
    """
    Temporarily applies ONLY this one feature's ablation (using make_ablation_hook),
    measures the resulting change in the TARGET token's probability, and returns
    (is_safe, target_prob_delta) where is_safe = True if target_prob_delta >= 0 (target wasn't hurt).
    Resets hooks after measurement. Accepts optional precomputed clean_target_prob and clean_rank.
    """
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    # Baseline target prob under current model state (recomputed only if not precomputed)
    if clean_target_prob is None or clean_rank is None:
        with torch.no_grad():
            clean_logits = model(tokens)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            clean_target_prob = clean_probs[target_token_id].item()
            clean_sorted_indices = torch.argsort(clean_probs, descending=True)
            clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = _single_scale_hook(sae, base_scale_map, feature_id, 1.0 - strength)
    
    # Run with temporary ablation hook for feature_id
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)]
        )
        ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
        ablated_target_prob = ablated_probs[target_token_id].item()
        ablated_sorted_indices = torch.argsort(ablated_probs, descending=True)
        ablated_rank = (ablated_sorted_indices == target_token_id).nonzero().item() + 1
        
    target_prob_delta = ablated_target_prob - clean_target_prob
    is_safe = bool(ablated_rank <= clean_rank and target_prob_delta >= -1e-6)
    
    return is_safe, target_prob_delta

def check_boost_safe(
    model, sae, prompt: str, feature_id: int, target_token_id: int, strength: float = 0.5,
    clean_target_prob: float = None, clean_rank: int = None,
    base_scale_map: dict | None = None
) -> tuple[bool, float, int]:
    """
    Temporarily applies ONLY this one feature's boost (using make_signed_ablation_hook with +strength),
    measures the resulting change in the TARGET token's probability and rank, and returns
    (is_safe, target_prob_delta, rank_improvement) where is_safe = True if target rank improves or stays same,
    and target probability does not decrease. Accepts optional precomputed clean_target_prob and clean_rank.
    """
    from src.hooks import make_signed_ablation_hook
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    if clean_target_prob is None or clean_rank is None:
        with torch.no_grad():
            clean_logits = model(tokens)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            clean_target_prob = clean_probs[target_token_id].item()
            clean_sorted_indices = torch.argsort(clean_probs, descending=True)
            clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
        
    hook_fn = _single_scale_hook(sae, base_scale_map, feature_id, 1.0 + strength) if base_scale_map else make_signed_ablation_hook([feature_id], sae, strength=+strength)
    
    with torch.no_grad():
        boosted_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, hook_fn)]
        )
        boosted_probs = F.softmax(boosted_logits[0, -1, :], dim=-1)
        boosted_target_prob = boosted_probs[target_token_id].item()
        boosted_sorted_indices = torch.argsort(boosted_probs, descending=True)
        boosted_rank = (boosted_sorted_indices == target_token_id).nonzero().item() + 1
        
    target_prob_delta = boosted_target_prob - clean_target_prob
    rank_improvement = clean_rank - boosted_rank
    
    is_safe = bool(boosted_rank <= clean_rank and target_prob_delta >= -1e-6)
    
    return is_safe, target_prob_delta, rank_improvement

def check_target_safe_batch(
    model, sae, clean_ctx: CleanContext, feature_ids: list[int],
    target_token_id: int, strength: float = 0.3,
    base_scale_map: dict | None = None
) -> list[tuple[bool, float]]:
    """
    Batched counterpart of check_target_safe.
    Evaluates all listed feature_ids in one or few batched forward passes using scale = (1.0 - strength).
    Returns list of (is_safe, target_prob_delta).
    Decision logic matches single-item version:
    mute-safe: ablated_rank <= clean_rank and target_prob_delta >= -1e-6
    """
    if not feature_ids:
        return []

    from src.batched_eval import batched_ablation_probs_and_ranks
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    scale = 1.0 - strength

    target_probs_tensor, ranks_tensor = batched_ablation_probs_and_ranks(
        model=model,
        sae=sae,
        tokens=clean_ctx.tokens,
        feature_ids=feature_ids,
        scale=scale,
        target_token_id=target_token_id,
        hook_name=hook_name,
        base_scale_map=base_scale_map
    )

    clean_target_prob = clean_ctx.clean_target_prob
    clean_rank = clean_ctx.clean_rank

    target_probs = target_probs_tensor.tolist()
    ranks = ranks_tensor.tolist()

    results = []
    for prob, ablated_rank in zip(target_probs, ranks):
        delta = prob - clean_target_prob
        is_safe = bool(ablated_rank <= clean_rank and delta >= -1e-6)
        results.append((is_safe, delta))

    return results


def check_boost_safe_batch(
    model, sae, clean_ctx: CleanContext, feature_ids: list[int],
    target_token_id: int, strength: float = 0.5,
    base_scale_map: dict | None = None
) -> list[tuple[bool, float, int]]:
    """
    Batched counterpart of check_boost_safe.
    Evaluates all listed feature_ids in one or few batched forward passes using scale = (1.0 + strength).
    Returns list of (is_safe, target_prob_delta, rank_improvement).
    Decision logic matches single-item version:
    boost-safe: boosted_rank <= clean_rank and target_prob_delta >= -1e-6
    """
    if not feature_ids:
        return []

    from src.batched_eval import batched_ablation_probs_and_ranks
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    scale = 1.0 + strength

    target_probs_tensor, ranks_tensor = batched_ablation_probs_and_ranks(
        model=model,
        sae=sae,
        tokens=clean_ctx.tokens,
        feature_ids=feature_ids,
        scale=scale,
        target_token_id=target_token_id,
        hook_name=hook_name,
        base_scale_map=base_scale_map
    )

    clean_target_prob = clean_ctx.clean_target_prob
    clean_rank = clean_ctx.clean_rank

    target_probs = target_probs_tensor.tolist()
    ranks = ranks_tensor.tolist()

    results = []
    for prob, boosted_rank in zip(target_probs, ranks):
        delta = prob - clean_target_prob
        rank_improvement = clean_rank - boosted_rank
        is_safe = bool(boosted_rank <= clean_rank and delta >= -1e-6)
        results.append((is_safe, delta, rank_improvement))

    return results



def check_combination_safe(
    model, sae, prompt: str,
    mute_feature_ids: list[int], mute_strength: float,
    boost_feature_ids: list[int], boost_strength: float,
    target_token_id: int, top_k: int = 10, scale_map: dict | None = None
) -> dict:
    """
    Evaluates the joint effect of all muted and boosted features applied together.
    If `scale_map` (fid -> scale) is given it is used directly (per-feature strengths) and the
    mute/boost id + strength arguments are ignored.
    Detects any 'new blockers' (tokens ranked below target or absent in clean top-k,
    but ranked above target in the new list).
    """
    from src.hooks import make_joint_ablation_hook, make_signed_ablation_hook
    
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    model.reset_hooks()

    # 1. Clean baseline pass
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        
    clean_target_prob = clean_probs[target_token_id].item()
    clean_sorted_indices = torch.argsort(clean_probs, descending=True)
    clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1


    #Temporary Addition to test the testcase for "a"
    print("CLEAN TOP 10:")
    for i in range(10):
        idx = clean_sorted_indices[i].item()
        print(f"  Rank {i+1}: {model.to_string([idx])!r} — {clean_probs[idx].item()*100:.2f}%")
    
    # Get clean top_k token info
    clean_top_k_indices = clean_sorted_indices[:top_k]
    clean_top_k_probs = clean_probs[clean_top_k_indices]
    
    clean_top_k_tokens = [model.to_string([idx.item()]) for idx in clean_top_k_indices]
    clean_top_k_probs_list = [p.item() for p in clean_top_k_probs]
    clean_top_k_ids = [idx.item() for idx in clean_top_k_indices]
    
    # Build clean maps
    clean_id_to_rank = {idx: rank for rank, idx in enumerate(clean_top_k_ids, start=1)}
    clean_id_to_prob = {idx: prob for idx, prob in zip(clean_top_k_ids, clean_top_k_probs_list)}
    
    '''
    # Temporary Addition to test the testcase for "a"
    print("NEW TOP 10:")
    for i in range(10):
        idx = new_sorted_indices[i].item()
        print(f"  Rank {i+1}: {model.to_string([idx])!r} — {new_probs[idx].item()*100:.2f}%")
    '''
# 2. Apply ALL mute features AND all boost features TOGETHER, in one real pass
    from src.hooks import make_mute_and_boost_hook
    model.reset_hooks()
    if scale_map is not None:
        from src.hooks import make_scale_map_hook
        combined_fn = make_scale_map_hook(scale_map, sae)
    else:
        combined_fn = make_mute_and_boost_hook(mute_feature_ids, mute_strength, boost_feature_ids, boost_strength, sae)
    model.add_hook(hook_name, combined_fn)
        
    with torch.no_grad():
        new_logits = model(tokens)
        new_probs = F.softmax(new_logits[0, -1, :], dim=-1)
        
    model.reset_hooks()  # Reset hooks immediately after measurement
    
    new_target_prob = new_probs[target_token_id].item()
    new_sorted_indices = torch.argsort(new_probs, descending=True)
    new_rank = (new_sorted_indices == target_token_id).nonzero().item() + 1
    
    # Get new top_k token info
    new_top_k_indices = new_sorted_indices[:top_k]
    new_top_k_probs = new_probs[new_top_k_indices]
    new_top_k_tokens = [model.to_string([idx.item()]) for idx in new_top_k_indices]
    new_top_k_probs_list = [p.item() for p in new_top_k_probs]
    new_top_k_ids = [idx.item() for idx in new_top_k_indices]
    
    # 3. Identify new_blockers
    new_blockers = []
    # A blocker is any token ranked ABOVE the target in the new list (rank < new_rank)
    # AND must have been ranked BELOW the target in the clean baseline (clean_tok_rank_val > clean_rank).
    for i, tok_id in enumerate(new_top_k_ids):
        new_tok_rank = i + 1
        if new_tok_rank >= new_rank:
            continue  # Not ranked above target
            
        tok_str = new_top_k_tokens[i]
        new_tok_prob = new_top_k_probs_list[i]
        
        # Get actual clean rank and probability from clean baseline
        clean_tok_rank_val = (clean_sorted_indices == tok_id).nonzero().item() + 1
        clean_tok_prob_val = clean_probs[tok_id].item()
        
        # Determine if it was in the clean top-k
        in_clean_top_k = tok_id in clean_id_to_rank
        clean_rank_report = clean_tok_rank_val if in_clean_top_k else "absent"
        clean_prob_report = clean_tok_prob_val if in_clean_top_k else "absent"
        
        # Blocker condition: must have been ranked below the target in clean baseline
        is_blocker = bool(clean_tok_rank_val > clean_rank)
        
        if is_blocker:
            new_blockers.append({
                "token": tok_str,
                "clean_prob_or_absent": clean_prob_report,
                "new_prob": new_tok_prob,
                "clean_rank_or_absent": clean_rank_report,
                "new_rank": new_tok_rank
            })
            
    # 4. is_safe condition
    is_safe = False
    if new_rank == 1:
        is_safe = True
    elif len(new_blockers) == 0 and new_rank <= clean_rank:
        is_safe = True
        
    return {
        "is_safe": is_safe,
        "target_clean_rank": clean_rank,
        "target_new_rank": new_rank,
        "target_clean_prob": clean_target_prob,
        "target_new_prob": new_target_prob,
        "new_blockers": new_blockers
    }


def make_weighted_ablation_hook(feature_to_strength: dict[int, float], sae):
    """
    Creates a joint ablation hook where each feature can have a different ablation strength.
    """
    def hook_fn(resid, hook):
        if not feature_to_strength:
            return resid
            
        feature_acts = sae.encode(resid)
        baseline_reconstructed = sae.decode(feature_acts)
        
        modified_acts = feature_acts.clone()
        for fid, strength in feature_to_strength.items():
            modified_acts[..., fid] = modified_acts[..., fid] * (1.0 - strength)
            
        reconstructed = sae.decode(modified_acts)
        delta = reconstructed - baseline_reconstructed
        return resid + delta
        
    return hook_fn


def run_weighted_multi_competitor_reduction(
    model, sae, prompt: str, target_token_id: int, max_strength: float = 0.7, top_n_candidates: int = 20
) -> dict:
    """
    Identifies all tokens currently ranked above the target.
    Finds each competitor's best driving feature and mutes it in proportion to its threat.
    """
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    
    # 1. Clean baseline pass
    model.reset_hooks()
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
        
    clean_target_prob = clean_probs[target_token_id].item()
    clean_sorted_indices = torch.argsort(clean_probs, descending=True)
    clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1
    
    # 2. Identify all competitor tokens currently ranked above the target
    competitor_token_ids = clean_sorted_indices[:clean_rank - 1].tolist()
    
    competitors_data = []
    total_competitor_prob = 0.0
    
    for tok_id in competitor_token_ids:
        tok_str = model.to_string([tok_id])
        tok_prob = clean_probs[tok_id].item()
        total_competitor_prob += tok_prob
        
        # Get the top driver feature
        top_features = get_top_competitor_features(model, sae, prompt, tok_id, top_n=top_n_candidates)
        if top_features:
            best_fid, best_delta = top_features[0]
        else:
            best_fid, best_delta = None, 0.0
            
        competitors_data.append({
            "token": tok_str,
            "token_id": tok_id,
            "probability": tok_prob,
            "top_feature": best_fid,
            "feature_delta": best_delta
        })
        
    # 3. Calculate weights and strengths using additive accumulation (Experiment H10)
    feature_to_raw_strength = {}
    feature_to_contributors = {}
    for comp in competitors_data:
        fid = comp["top_feature"]
        if fid is None:
            continue
        weight = comp["probability"] / total_competitor_prob if total_competitor_prob > 0 else 0.0
        mute_strength = weight * max_strength
        if fid not in feature_to_raw_strength:
            feature_to_raw_strength[fid] = 0.0
            feature_to_contributors[fid] = []
        feature_to_raw_strength[fid] += mute_strength
        feature_to_contributors[fid].append({
            "token": comp["token"],
            "prob": comp["probability"],
            "weight": weight,
            "contrib_strength": mute_strength
        })
        
    feature_to_strength = {}
    print("\n=== EXPERIMENT H10 DIAGNOSTICS ===")
    for fid, raw_strength in feature_to_raw_strength.items():
        contributors = feature_to_contributors[fid]
        final_strength = min(raw_strength, max_strength)
        saturated = raw_strength > max_strength
        print(f"Feature {fid}\n")
        print("Contributors:")
        for c in contributors:
            print(f"    {c['token']} : {c['contrib_strength']:.6f} (Prob: {c['prob']*100:.4f}%, Weight: {c['weight']:.4f})")
        print(f"\nBefore clamp:\n{raw_strength:.6f}")
        print(f"\nAfter clamp:\n{final_strength:.6f}")
        print(f"\nSaturated:\n{saturated}\n")
        
        feature_to_strength[fid] = final_strength
        
    # 4. Apply all selected features simultaneously
    model.reset_hooks()
    weighted_hook = make_weighted_ablation_hook(feature_to_strength, sae)
    model.add_hook(hook_name, weighted_hook)
    
    with torch.no_grad():
        new_logits = model(tokens)
        new_probs = F.softmax(new_logits[0, -1, :], dim=-1)
        
    model.reset_hooks()
    
    new_target_prob = new_probs[target_token_id].item()
    new_sorted_indices = torch.argsort(new_probs, descending=True)
    new_rank = (new_sorted_indices == target_token_id).nonzero().item() + 1
    
    # 5. Safety checks (Blocker detection)
    top_k = 10
    clean_top_k_indices = clean_sorted_indices[:top_k]
    clean_top_k_ids = [idx.item() for idx in clean_top_k_indices]
    clean_id_to_rank = {idx: r for r, idx in enumerate(clean_top_k_ids, start=1)}
    clean_id_to_prob = {idx: clean_probs[idx].item() for idx in clean_top_k_ids}
    
    new_top_k_indices = new_sorted_indices[:top_k]
    new_top_k_probs_list = [new_probs[idx].item() for idx in new_top_k_indices]
    new_top_k_tokens = [model.to_string([idx.item()]) for idx in new_top_k_indices]
    new_top_k_ids = [idx.item() for idx in new_top_k_indices]
    
    new_blockers = []
    for i, tok_id in enumerate(new_top_k_ids):
        new_tok_rank = i + 1
        if new_tok_rank >= new_rank:
            continue
            
        tok_str = new_top_k_tokens[i]
        new_tok_prob = new_top_k_probs_list[i]
        
        clean_tok_rank_val = (clean_sorted_indices == tok_id).nonzero().item() + 1
        clean_tok_prob_val = clean_probs[tok_id].item()
        
        in_clean_top_k = tok_id in clean_id_to_rank
        clean_rank_report = clean_tok_rank_val if in_clean_top_k else "absent"
        clean_prob_report = clean_tok_prob_val if in_clean_top_k else "absent"
        
        # Blocker condition: must have been ranked below the target in clean baseline
        if clean_tok_rank_val > clean_rank:
            new_blockers.append({
                "token": tok_str,
                "clean_prob_or_absent": clean_prob_report,
                "new_prob": new_tok_prob,
                "clean_rank_or_absent": clean_rank_report,
                "new_rank": new_tok_rank
            })
            
    is_safe = False
    if new_rank == 1:
        is_safe = True
    elif len(new_blockers) == 0 and new_rank <= clean_rank:
        is_safe = True
        
    return {
        "is_safe": is_safe,
        "target_clean_rank": clean_rank,
        "target_new_rank": new_rank,
        "target_clean_prob": clean_target_prob,
        "target_new_prob": new_target_prob,
        "new_blockers": new_blockers,
        "competitors": competitors_data,
        "feature_to_strength": feature_to_strength
    }


def run_weighted_multi_feature_competitor_reduction(
    model, sae, prompt: str, target_token_id: int,
    max_strength: float = 0.7,
    top_n_candidates: int = 20,
    top_k_features_per_competitor: int = 3,
    feature_weighting_method: str = "delta_normalized",  # "equal" | "delta_normalized" | "softmax"
    softmax_temperature: float = 1.0,
) -> dict:
    """
    For each competitor above the target, take its top-K driving features,
    distribute that competitor's weight across those K features according to `feature_weighting_method`,
    then aggregate (sum) across all competitors per feature.
    """
    tokens = model.to_tokens(prompt)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)

    model.reset_hooks()
    with torch.no_grad():
        clean_logits = model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)

    clean_target_prob = clean_probs[target_token_id].item()
    clean_sorted_indices = torch.argsort(clean_probs, descending=True)
    clean_rank = (clean_sorted_indices == target_token_id).nonzero().item() + 1

    competitor_token_ids = clean_sorted_indices[:clean_rank - 1].tolist()
    total_competitor_prob = sum(clean_probs[t].item() for t in competitor_token_ids)

    competitors_data = []
    feature_to_raw_strength: dict[int, float] = {}
    feature_to_contributors: dict[int, list] = {}

    for tok_id in competitor_token_ids:
        tok_str = model.to_string([tok_id])
        tok_prob = clean_probs[tok_id].item()
        competitor_weight = tok_prob / total_competitor_prob if total_competitor_prob > 0 else 0.0

        # Get top-K driving features for THIS competitor
        ranked_features = get_top_competitor_features(model, sae, prompt, tok_id, top_n=top_n_candidates)
        top_k = ranked_features[:top_k_features_per_competitor]

        if not top_k:
            competitors_data.append({"token": tok_str, "probability": tok_prob, "features": []})
            continue

        deltas = [abs(delta) for _, delta in top_k]  # magnitude of causal effect
        if feature_weighting_method == "equal":
            feature_weights = [1.0 / len(top_k)] * len(top_k)
        elif feature_weighting_method == "delta_normalized":
            total_delta = sum(deltas) or 1e-9
            feature_weights = [d / total_delta for d in deltas]
        elif feature_weighting_method == "softmax":
            import math
            scaled = [d / softmax_temperature for d in deltas]
            m = max(scaled)
            exps = [math.exp(s - m) for s in scaled]  # numerically stable
            total_exp = sum(exps)
            feature_weights = [e / total_exp for e in exps]
        else:
            raise ValueError(f"Unknown weighting method: {feature_weighting_method}")

        feature_entries = []
        for (fid, delta), fw in zip(top_k, feature_weights):
            joint_weight = competitor_weight * fw
            mute_strength = joint_weight * max_strength
            
            feature_to_raw_strength[fid] = feature_to_raw_strength.get(fid, 0.0) + mute_strength
            
            if fid not in feature_to_contributors:
                feature_to_contributors[fid] = []
            
            feature_to_contributors[fid].append({
                "token": tok_str,
                "prob": tok_prob,
                "competitor_weight": competitor_weight,
                "feature_weight": fw,
                "joint_weight": joint_weight,
                "contrib_strength": mute_strength
            })
            
            feature_entries.append({
                "feature_id": fid,
                "delta": delta,
                "within_competitor_weight": fw,
                "joint_weight": joint_weight
            })

        competitors_data.append({
            "token": tok_str,
            "probability": tok_prob,
            "features": feature_entries
        })

    # Clamp each feature's final accumulated strength to max_strength
    feature_to_strength = {}
    saturated = {}
    
    print("\n=== EXPERIMENT H11 DIAGNOSTICS ===")
    for fid, raw_strength in feature_to_raw_strength.items():
        contributors = feature_to_contributors[fid]
        final_strength = min(raw_strength, max_strength)
        saturated_flag = raw_strength > max_strength
        
        print(f"Feature {fid}\n")
        print("Contributors:")
        for c in contributors:
            print(f"    {c['token']} : {c['contrib_strength']:.6f} (Prob: {c['prob']*100:.4f}%, CompWeight: {c['competitor_weight']:.4f}, FeatWeight: {c['feature_weight']:.4f})")
        print(f"\nBefore clamp:\n{raw_strength:.6f}")
        print(f"\nAfter clamp:\n{final_strength:.6f}")
        print(f"\nSaturated:\n{saturated_flag}\n")
        
        feature_to_strength[fid] = final_strength
        saturated[fid] = {
            "saturated": saturated_flag,
            "before": raw_strength,
            "after": final_strength
        }

    # Apply weighted hook
    model.reset_hooks()
    weighted_hook = make_weighted_ablation_hook(feature_to_strength, sae)
    model.add_hook(hook_name, weighted_hook)

    with torch.no_grad():
        new_logits = model(tokens)
        new_probs = F.softmax(new_logits[0, -1, :], dim=-1)
    model.reset_hooks()

    new_target_prob = new_probs[target_token_id].item()
    new_sorted_indices = torch.argsort(new_probs, descending=True)
    new_rank = (new_sorted_indices == target_token_id).nonzero().item() + 1

    # Safety checks (Blocker detection)
    top_k_blocker = 10
    clean_top_k_indices = clean_sorted_indices[:top_k_blocker]
    clean_top_k_ids = [idx.item() for idx in clean_top_k_indices]
    clean_id_to_rank = {idx: r for r, idx in enumerate(clean_top_k_ids, start=1)}
    
    new_top_k_indices = new_sorted_indices[:top_k_blocker]
    new_top_k_probs_list = [new_probs[idx].item() for idx in new_top_k_indices]
    new_top_k_tokens = [model.to_string([idx.item()]) for idx in new_top_k_indices]
    new_top_k_ids = [idx.item() for idx in new_top_k_indices]
    
    new_blockers = []
    for i, tok_id in enumerate(new_top_k_ids):
        new_tok_rank = i + 1
        if new_tok_rank >= new_rank:
            continue
            
        tok_str = new_top_k_tokens[i]
        new_tok_prob = new_top_k_probs_list[i]
        
        clean_tok_rank_val = (clean_sorted_indices == tok_id).nonzero().item() + 1
        clean_tok_prob_val = clean_probs[tok_id].item()
        
        in_clean_top_k = tok_id in clean_id_to_rank
        clean_rank_report = clean_tok_rank_val if in_clean_top_k else "absent"
        clean_prob_report = clean_tok_prob_val if in_clean_top_k else "absent"
        
        if clean_tok_rank_val > clean_rank:
            new_blockers.append({
                "token": tok_str,
                "clean_prob_or_absent": clean_prob_report,
                "new_prob": new_tok_prob,
                "clean_rank_or_absent": clean_rank_report,
                "new_rank": new_tok_rank
            })
            
    is_safe = False
    if new_rank == 1:
        is_safe = True
    elif len(new_blockers) == 0 and new_rank <= clean_rank:
        is_safe = True

    return {
        "is_safe": is_safe,
        "new_blockers": new_blockers,
        "target_clean_rank": clean_rank,
        "target_clean_prob": clean_target_prob,
        "target_new_rank": new_rank,
        "target_new_prob": new_target_prob,
        "competitors": competitors_data,
        "feature_to_strength": feature_to_strength,
        "saturated": saturated,
        "feature_to_contributors": feature_to_contributors
    }





