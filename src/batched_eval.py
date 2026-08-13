"""
Batched candidate evaluation primitives. Turns N sequential single-candidate
forward calls into 1 (or a few, if chunked) batched forward calls, without
changing what is scientifically being evaluated.
"""

import torch
from src.hooks import make_per_row_scale_hook

MAX_EVAL_BATCH = 32


def batched_ablation_probs(
    model, sae, tokens, feature_ids: list[int], scale: float,
    token_ids_of_interest: list[int], hook_name: str, max_eval_batch: int = MAX_EVAL_BATCH
) -> torch.Tensor:
    """
    Evaluates `len(feature_ids)` single-feature ablations of `tokens` in one or
    more batched forward calls, chunked to at most max_eval_batch rows per call.

    Returns: torch.Tensor of shape [len(feature_ids), len(token_ids_of_interest)],
    on-device, giving the probability of each token of interest at the final
    sequence position, under each single-feature ablation.

    No .item() calls inside the loop. Concatenate results across chunks and
    return a single tensor; callers pull out .tolist()/.item() once, if needed,
    at the call site.
    """
    if not feature_ids:
        return torch.empty((0, len(token_ids_of_interest)), device=tokens.device)

    chunk_results = []
    num_features = len(feature_ids)

    for start_idx in range(0, num_features, max_eval_batch):
        chunk_fids = feature_ids[start_idx : start_idx + max_eval_batch]
        chunk_size = len(chunk_fids)

        batch_tokens = tokens.repeat(chunk_size, 1)
        hook_fn = make_per_row_scale_hook(chunk_fids, sae, scale)

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            last_token_logits = ablated_logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)
            chunk_probs = probs[:, token_ids_of_interest]
            chunk_results.append(chunk_probs)

    return torch.cat(chunk_results, dim=0)


def batched_ablation_probs_and_ranks(
    model, sae, tokens, feature_ids: list[int], scale: float,
    target_token_id: int, hook_name: str, max_eval_batch: int = MAX_EVAL_BATCH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Like batched_ablation_probs, but also returns the target token's rank
    under each ablation, computed on-device without a full [N, vocab] argsort:

        target_probs = <as above, restricted to target_token_id>
        ranks = (all_probs > target_probs.unsqueeze(-1)).sum(dim=-1) + 1

    Returns: (target_probs [N], target_ranks [N]) as tensors.
    """
    if not feature_ids:
        return (
            torch.empty((0,), device=tokens.device),
            torch.empty((0,), dtype=torch.long, device=tokens.device),
        )

    prob_chunks = []
    rank_chunks = []
    num_features = len(feature_ids)

    for start_idx in range(0, num_features, max_eval_batch):
        chunk_fids = feature_ids[start_idx : start_idx + max_eval_batch]
        chunk_size = len(chunk_fids)

        batch_tokens = tokens.repeat(chunk_size, 1)
        hook_fn = make_per_row_scale_hook(chunk_fids, sae, scale)

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            last_token_logits = ablated_logits[:, -1, :]
            all_probs = torch.softmax(last_token_logits, dim=-1)
            target_probs = all_probs[:, target_token_id]

            ranks = (all_probs > target_probs.unsqueeze(-1)).sum(dim=-1) + 1

            prob_chunks.append(target_probs)
            rank_chunks.append(ranks)

    return torch.cat(prob_chunks, dim=0), torch.cat(rank_chunks, dim=0)
