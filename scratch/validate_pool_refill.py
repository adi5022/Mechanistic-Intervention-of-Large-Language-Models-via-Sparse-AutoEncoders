import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sae_utils import load_model_and_sae
from src.editing import (
    get_target_token_id, get_top_competitor_features, get_top_target_features,
    check_target_safe, check_boost_safe, check_target_safe_batch, check_boost_safe_batch,
    build_clean_context, build_steered_context, HOOK_NAME,
)
from src.hooks import make_mute_and_boost_hook, build_scale_map

PROMPT = "The leading cause of death in men is"
TARGET = " violence"
LAYER = 8
SM, SB = 0.3, 0.5


def main():
    model, sae = load_model_and_sae(layer=LAYER)
    hook_name = getattr(sae.cfg, "hook_name", HOOK_NAME)
    tid = get_target_token_id(model, TARGET)
    clean = build_clean_context(model, sae, PROMPT, tid)
    top1 = int(torch.argmax(clean.clean_probs))
    ok = True

    comp = [f for f, _ in get_top_competitor_features(model, sae, PROMPT, top1, top_n=12, clean_ctx=clean, use_batched=True)]
    tgt = [f for f, _ in get_top_target_features(model, sae, PROMPT, tid, top_n=12, clean_ctx=clean, use_batched=True)]
    mutes, boosts = comp[:3], [f for f in tgt if f not in comp[:3]][:3]
    base = build_scale_map(mutes, SM, boosts, SB)

    # 1. steered context == the combined-hook pass used by the sweep
    steered = build_steered_context(model, sae, clean, base, tid)
    model.reset_hooks()
    model.add_hook(hook_name, make_mute_and_boost_hook(mutes, SM, boosts, SB, sae))
    with torch.no_grad():
        p = F.softmax(model(clean.tokens)[0, -1, :], dim=-1)
    model.reset_hooks()
    d = (p - steered.clean_probs).abs().max().item()
    print(f"[1] steered ctx vs combined hook: max|dp|={d:.2e}  rank={steered.clean_rank}")
    ok &= d < 1e-6

    # 2. exclude_ids really removes applied features from the candidate lists
    top1_s = int(torch.argmax(steered.clean_probs))
    excl = set(mutes) | set(boosts)
    cs_b = get_top_competitor_features(model, sae, PROMPT, top1_s, top_n=12, clean_ctx=steered, use_batched=True,
                                       exclude_ids=excl, base_scale_map=base)
    cs_s = get_top_competitor_features(model, sae, PROMPT, top1_s, top_n=12, clean_ctx=steered, use_batched=False,
                                       exclude_ids=excl, base_scale_map=base)
    ids_b, ids_s = [f for f, _ in cs_b], [f for f, _ in cs_s]
    print(f"[2] competitor ranking steered: batched==sequential order: {ids_b == ids_s}; leaked applied ids: {excl & set(ids_b)}")
    ok &= (ids_b == ids_s) and not (excl & set(ids_b))
    dmax = max(abs(a[1] - b[1]) for a, b in zip(sorted(cs_b), sorted(cs_s)))
    print(f"    max delta diff: {dmax:.2e}")
    ok &= dmax < 1e-5

    # 3. steered safety filters: batched vs sequential
    cand = ids_b
    mb = check_target_safe_batch(model, sae, steered, cand, tid, strength=SM, base_scale_map=base)
    bb = check_boost_safe_batch(model, sae, steered, cand, tid, strength=SB, base_scale_map=base)
    agree = 0
    for f, (sm_ok, sm_d), (sb_ok, sb_d, _) in zip(cand, mb, bb):
        s_ok, s_d = check_target_safe(model, sae, PROMPT, f, tid, strength=SM,
                                      clean_target_prob=steered.clean_target_prob, clean_rank=steered.clean_rank, base_scale_map=base)
        b_ok, b_d, _ = check_boost_safe(model, sae, PROMPT, f, tid, strength=SB,
                                        clean_target_prob=steered.clean_target_prob, clean_rank=steered.clean_rank, base_scale_map=base)
        agree += int(s_ok == sm_ok) + int(b_ok == sb_ok)
        ok &= abs(s_d - sm_d) < 1e-5 and abs(b_d - sb_d) < 1e-5
    print(f"[3] steered safety agreement (batched vs sequential): {agree}/{2 * len(cand)}")
    ok &= agree == 2 * len(cand)

    # 4. empty base is bit-identical to the original (no-base) path
    a = check_target_safe_batch(model, sae, clean, cand, tid, strength=SM)
    b = check_target_safe_batch(model, sae, clean, cand, tid, strength=SM, base_scale_map={})
    print(f"[4] empty base identical to original path: {a == b}")
    ok &= a == b

    # 5. candidate source = all prompt positions
    from src.editing import get_top_active_features
    last_ids = {f for f, _ in get_top_active_features(model, sae, PROMPT, top_n=500, clean_ctx=clean)}
    all_ids = {f for f, _ in get_top_active_features(model, sae, PROMPT, top_n=5000, clean_ctx=clean, positions="all")}
    print(f"[5] last-token features: {len(last_ids)}; all-position features: {len(all_ids)}; last subset of all: {last_ids <= all_ids}")
    ok &= last_ids <= all_ids and len(all_ids) > len(last_ids)

    ca_b = get_top_competitor_features(model, sae, PROMPT, top1, top_n=40, clean_ctx=clean, use_batched=True, positions="all")
    ca_s = get_top_competitor_features(model, sae, PROMPT, top1, top_n=40, clean_ctx=clean, use_batched=False, positions="all")
    same = [f for f, _ in ca_b] == [f for f, _ in ca_s]
    dmax = max(abs(a[1] - b[1]) for a, b in zip(sorted(ca_b), sorted(ca_s)))
    print(f"[6] all-position competitor ranking batched==sequential: {same}; max delta diff {dmax:.2e}; candidates {len(ca_b)}")
    ok &= same and dmax < 1e-5

    ts_b = get_top_target_features(model, sae, PROMPT, tid, top_n=40, clean_ctx=steered, use_batched=True, positions="all", exclude_ids=excl, base_scale_map=base)
    ts_s = get_top_target_features(model, sae, PROMPT, tid, top_n=40, clean_ctx=steered, use_batched=False, positions="all", exclude_ids=excl, base_scale_map=base)
    same2 = [f for f, _ in ts_b] == [f for f, _ in ts_s]
    print(f"[7] steered all-position target ranking batched==sequential: {same2}; leaked applied ids: {excl & {f for f, _ in ts_b}}")
    ok &= same2 and not (excl & {f for f, _ in ts_b})

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
