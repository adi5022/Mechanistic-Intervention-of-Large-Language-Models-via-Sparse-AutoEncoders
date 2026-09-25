import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt
import json
import time
from datetime import datetime

from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.editing import (
    get_target_token_id,
    get_top_competitor_features,
    get_top_target_features,
    check_target_safe,
    check_boost_safe,
    check_target_safe_batch,
    check_boost_safe_batch,
    check_combination_safe,
    build_clean_context,
    build_steered_context,
    run_weighted_multi_competitor_reduction,
    run_weighted_multi_feature_competitor_reduction,
    make_weighted_ablation_hook,
)

from src.hooks import (
    make_ablation_hook,
    make_joint_ablation_hook,
    make_signed_ablation_hook,
    make_mute_and_boost_hook,
    build_scale_map,
)
from src.batched_eval import MAX_EVAL_BATCH
from src.monosemanticity import (
    find_max_activating_examples,
    find_max_activating_neuron_examples,
    scan_dual_activations,
    score_feature_interpretability,
    compute_sparsity_stats,
    compute_feature_similarity,
    find_most_similar_features,
    get_default_corpus,
    validate_feature_id,
    validate_neuron_index,
    get_curated_feature_registry,
    get_sae_status_summary,
    generate_synthetic_corpus,
)

from src.explain import (
    get_xai_guidance_card,
    generate_polysemanticity_comparison_xai,
    generate_sparsity_xai,
    generate_decoder_similarity_xai,
    generate_mechanistic_explanation,
    get_empty_state_guidance,
)

def render_xai_guidance_card(card: dict):
    """Renders a structured 6-part Educational XAI Guidance Card in the UI."""
    with st.expander("🎓 **XAI Guidance Card — Educational Deep Dive (6-Part Framework)**", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**🔍 Trigger Context (What triggered this?):**\n{card['trigger']}")
            st.markdown(f"**❓ What is this?:**\n{card['what']}")
            st.markdown(f"**🎯 Why does it matter at this stage?:**\n{card['why']}")
        with col_b:
            st.markdown(f"**⚡ What is happening internally?:**\n{card['internal']}")
            st.markdown(f"**📊 How to interpret results?:**\n{card['interpret']}")
            st.markdown(f"**🛠️ Contribution to Intervention Workflow:**\n{card['workflow']}")

def render_empty_state_card(state_dict: dict):
    """Renders a structured, informative empty state card when data or metadata is unavailable."""
    with st.container(border=True):
        st.markdown(f"### {state_dict['title']}")
        st.warning(f"**Why is this unavailable?:** {state_dict['why']}")
        st.info(f"**Is this expected?:** {state_dict['expected']}\n\n👉 **Recommended Next Steps:** {state_dict['next_steps']}")

def render_rank_progression_chart(rank_progression: list[dict], best_step: int | None = None, height: int = 320, refill_steps: list[int] | None = None):
    """
    Renders a labeled, publication-quality line chart of the target token's rank across
    sweep steps (Step 0 = clean baseline). The y-axis is inverted so that improvement
    (a lower, better rank) reads as an upward-moving line, with a dashed reference line
    at Rank #1 and the best-so-far step highlighted as a distinct marker.

    rank_progression: list of {"Step": int, "Label": str, "Target Rank": int, "Target Prob (%)": float}
    """
    df = pd.DataFrame(rank_progression)
    max_rank = max(int(df["Target Rank"].max()), 2)
    y_domain = [max_rank * 1.08, 0.5]

    base = alt.Chart(df)

    line = base.mark_line(
        interpolate="monotone",
        strokeWidth=2.5,
        color="#4C78A8",
    ).encode(
        x=alt.X("Step:Q", title="Sweep Step (0 = Baseline)", axis=alt.Axis(tickMinStep=1, format="d", grid=True)),
        y=alt.Y("Target Rank:Q", title="Target Rank (lower is better)",
                scale=alt.Scale(domain=y_domain), axis=alt.Axis(grid=True)),
    )

    points = base.mark_circle(size=90, color="#4C78A8", opacity=0.9).encode(
        x="Step:Q",
        y="Target Rank:Q",
        tooltip=[
            alt.Tooltip("Label:N", title="Step"),
            alt.Tooltip("Target Rank:Q", title="Target Rank"),
            alt.Tooltip("Target Prob (%):Q", title="Target Probability", format=".2f"),
        ],
    )

    target_ref_line = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(
        strokeDash=[5, 4], color="#59A14F", strokeWidth=1.5
    ).encode(y=alt.Y("y:Q"))

    layers = [line, points, target_ref_line]

    if refill_steps:
        refill_rules = alt.Chart(pd.DataFrame({"Step": refill_steps, "Event": "Pool refill"})).mark_rule(
            strokeDash=[2, 3], color="#F58518", strokeWidth=1.5
        ).encode(x="Step:Q", tooltip=[alt.Tooltip("Event:N"), alt.Tooltip("Step:Q")])
        layers.append(refill_rules)

    if best_step is not None:
        best_row = df[df["Step"] == best_step]
        if not best_row.empty:
            best_marker = alt.Chart(best_row).mark_point(
                shape="diamond", size=260, color="#E45756", filled=True, stroke="white", strokeWidth=1.5
            ).encode(
                x="Step:Q", y="Target Rank:Q",
                tooltip=[
                    alt.Tooltip("Label:N", title="Best Step"),
                    alt.Tooltip("Target Rank:Q", title="Best Rank"),
                    alt.Tooltip("Target Prob (%):Q", title="Probability", format=".2f"),
                ],
            )
            layers.append(best_marker)

    chart = alt.layer(*layers).properties(height=height).configure_axis(
        labelFontSize=12, titleFontSize=13, titleFontWeight="normal", labelColor="#666", titleColor="#333"
    ).configure_view(strokeWidth=0)

    st.altair_chart(chart, use_container_width=True)

import requests

@st.cache_data(ttl=3600)
def get_neuronpedia_explanation(feature_id: int, layer: int) -> str:
    if feature_id is None:
        return "None"
    try:
        url = f"https://neuronpedia.org/api/feature/gpt2-small/{layer}-res-jb/{feature_id}"
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            data = r.json()
            explanations = data.get("explanations", [])
            if explanations:
                return explanations[0].get("description", "No description available")
    except Exception:
        pass
    return "Explanation unavailable"

def make_feature_hover_link(feature_id: int, layer: int) -> str:
    if feature_id is None:
        return "None"
    explanation = get_neuronpedia_explanation(feature_id, layer)
    url = f"https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_id}"
    return f'<a href="{url}" target="_blank" title="Neuronpedia: {explanation}">Feature {feature_id}</a>'


st.set_page_config(page_title="FeatureScalpel — Experimentation Bench", page_icon="🧪", layout="wide")

st.title("FeatureScalpel — Experimentation & Benchmarking Bench")

# Initialize session history in st.session_state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Every run is also written to disk so a browser refresh or server restart cannot lose it.
import os as _os
_HISTORY_DIR = _os.path.join("outputs", "session_history")
if "history_file" not in st.session_state:
    st.session_state["history_file"] = _os.path.join(_HISTORY_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


def persist_history():
    """Rewrite this session's history file with everything captured so far."""
    try:
        _os.makedirs(_HISTORY_DIR, exist_ok=True)
        with open(st.session_state["history_file"], "w", encoding="utf-8") as f:
            json.dump(st.session_state["history"], f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Could not auto-save session history to disk: {e}")

# Cached base model (loaded ONCE per app session)
@st.cache_resource
def get_cached_base_model():
    return load_base_model()

# Cached layer SAE (loaded per layer, tiny overhead)
@st.cache_resource
def get_cached_sae(layer: int):
    return load_sae_for_layer(layer=layer)

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

default_groq_key = os.environ.get("GROQ_API_KEY", "")
if not default_groq_key:
    try:
        if hasattr(st, "secrets") and "groq" in st.secrets:
            default_groq_key = st.secrets["groq"].get("api_key", "")
    except Exception:
        pass

device = get_default_device()
st.sidebar.markdown(f"**⚡ Compute Device:** `{device.upper()}`")

layer = st.sidebar.selectbox("Select Model Layer", options=list(range(12)), index=8)

with st.spinner("Loading Base Model (GPT-2 Small)..."):
    model = get_cached_base_model()

with st.spinner(f"Loading SAE for Layer {layer}..."):
    sae = get_cached_sae(layer)

hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Explainable AI Layer")
enable_xai = st.sidebar.checkbox("Enable AI Explanations (Groq)", value=bool(default_groq_key))
groq_key_input = default_groq_key

# Tabs setup
tab4, tab6, tab7, tab11, tab12 = st.tabs([
    "🔄 Hybrid mute and boost",
    "📚 Monosemanticity Analysis",
    "📊 Session history and benchmarks",
    "⏱️ Sequential vs Batched Proof",
    "🧮 Batch: Last vs All Tokens"
])



# --- TAB 4: Hybrid Mute & Boost (Dual Intervention) ---
with tab4:
    st.header("Hybrid Mute & Boost (Dual Intervention)")
    st.markdown("Simultaneously **mutes competitor features** and **amplifies target features** in a single joint forward pass.")
    
    col1, col2 = st.columns(2)
    with col1:
        prompt_4 = st.text_input("Prompt", "Seiyu Group's headquarters are in", key="t4_prompt")
        target_4 = st.text_input("Target Completion", "Tokyo", key="t4_target")
        top_n_4 = st.number_input(
            "Top N Candidate Features (maximum)", value=30, min_value=1, step=1, key="t4_topn",
            help="A ceiling, not a target. It is automatically capped at the number of features that are actually active for the chosen candidate source, so asking for more than exist just uses all of them (the pool table shows requested vs available)."
        )
        candidate_source_4 = st.radio(
            "Candidate source", ["All prompt positions", "Last token only"], horizontal=True, key="t4_cand_source",
            help="Last token only = features active on the final prompt token (the original behaviour). All prompt positions = also features active on earlier words (BOS excluded), e.g. 'sky' in 'The sky is'. The edit already scales a feature at every position; this only widens which features are considered."
        )
        cand_positions_4 = "all" if candidate_source_4.startswith("All") else "last"
    with col2:
        cumulative_sweep_4 = st.checkbox(
            "🔁 Cumulative Sweep (pile on one feature at a time until Target reaches Rank 1)", value=False, key="t4_cumulative",
            help="Ignores the Mute/Boost Batch Sizes below. Instead runs Mute 1/Boost 1, then Mute 2/Boost 2, then Mute 3/Boost 3, and so on — one feature added to each side per step — stopping the moment the target reaches Rank #1 (or once the candidate pool runs out)."
        )
        refill_enabled_4 = st.checkbox(
            "♻️ Pool Refill (adaptive rounds — only with Cumulative Sweep)", value=True, key="t4_refill", disabled=not cumulative_sweep_4,
            help="When the safe-candidate pool is used up and the target is not yet Rank #1, keep every applied feature, run a fresh forward pass to get a new steered baseline, then re-rank and re-filter candidates (including ones rejected earlier) against that state, and continue. Repeats until Rank #1 or no safe candidate remains."
        )
        max_rounds_4 = st.number_input(
            "Max Refill Rounds (0 = unlimited)", value=0, min_value=0, step=1, key="t4_max_rounds",
            disabled=not (cumulative_sweep_4 and refill_enabled_4)
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            strength_mute_4 = st.number_input("Mute Strength", value=0.3, step=0.1, key="t4_m_strength")
            mute_sizes_str_4 = st.text_input("Mute Batch Sizes", "1, 3, 5", key="t4_m_bs", disabled=cumulative_sweep_4)
        with col_m2:
            strength_boost_4 = st.number_input("Boost Strength", value=0.5, step=0.1, key="t4_b_strength")
            boost_sizes_str_4 = st.text_input("Boost Batch Sizes", "1, 3, 5", key="t4_b_bs", disabled=cumulative_sweep_4)
        use_safety_4 = st.checkbox("Enable Safety Filter (Target Protection)", value=True, key="t4_safety")
        stop_on_rank1_4 = st.checkbox("Stop sweep once Target reaches Rank 1", value=True, key="t4_stop_rank1")
        use_batched_4 = st.checkbox(
            "⚡ Use GPU-Batched Optimization", value=True, key="t4_use_batched",
            help="ON = candidate ranking & safety filtering run as batched GPU calls (fast, current default). OFF = the original one-candidate-at-a-time method (slow, kept for comparison)."
        )

    if st.button("Run Hybrid Mute & Boost Test", key="btn_t4"):
        try:
            mute_sizes_4 = [int(x.strip()) for x in mute_sizes_str_4.split(",") if x.strip()]
        except ValueError:
            mute_sizes_4 = [1, 3, 5]
            st.warning("Invalid mute batch sizes; using default [1, 3, 5].")

        try:
            boost_sizes_4 = [int(x.strip()) for x in boost_sizes_str_4.split(",") if x.strip()]
        except ValueError:
            boost_sizes_4 = [1, 3, 5]
            st.warning("Invalid boost batch sizes; using default [1, 3, 5].")
            
        prompt_4 = prompt_4.strip()
        target_4 = target_4.strip()
        target_str = " " + target_4
        target_token_id = get_target_token_id(model, target_str)
        _n_tgt_tokens = int(model.to_tokens(target_str, prepend_bos=False).numel())
        if _n_tgt_tokens > 1:
            st.warning(
                f"⚠️ '{target_str}' is {_n_tgt_tokens} GPT-2 tokens. Only its LAST piece ('{model.to_string([target_token_id])}') is being scored, "
                f"so the rank/probability below is NOT for the whole word. Use a single-token target for a valid result."
            )

        # --- Stopwatch start ---
        if device == "cuda":
            torch.cuda.synchronize()
        t4_start = time.perf_counter()

        # Clean baseline pass
        model.reset_hooks()
        clean_ctx = build_clean_context(model, sae, prompt_4, target_token_id)
        tokens = clean_ctx.tokens
        probs = clean_ctx.clean_probs
        current_top1_id = torch.argmax(probs).item()
        current_top1_str = model.to_string([current_top1_id])
        baseline_target_prob = clean_ctx.clean_target_prob
        t4_baseline_done = time.perf_counter()

        tracker_box = st.container(border=True)
        with tracker_box:
            st.markdown("**📡 Live progress — all rounds so far** (updates as the run goes; details continue below)")
            tr_status = st.empty()
            tr_table = st.empty()
            tr_chart = st.empty()
        st.write(f"**Baseline Top-1:** `{current_top1_str}` | **Target '{target_str}' Prob:** `{baseline_target_prob*100:.2f}%`")

        # ==================================================================
        # POOL-REFILL SWEEP
        # Round 0 builds candidate pools against the CLEAN model. Every later round
        # (only when Cumulative Sweep + Pool Refill are on) re-scores and re-filters
        # candidates against the STEERED model — the clean model plus every mute/boost
        # applied so far — so the pool is never a stale snapshot of the original prompt.
        # ==================================================================
        refill_on = bool(cumulative_sweep_4 and refill_enabled_4)
        sm4, sb4 = strength_mute_4, strength_boost_4

        def _chunks(n):
            return (-(-n // MAX_EVAL_BATCH) if use_batched_4 else n) if n else 0

        def build_pools(ctx, base_map, exclude_ids):
            """Rank + safety-filter candidates against `ctx` (clean or steered baseline)."""
            blocker_id = int(torch.argmax(ctx.clean_probs).item())
            with torch.no_grad():
                acts_all = sae.encode(ctx.resid_all[0])          # [seq, n_features]
            tok_strs = model.to_str_tokens(prompt_4)
            per_token = [(tok_strs[i], int((acts_all[i] > 0).sum().item())) for i in range(1, acts_all.shape[0])]
            if cand_positions_4 == "all":
                active_mask = (acts_all[1:] > 0).any(dim=0)
            else:
                active_mask = acts_all[-1] > 0
            active_ids = set(torch.nonzero(active_mask).flatten().tolist())
            available = len(active_ids - set(exclude_ids))
            eff_top_n = min(top_n_4, available)

            comp_feats = get_top_competitor_features(
                model, sae, prompt_4, blocker_id, top_n=eff_top_n, clean_ctx=ctx,
                use_batched=use_batched_4, exclude_ids=exclude_ids, base_scale_map=base_map, positions=cand_positions_4
            ) if eff_top_n > 0 else []
            tgt_feats = get_top_target_features(
                model, sae, prompt_4, target_token_id, top_n=eff_top_n, clean_ctx=ctx,
                use_batched=use_batched_4, exclude_ids=exclude_ids, base_scale_map=base_map, positions=cand_positions_4
            ) if eff_top_n > 0 else []
            comp_fids = [f for f, _ in comp_feats]
            tgt_fids = [f for f, _ in tgt_feats]
            comp_effect = dict(comp_feats)
            tgt_effect = dict(tgt_feats)
            n_passes = _chunks(len(comp_fids)) + _chunks(len(tgt_fids))

            comp_ok, comp_rej, comp_delta = [], [], {}
            tgt_ok, tgt_rej, tgt_delta = [], [], {}
            if use_safety_4:
                if use_batched_4:
                    res = check_target_safe_batch(model, sae, ctx, comp_fids, target_token_id, strength=sm4, base_scale_map=base_map)
                    for fid, (ok, d) in zip(comp_fids, res):
                        (comp_ok if ok else comp_rej).append(fid if ok else (fid, d))
                        comp_delta[fid] = d
                    res = check_boost_safe_batch(model, sae, ctx, tgt_fids, target_token_id, strength=sb4, base_scale_map=base_map)
                    for fid, (ok, d, _r) in zip(tgt_fids, res):
                        (tgt_ok if ok else tgt_rej).append(fid if ok else (fid, d))
                        tgt_delta[fid] = d
                else:
                    for fid in comp_fids:
                        ok, d = check_target_safe(model, sae, prompt_4, fid, target_token_id, strength=sm4,
                                                  clean_target_prob=ctx.clean_target_prob, clean_rank=ctx.clean_rank,
                                                  base_scale_map=base_map)
                        (comp_ok if ok else comp_rej).append(fid if ok else (fid, d))
                        comp_delta[fid] = d
                    for fid in tgt_fids:
                        ok, d, _r = check_boost_safe(model, sae, prompt_4, fid, target_token_id, strength=sb4,
                                                     clean_target_prob=ctx.clean_target_prob, clean_rank=ctx.clean_rank,
                                                     base_scale_map=base_map)
                        (tgt_ok if ok else tgt_rej).append(fid if ok else (fid, d))
                        tgt_delta[fid] = d
                n_passes += _chunks(len(comp_fids)) + _chunks(len(tgt_fids))
            else:
                comp_ok, tgt_ok = list(comp_fids), list(tgt_fids)

            # Overlap resolution: a feature that passed BOTH safety tests would be muted (x(1-sm)) and
            # boosted (x(1+sb)) at once, which nearly cancels out. Give each such feature to ONE side.
            # With the safety filter on, compare what each action does to the TARGET's probability
            # (same units for both) and keep the side that helps the target more. With it off there
            # are no such deltas, so keep the side where the feature ranks higher in its own list.
            overlap = []
            both = [f for f in comp_ok if f in set(tgt_ok)]
            for fid in both:
                if use_safety_4:
                    md, bd = comp_delta[fid], tgt_delta[fid]
                    keep = "boost" if bd > md else "mute"
                    detail = {"Target Δprob if muted (%)": md * 100, "Target Δprob if boosted (%)": bd * 100}
                else:
                    keep = "boost" if tgt_fids.index(fid) < comp_fids.index(fid) else "mute"
                    detail = {"Position in mute list": comp_fids.index(fid) + 1, "Position in boost list": tgt_fids.index(fid) + 1}
                overlap.append({"Feature": fid, **detail, "Kept on": keep})
            drop_from_boost = {o["Feature"] for o in overlap if o["Kept on"] == "mute"}
            drop_from_mute = {o["Feature"] for o in overlap if o["Kept on"] == "boost"}
            comp_ok = [f for f in comp_ok if f not in drop_from_mute]
            tgt_ok = [f for f in tgt_ok if f not in drop_from_boost]

            return {
                "overlap": overlap,
                "blocker_id": blocker_id, "available": available, "n_active": len(active_ids),
                "eff_top_n": eff_top_n, "per_token": per_token,
                "comp_fids": comp_fids, "tgt_fids": tgt_fids,
                "comp_ok": comp_ok, "tgt_ok": tgt_ok, "comp_rej": comp_rej, "tgt_rej": tgt_rej,
                "comp_delta": comp_delta, "tgt_delta": tgt_delta,
                "comp_effect": comp_effect, "tgt_effect": tgt_effect, "passes": n_passes,
            }

        def _fmt_rank_change(new, old):
            d = old - new
            return f"{d:+d} ranks ({'better' if d > 0 else 'worse' if d < 0 else 'no change'})"

        hybrid_details = []
        best_so_far = {
            "rank": clean_ctx.clean_rank, "prob": baseline_target_prob, "top1": current_top1_str,
            "mute_size": 0, "boost_size": 0, "mute_features": [], "boost_features": [], "step": 0,
        }
        rank_progression = [{
            "Step": 0, "Label": "Baseline", "Round": 0,
            "Target Rank": clean_ctx.clean_rank, "Target Prob (%)": baseline_target_prob * 100,
        }]
        refill_markers = []          # steps at which a pool refill happened

        is_any_success = False
        best_final_top1 = current_top1_str
        best_final_target_prob = baseline_target_prob
        step_counter = 0
        stop_sweep = False
        stop_reason = None

        applied_mutes, applied_boosts = [], []
        ledger = []                  # every applied feature: id, side, round, effects
        rejected_history = {"mute": {}, "boost": {}}   # fid -> [rounds in which it was rejected]
        all_rejections = []          # (round, side, fid, delta)
        round_records = []
        event_log = []
        total_passes = {"baseline": 1, "ranking+safety": 0, "steered baseline": 0, "sweep": 0}
        filter_time = 0.0

        ctx = clean_ctx
        prev_ctx = clean_ctx
        prev_start_counts = (0, 0)
        round_start_counts = (0, 0)
        round_idx = 0
        comp_ids, target_ids = [], []
        status_box = st.empty()
        cur_round = {}

        def update_tracker(phase, final=False):
            rows = []
            for r in round_records:
                rows.append({
                    "Round": r["round"], "Status": "✅ done",
                    "Started at rank": r["baseline_rank"], "Ended at rank": r["end_rank"], "Best rank in round": r["best_rank"],
                    "Safe mute / boost": f"{r['safe_mutes']} / {r['safe_boosts']}", "Steps": r["steps"],
                })
            if cur_round and not final:
                rows.append({
                    "Round": cur_round["round"], "Status": f"⏳ {phase}",
                    "Started at rank": cur_round["start_rank"], "Ended at rank": cur_round.get("now_rank", "…"),
                    "Best rank in round": cur_round.get("best_rank", "…"),
                    "Safe mute / boost": cur_round.get("pools", "…"), "Steps": cur_round.get("steps", 0),
                })
            last = rank_progression[-1]
            summary = (
                f"Rounds finished: **{len(round_records)}** · steps run: **{step_counter}** · target rank now **#{last['Target Rank']}** "
                f"({last['Target Prob (%)']:.2f}%) · best so far **#{best_so_far['rank']}** (started at #{clean_ctx.clean_rank}) · "
                f"features applied: **{len(applied_mutes)} mute / {len(applied_boosts)} boost** (committed at round ends)"
            )
            if final:
                msg = f"**Finished — {stop_reason or 'done'}**  \n{summary}"
                (tr_status.success if is_any_success else tr_status.warning)(msg)
            else:
                tr_status.info(f"🔄 **{phase}**  \n{summary}")
            if rows:
                tr_table.dataframe(pd.DataFrame(rows).set_index("Round"), use_container_width=True)
            with tr_chart.container():
                render_rank_progression_chart(rank_progression, best_step=best_so_far["step"], height=220, refill_steps=refill_markers)

        update_tracker("starting")

        while True:
            round_t0 = time.perf_counter()
            base_map = build_scale_map(applied_mutes, sm4, applied_boosts, sb4)
            round_top1_id = int(torch.argmax(ctx.clean_probs).item())
            round_top1_str = model.to_string([round_top1_id])
            round_top1_prob = ctx.clean_probs[round_top1_id].item()

            if round_idx == 0 and round_top1_id == target_token_id:
                stop_reason = "Target is already Rank #1 in the clean baseline — nothing to steer."
                is_any_success = True
                st.info(f"ℹ️ {stop_reason}")
                break

            st.markdown("---")
            st.header(f"🔄 Round {round_idx}" + (" — clean baseline" if round_idx == 0 else " — pool refill against the steered model"))

            # ---- Where things stand: original prompt vs how the previous round went vs where this round starts ----
            round_start_counts = (len(applied_mutes), len(applied_boosts))
            top1_orig_str = current_top1_str
            if round_idx == 0:
                st.markdown("**Starting point — the original prompt, nothing applied yet**")
                st.metric("Target rank", f"#{ctx.clean_rank}")
                st.caption(f"Target prob {ctx.clean_target_prob*100:.2f}% · the model's top-1 (the blocker) is `{round_top1_str}` at {round_top1_prob*100:.2f}%.")
            else:
                prev_top1_str = model.to_string([int(torch.argmax(prev_ctx.clean_probs))])
                bc_a, bc_b, bc_c = st.columns(3)
                with bc_a:
                    st.markdown("**1 · Original prompt** (nothing applied — never changes)")
                    st.metric("Target rank", f"#{clean_ctx.clean_rank}")
                    st.caption(f"Prob {clean_ctx.clean_target_prob*100:.2f}% · top-1 `{top1_orig_str}`")
                with bc_b:
                    st.markdown(f"**2 · How Round {round_idx - 1} went** (start → end)")
                    st.metric(
                        "Target rank", f"#{prev_ctx.clean_rank} → #{ctx.clean_rank}",
                        delta=(f"{prev_ctx.clean_rank - ctx.clean_rank:+d} ranks" if prev_ctx.clean_rank != ctx.clean_rank else "no change"),
                        delta_color="normal" if prev_ctx.clean_rank != ctx.clean_rank else "off",
                    )
                    st.caption(
                        f"Prob {prev_ctx.clean_target_prob*100:.2f}% → {ctx.clean_target_prob*100:.2f}% · "
                        f"applied features {prev_start_counts[0]} mute / {prev_start_counts[1]} boost → {len(applied_mutes)} / {len(applied_boosts)}"
                    )
                with bc_c:
                    st.markdown(f"**3 · Round {round_idx} starts here** (= where Round {round_idx - 1} ended)")
                    st.metric("Target rank", f"#{ctx.clean_rank}")
                    st.caption(f"Prob {ctx.clean_target_prob*100:.2f}% · top-1 (the blocker) `{round_top1_str}` at {round_top1_prob*100:.2f}%")
                if prev_ctx.clean_rank == ctx.clean_rank:
                    st.warning(
                        f"⚠️ Round {round_idx - 1} did not improve the target's rank (#{prev_ctx.clean_rank} → #{ctx.clean_rank}). "
                        f"Candidates are still re-ranked and re-tested below, but this round may not help either."
                    )
                st.info(
                    f"Round {round_idx - 1}'s candidate pool ran out and all of its features stay applied. Candidates for Round {round_idx} "
                    f"are now re-ranked and safety-tested against state 3 (target rank #{ctx.clean_rank}, blocker `{round_top1_str}`), not the original prompt."
                )

            # ---- Build this round's pools ----
            tf0 = time.perf_counter()
            with st.spinner(f"Round {round_idx}: ranking candidates and running the safety filter..."):
                pools = build_pools(ctx, base_map, set(applied_mutes) | set(applied_boosts))
            filter_time += time.perf_counter() - tf0
            total_passes["ranking+safety"] += pools["passes"]
            comp_ids, target_ids = pools["comp_ok"], pools["tgt_ok"]

            for fid, d in pools["comp_rej"]:
                rejected_history["mute"].setdefault(fid, []).append(round_idx)
                all_rejections.append((round_idx, "mute", fid, d))
            for fid, d in pools["tgt_rej"]:
                rejected_history["boost"].setdefault(fid, []).append(round_idx)
                all_rejections.append((round_idx, "boost", fid, d))

            # ---- Pool summary ----
            cur_round.clear()
            cur_round.update(round=round_idx, start_rank=ctx.clean_rank, now_rank=ctx.clean_rank, best_rank=ctx.clean_rank, steps=0, pools="…")
            st.subheader(f"Round {round_idx} candidate pools")
            st.write(f"Blocking token being suppressed this round: `{round_top1_str}` ({round_top1_prob*100:.2f}%).")
            st.write(
                "Active SAE features per prompt token (BOS excluded): "
                + " · ".join(f"`{t}` **{n}**" for t, n in pools["per_token"])
            )
            st.write(
                f"Candidate source: **{candidate_source_4}** → **{pools['n_active']}** distinct active features, "
                f"**{pools['available']}** not yet applied. Top N requested **{top_n_4}**, "
                f"so at most **{pools['eff_top_n']}** can be evaluated per side."
            )
            st.table([
                {"Pool": "Mute (competitor features)", "Requested (Top N)": top_n_4, "Available": pools["available"], "Candidates evaluated": len(pools["comp_fids"]),
                 "Safe (after overlap fix)": len(pools["comp_ok"]) if use_safety_4 else "filter off", "Rejected": len(pools["comp_rej"]),
                 "Moved to other side (overlap)": sum(1 for o in pools["overlap"] if o["Kept on"] == "boost")},
                {"Pool": "Boost (target features)", "Requested (Top N)": top_n_4, "Available": pools["available"], "Candidates evaluated": len(pools["tgt_fids"]),
                 "Safe (after overlap fix)": len(pools["tgt_ok"]) if use_safety_4 else "filter off", "Rejected": len(pools["tgt_rej"]),
                 "Moved to other side (overlap)": sum(1 for o in pools["overlap"] if o["Kept on"] == "mute")},
            ])
            if pools["overlap"]:
                st.caption(
                    f"🔀 **Overlap fix:** {len(pools['overlap'])} feature(s) passed the safety test on BOTH sides. Applied to both they would be muted and boosted at "
                    f"once and mostly cancel out, so each is now used on one side only. "
                    + ("The side that raises the target's probability more was kept." if use_safety_4 else "The side where it ranks higher in its own list was kept.")
                )
                with st.expander(f"🔀 Features assigned to one side ({len(pools['overlap'])})"):
                    st.dataframe(pd.DataFrame(pools["overlap"]).set_index("Feature"), use_container_width=True)
            else:
                st.caption("🔀 Overlap check: no feature was safe on both sides this round.")
            if top_n_4 > pools["available"]:
                st.caption(f"ℹ️ Top N was capped: you asked for {top_n_4} but only {pools['available']} unapplied feature(s) are active for '{candidate_source_4}', so that is the most that can be evaluated.")

            retested = []
            for side, fids, rej, ok in (("mute", pools["comp_fids"], pools["comp_rej"], pools["comp_ok"]),
                                       ("boost", pools["tgt_fids"], pools["tgt_rej"], pools["tgt_ok"])):
                rej_now = {f for f, _ in rej}
                for fid in fids:
                    earlier = [r for r in rejected_history[side].get(fid, []) if r < round_idx]
                    if earlier:
                        retested.append({
                            "Side": side, "Feature": fid,
                            "Rejected in round(s)": ", ".join(map(str, earlier)),
                            f"Round {round_idx} verdict": "❌ rejected again" if fid in rej_now else "✅ now safe",
                        })
            if retested:
                with st.expander(f"♻️ Previously rejected features re-tested this round ({len(retested)})", expanded=True):
                    st.table(retested)

            M, B = len(comp_ids), len(target_ids)
            event_log.append(
                f"Round {round_idx}: baseline rank #{ctx.clean_rank} (top-1 `{round_top1_str}`); pools built — "
                f"{M} safe mute / {B} safe boost (rejected {len(pools['comp_rej'])} / {len(pools['tgt_rej'])}; {len(pools['overlap'])} overlapping feature(s) assigned to one side); "
                f"{pools['passes']} forward passes for ranking + safety."
            )

            if M == 0 and B == 0:
                stop_reason = ("No unapplied active features remain at this position." if pools["available"] == 0
                               else f"Dead end: Round {round_idx} produced no safe mute or boost candidates.")
                st.warning(f"🛑 {stop_reason}")
                round_records.append({
                    "round": round_idx, "baseline_rank": ctx.clean_rank, "baseline_prob": ctx.clean_target_prob,
                    "blocker": round_top1_str, "safe_mutes": 0, "safe_boosts": 0, "overlap_resolved": len(pools["overlap"]),
                    "rejected_mutes": len(pools["comp_rej"]), "rejected_boosts": len(pools["tgt_rej"]),
                    "safe_mute_ids": list(comp_ids), "safe_boost_ids": list(target_ids),
                    "rejected_mute_ids": [f for f, _ in pools["comp_rej"]], "rejected_boost_ids": [f for f, _ in pools["tgt_rej"]],
                    "overlap": pools["overlap"],
                    "steps": 0, "end_rank": ctx.clean_rank, "best_rank": ctx.clean_rank,
                    "time_s": round(time.perf_counter() - round_t0, 3),
                })
                break

            if cumulative_sweep_4:
                combo_pairs_4 = [(min(k, M), min(k, B)) for k in range(1, max(M, B) + 1)]
            else:
                combo_pairs_4 = [(m_n, b_n) for m_n in mute_sizes_4 for b_n in boost_sizes_4]

            round_start_step = step_counter
            cur_round["pools"] = f"{M} / {B}"
            update_tracker(f"Round {round_idx}: pools built ({M} safe mute / {B} safe boost) — sweeping")
            round_best_rank = ctx.clean_rank
            round_end_rank = ctx.clean_rank
            st.subheader(f"Round {round_idx} sweep ({len(combo_pairs_4)} steps)")

            for m_n, b_n in combo_pairs_4:
                if stop_sweep:
                    break
                mute_batch = comp_ids[:m_n]
                boost_batch = target_ids[:b_n]
                full_mutes = applied_mutes + mute_batch
                full_boosts = applied_boosts + boost_batch

                model.reset_hooks()
                # Single combined hook (mutes + boosts from the same original signal) — matches
                # check_combination_safe exactly, so the top-5 table and the safety-check rank agree.
                combined_fn = make_mute_and_boost_hook(full_mutes, sm4, full_boosts, sb4, sae)
                model.add_hook(hook_name, combined_fn)

                with torch.no_grad():
                    logits = model(tokens)
                total_passes["sweep"] += 1
                probs = F.softmax(logits[0, -1, :], dim=-1)
                top5_probs, top5_indices = torch.topk(probs, k=5)
                new_top1_id = top5_indices[0].item()
                new_top1_str = model.to_string([new_top1_id])
                target_prob = probs[target_token_id].item()

                if new_top1_id == target_token_id:
                    is_any_success = True

                st.subheader(f"Round {round_idx} · Mute: {len(full_mutes)} features (-{sm4}) | Boost: {len(full_boosts)} features (+{sb4})")
                st.caption(f"{len(applied_mutes)} mutes / {len(applied_boosts)} boosts carried over from earlier rounds; this round adds {len(mute_batch)} / {len(boost_batch)}.")
                st.write(f"**Muted Features:** `{full_mutes}` | **Boosted Features:** `{full_boosts}`")
                st.write(f"**New Top-1:** `{new_top1_str}` | **Target Prob:** `{target_prob*100:.2f}%`")

                table_data = []
                for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                    tok_str = model.to_string([idx.item()])
                    is_target = "Yes (TARGET)" if idx.item() == target_token_id else "No"
                    table_data.append({
                        "Rank": rank_idx, "Token": tok_str,
                        "Probability": f"{p.item()*100:.2f}%", "Is Target": is_target,
                    })
                st.table(table_data)

                safety_res = check_combination_safe(
                    model, sae, prompt_4,
                    mute_feature_ids=full_mutes, mute_strength=sm4,
                    boost_feature_ids=full_boosts, boost_strength=sb4,
                    target_token_id=target_token_id, top_k=10
                )
                total_passes["sweep"] += 2

                st.write(f"**Combination Safety Check:** Target clean rank: `{safety_res['target_clean_rank']}` -> New rank: `{safety_res['target_new_rank']}`")
                if safety_res["new_blockers"]:
                    for blocker in safety_res["new_blockers"]:
                        c_rank = blocker["clean_rank_or_absent"]
                        was_str = f"was rank {c_rank}" if isinstance(c_rank, int) else "absent"
                        st.write(f"⚠️ New blocker: {blocker['token']!r} rose to rank {blocker['new_rank']} ({was_str} in clean baseline)")
                else:
                    st.write("✅ No new blockers detected")

                step_counter += 1
                combo_rank = safety_res["target_new_rank"]
                round_end_rank = combo_rank
                round_best_rank = min(round_best_rank, combo_rank)
                st.caption(
                    f"Round {round_idx} baseline was rank #{ctx.clean_rank} → now #{combo_rank} "
                    f"({_fmt_rank_change(combo_rank, ctx.clean_rank)}); vs the clean prompt: {_fmt_rank_change(combo_rank, clean_ctx.clean_rank)}."
                )
                rank_progression.append({
                    "Step": step_counter,
                    "Label": f"R{round_idx} M{len(full_mutes)}/B{len(full_boosts)}",
                    "Round": round_idx,
                    "Target Rank": combo_rank,
                    "Target Prob (%)": target_prob * 100,
                })
                is_new_best = (
                    combo_rank < best_so_far["rank"]
                    or (combo_rank == best_so_far["rank"] and target_prob > best_so_far["prob"])
                )
                if is_new_best:
                    best_so_far = {
                        "rank": combo_rank, "prob": target_prob, "top1": new_top1_str,
                        "mute_size": len(full_mutes), "boost_size": len(full_boosts),
                        "mute_features": list(full_mutes), "boost_features": list(full_boosts),
                        "step": step_counter,
                    }
                    st.caption(f"🏆 New best so far — target rank {combo_rank}")
                elif combo_rank > best_so_far["rank"]:
                    st.caption(f"↘️ Worse than the best so far (rank {best_so_far['rank']}, found at Mute {best_so_far['mute_size']}/Boost {best_so_far['boost_size']}) — that best is still kept as the reported result.")

                hybrid_details.append({
                    "round": round_idx,
                    "mute_batch_size": len(full_mutes),
                    "mute_features": list(full_mutes),
                    "mute_strength": sm4,
                    "boost_batch_size": len(full_boosts),
                    "boost_features": list(full_boosts),
                    "boost_strength": sb4,
                    "new_top1": new_top1_str,
                    "target_prob": f"{target_prob*100:.2f}%",
                    "top5": table_data,
                    "combination_safety_check": safety_res
                })

                cur_round.update(now_rank=combo_rank, best_rank=round_best_rank, steps=step_counter - round_start_step)
                update_tracker(f"Round {round_idx}: sweep step {step_counter - round_start_step} of {len(combo_pairs_4)}")
                status_box.info(
                    f"🔄 **Live status** — Round {round_idx} · step {step_counter} · target rank **#{combo_rank}** "
                    f"({target_prob*100:.2f}%) · top-1 `{new_top1_str}` · applied {len(full_mutes)} mutes / {len(full_boosts)} boosts · "
                    f"best so far #{best_so_far['rank']}"
                )

                if stop_on_rank1_4 and new_top1_id == target_token_id:
                    st.success(f"🎯 **Target token '{target_str}' reached Rank #1** in Round {round_idx} with {len(full_mutes)} mutes & {len(full_boosts)} boosts! Stopping.")
                    stop_sweep = True
                    break

            model.reset_hooks()
            round_records.append({
                "round": round_idx, "baseline_rank": ctx.clean_rank, "baseline_prob": ctx.clean_target_prob,
                "blocker": round_top1_str, "safe_mutes": M, "safe_boosts": B, "overlap_resolved": len(pools["overlap"]),
                "rejected_mutes": len(pools["comp_rej"]), "rejected_boosts": len(pools["tgt_rej"]),
                "safe_mute_ids": list(comp_ids), "safe_boost_ids": list(target_ids),
                "rejected_mute_ids": [f for f, _ in pools["comp_rej"]], "rejected_boost_ids": [f for f, _ in pools["tgt_rej"]],
                "overlap": pools["overlap"],
                "steps": step_counter - round_start_step, "end_rank": round_end_rank, "best_rank": round_best_rank,
                "time_s": round(time.perf_counter() - round_t0, 3),
            })
            regressed = round_end_rank > ctx.clean_rank
            cur_round.clear()
            update_tracker(f"Round {round_idx} finished — deciding whether to refill")

            if stop_sweep:
                stop_reason = f"Target reached Rank #1 in Round {round_idx}."
                break
            if not refill_on:
                stop_reason = ("Candidate pool exhausted (pool refill is off)." if cumulative_sweep_4
                               else "All requested batch-size combinations were run.")
                break

            # ---- Pool exhausted: commit this round's features, then refill ----
            for fid in comp_ids:
                ledger.append({"Feature": fid, "Side": "mute", "Round joined": round_idx,
                               "Target Δprob when tested (%)": (pools["comp_delta"][fid] * 100) if fid in pools["comp_delta"] else None,
                               "Effect on blocker if fully ablated (%)": pools["comp_effect"].get(fid, 0.0) * 100})
            for fid in target_ids:
                ledger.append({"Feature": fid, "Side": "boost", "Round joined": round_idx,
                               "Target Δprob when tested (%)": (pools["tgt_delta"][fid] * 100) if fid in pools["tgt_delta"] else None,
                               "Effect on target if fully ablated (%)": pools["tgt_effect"].get(fid, 0.0) * 100})
            applied_mutes = applied_mutes + comp_ids
            applied_boosts = applied_boosts + target_ids

            if max_rounds_4 and round_idx >= max_rounds_4:
                stop_reason = f"Reached the Max Refill Rounds limit ({max_rounds_4})."
                break

            tf0 = time.perf_counter()
            new_ctx = build_steered_context(model, sae, clean_ctx, build_scale_map(applied_mutes, sm4, applied_boosts, sb4), target_token_id)
            filter_time += time.perf_counter() - tf0
            total_passes["steered baseline"] += 1

            event_log.append(
                f"Round {round_idx} pool exhausted at {len(applied_mutes)} mutes / {len(applied_boosts)} boosts applied "
                f"(target rank #{round_end_rank}" + (", REGRESSED vs round baseline — kept applied (track-only)" if regressed else "") +
                f"). Refill: fresh steered forward pass → new baseline rank #{new_ctx.clean_rank}."
            )
            refill_markers.append(step_counter)
            if int(torch.argmax(new_ctx.clean_probs).item()) == target_token_id:
                stop_reason = "Target is already Rank #1 in the steered model (stop-on-rank-1 was off)."
                break
            prev_start_counts = round_start_counts
            prev_ctx, ctx = ctx, new_ctx
            round_idx += 1

        model.reset_hooks()
        status_box.empty()
        cur_round.clear()
        update_tracker("finished", final=True)

        if stop_reason:
            (st.success if is_any_success else st.warning)(f"**Run finished:** {stop_reason}")

        best_final_top1 = best_so_far["top1"]
        best_final_target_prob = best_so_far["prob"]

        st.markdown("---")
        st.subheader("🏆 Best Result Found During Sweep")
        if enable_xai:
            from src.explain import generate_best_result_explanation
            with st.spinner("Generating explanation..."):
                best_result_explanation = generate_best_result_explanation(
                    prompt=prompt_4,
                    target=target_4,
                    baseline_prob=baseline_target_prob,
                    baseline_rank=clean_ctx.clean_rank,
                    best_prob=best_so_far["prob"],
                    best_rank=best_so_far["rank"],
                    best_mute_size=best_so_far["mute_size"],
                    best_boost_size=best_so_far["boost_size"],
                    reached_target=is_any_success,
                    n_combinations_run=step_counter,
                    api_key=groq_key_input,
                )
            st.markdown(best_result_explanation)
        else:
            st.markdown(
                "This is the best combination seen across every row above — picked by lowest target rank "
                "(ties broken by higher target probability), **not** just whichever row ran last. The safety "
                "filter only vets individual candidate features before the sweep starts; it doesn't stop the "
                "target's rank from drifting up and down as more features get piled on within the sweep itself, "
                "so this is what tracks and protects the best result actually found. Enable **AI Explanations (Groq)** "
                "in the sidebar for an explanation generated specifically for this run's numbers."
            )
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Best Target Rank", f"#{best_so_far['rank']}", delta=f"{clean_ctx.clean_rank - best_so_far['rank']:+d} vs baseline", delta_color="normal")
        bc2.metric("Best Target Prob", f"{best_so_far['prob']*100:.2f}%")
        bc3.metric("Found at", f"Mute {best_so_far['mute_size']} / Boost {best_so_far['boost_size']}" if best_so_far["step"] > 0 else "Baseline (no combo beat it)")
        st.write(f"**Muted Features:** `{best_so_far['mute_features']}` | **Boosted Features:** `{best_so_far['boost_features']}` | **New Top-1:** `{best_so_far['top1']}`")

        st.subheader("🧾 Run Summary")
        rs1, rs2, rs3, rs4 = st.columns(4)
        rs1.metric("Rounds run", len(round_records))
        rs2.metric("Features applied at end", f"{len(applied_mutes)} mute / {len(applied_boosts)} boost" if refill_on else "n/a (single pool)")
        rs3.metric("Sweep steps", step_counter)
        rs4.metric("Forward passes (est.)", sum(total_passes.values()))
        st.write(f"**Stop reason:** {stop_reason or 'n/a'}")
        st.caption(
            "Forward passes: " + " · ".join(f"{k} {v}" for k, v in total_passes.items())
            + ". Sweep passes include the 2 extra passes the per-step Combination Safety Check runs."
        )
        if round_records:
            st.write("**Per-round timeline**")
            st.dataframe(pd.DataFrame([{
                "Round": r["round"], "Blocking token": r["blocker"],
                "Baseline rank": r["baseline_rank"], "Baseline prob (%)": round(r["baseline_prob"] * 100, 3),
                "Safe mute / boost": f"{r['safe_mutes']} / {r['safe_boosts']}",
                "Rejected mute / boost": f"{r['rejected_mutes']} / {r['rejected_boosts']}",
                "Steps": r["steps"], "Best rank in round": r["best_rank"], "End rank": r["end_rank"],
                "Time (s)": r["time_s"],
            } for r in round_records]).set_index("Round"), use_container_width=True)
        if event_log:
            with st.expander(f"📜 Refill event log ({len(event_log)} entries)", expanded=refill_on):
                for line in event_log:
                    st.markdown(f"- {line}")

        st.subheader("📉 Target Rank Progression")
        st.caption("The dashed green line marks Rank #1. The red diamond marks the best step found. Dotted orange lines mark where the pool was refilled. Hover a point for exact values.")
        render_rank_progression_chart(rank_progression, best_step=best_so_far["step"], refill_steps=refill_markers)
        with st.expander("Show underlying data"):
            st.dataframe(pd.DataFrame(rank_progression).set_index("Step"), use_container_width=True)

        # --- Compute stopwatch stop (GPU/CPU work only) ---
        if device == "cuda":
            torch.cuda.synchronize()
        t4_end = time.perf_counter()

        st.markdown("---")
        st.subheader("⏱️ Stopwatch")
        sw1, sw2, sw3 = st.columns(3)
        sw1.metric("Model compute time", f"{t4_end - t4_start:.3f}s")
        sw2.metric("Candidate ranking + safety filter (all rounds)", f"{filter_time:.3f}s")
        sw3.metric("Sweep (intervention passes)", f"{t4_end - t4_baseline_done - filter_time:.3f}s")
        st.caption(
            ("⚡ GPU-Batched mode was ON for this run." if use_batched_4 else "🐢 GPU-Batched mode was OFF — this run used the original one-at-a-time method.")
            + " Toggle the checkbox above and re-run with the same prompt to compare timings directly."
            + " This breakdown isolates the model computation (the thing the optimization actually changes) — it excludes the Neuronpedia network lookups below, which cost the same either way and depend on your internet, not on this code."
        )

        if ledger:
            with st.expander(f"📒 Applied-features ledger ({len(ledger)} features committed by pool refills)"):
                st.caption("Features committed at the end of each exhausted round, with the round they joined. The final round's pool is shown in its sweep rows above.")
                for row in ledger:
                    tested = row["Target Δprob when tested (%)"]
                    st.markdown(
                        f"- Round {row['Round joined']} · **{row['Side']}** · {make_feature_hover_link(row['Feature'], layer)} · target Δprob when tested alone: "
                        + (f"`{tested:+.3f}%`" if tested is not None else "`safety filter off`"),
                        unsafe_allow_html=True,
                    )

        if all_rejections:
            with st.expander(f"❌ Candidates rejected by safety filter ({len(all_rejections)} rejections across {len(round_records)} round(s))"):
                for rnd, side, fid, t_delta in all_rejections:
                    if side == "mute":
                        st.markdown(f"❌ Round {rnd}: competitor {make_feature_hover_link(fid, layer)} excluded from mute pool (target Δprob `{t_delta*100:+.2f}%`, or its rank got worse).", unsafe_allow_html=True)
                    else:
                        st.markdown(f"❌ Round {rnd}: target {make_feature_hover_link(fid, layer)} excluded from boost pool (target Δprob `{t_delta*100:+.2f}%`, or its rank got worse).", unsafe_allow_html=True)

        # --- Full wall-clock stopwatch stop (everything, including the network lookups above) ---
        t4_wall_end = time.perf_counter()
        st.metric("⏱️ Total wall-clock time (entire run, including network lookups)", f"{t4_wall_end - t4_start:.3f}s")
        st.caption("This is the number that matches how long you actually waited — model compute plus everything else the run did on screen.")

        # Save run to session history
        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Hybrid Mute & Boost",
            "layer": layer,
            "prompt": prompt_4,
            "target": target_4,
            "mute_strength": strength_mute_4,
            "boost_strength": strength_boost_4,
            "cumulative_sweep": cumulative_sweep_4,
            "pool_refill": refill_on,
            "max_refill_rounds": int(max_rounds_4) if refill_on else None,
            "rounds": round_records,
            "stop_reason": stop_reason,
            "forward_passes": total_passes,
            "mute_sizes": (f"cumulative 1..{hybrid_details[-1]['mute_batch_size']} across {len(round_records)} round(s)" if cumulative_sweep_4 and hybrid_details else mute_sizes_str_4),
            "boost_sizes": (f"cumulative 1..{hybrid_details[-1]['boost_batch_size']} across {len(round_records)} round(s)" if cumulative_sweep_4 and hybrid_details else boost_sizes_str_4),
            "top_n": top_n_4,
            "candidate_source": candidate_source_4,
            "use_batched": use_batched_4,
            "total_time_s": round(t4_end - t4_start, 3),
            "baseline_top1": current_top1_str,
            "baseline_target_prob": f"{baseline_target_prob*100:.2f}%",
            "final_top1": best_final_top1,
            "final_target_prob": f"{best_final_target_prob*100:.2f}%",
            "success": is_any_success,
            "hybrid_details": hybrid_details,
            # --- everything else shown on screen during the run ---
            "settings": {
                "safety_filter": use_safety_4, "stop_on_rank1": stop_on_rank1_4,
                "mute_strength": strength_mute_4, "boost_strength": strength_boost_4,
                "top_n": int(top_n_4), "candidate_source": candidate_source_4,
                "cumulative_sweep": cumulative_sweep_4, "pool_refill": refill_on,
                "max_refill_rounds": int(max_rounds_4), "gpu_batched": use_batched_4,
                "sae_layer": layer, "hook_name": hook_name, "device": device,
            },
            "baseline_rank": clean_ctx.clean_rank,
            "best_result": best_so_far,
            "rank_progression": rank_progression,
            "refill_steps": refill_markers,
            "event_log": event_log,
            "applied_features_ledger": ledger,
            "rejections": [{"round": r, "side": sd, "feature": f, "target_prob_delta": d} for r, sd, f, d in all_rejections],
            "applied_mutes_final": applied_mutes,
            "applied_boosts_final": applied_boosts,
            "timing": {
                "model_compute_s": round(t4_end - t4_start, 3),
                "filter_s": round(filter_time, 3),
                "sweep_s": round(t4_end - t4_baseline_done - filter_time, 3),
                "wall_clock_s": round(t4_wall_end - t4_start, 3),
            },
            "xai_best_result_explanation": None,
            "xai_mechanistic_explanation": None,
        }
        if enable_xai:
            run_record["xai_best_result_explanation"] = best_result_explanation
        st.session_state["history"].append(run_record)
        persist_history()
        st.success(f"Run saved to Session History (auto-saved to {st.session_state['history_file']})")

        if enable_xai:
            st.markdown("---")
            st.subheader("🤖 Explainable AI Analysis")
            with st.spinner("Generating mechanistic explanation..."):
                from src.explain import generate_mechanistic_explanation
                interventions = []
                unique_mutes = set()
                unique_boosts = set()
                for hd in hybrid_details:
                    for fid in hd.get("mute_features", []):
                        unique_mutes.add(fid)
                    for fid in hd.get("boost_features", []):
                        unique_boosts.add(fid)
                for fid in unique_mutes:
                    interventions.append({
                        "feature_id": fid,
                        "description": get_neuronpedia_explanation(fid, layer),
                        "action": "muted",
                        "strength": strength_mute_4
                    })
                for fid in unique_boosts:
                    interventions.append({
                        "feature_id": fid,
                        "description": get_neuronpedia_explanation(fid, layer),
                        "action": "boosted",
                        "strength": strength_boost_4
                    })
                
                # Determine baseline and final rank
                baseline_rank_val = (torch.argsort(probs, descending=True) == target_token_id).nonzero().item() + 1
                explanation = generate_mechanistic_explanation(
                    prompt=prompt_4,
                    target=target_4,
                    baseline_prob=baseline_target_prob,
                    baseline_rank=baseline_rank_val,
                    final_prob=best_final_target_prob,
                    final_rank=baseline_rank_val, # approximation
                    interventions=interventions,
                    api_key=groq_key_input
                )
                st.info(explanation, icon=":material/psychology:")
                run_record["xai_mechanistic_explanation"] = explanation
                persist_history()

# --- TAB 6: Monosemanticity Analysis ---
with tab6:
    st.header("Monosemanticity Analysis & Feature Audit")
    
    # Local Educational Pipeline Map
    st.info(
        "🗺️ **Mechanistic Interpretability Educational Pipeline Map**\n\n"
        "`Stage 1: Baseline Model State` ➔ **`Stage 2: Polysemanticity & Monosemanticity Audit` (ACTIVE STAGE)** ➔ `Stage 3: Feature Selection` ➔ `Stage 4: Activation Intervention` ➔ `Stage 5: Evaluation & Safety` \n\n"
        "💡 **Stage Goal:** Verify whether internal representation dials perform single, isolated jobs before attempting activation editing. "
        "Editing polysemantic dials causes widespread collateral damage, whereas editing clean SAE dials enables precise factual steering."
    )

    # 1. Dataset & SAE Status Panel
    status_summary = get_sae_status_summary(model, sae, layer=layer, corpus=get_default_corpus())
    with st.expander("📊 **Active SAE Model & Dataset Status Panel**", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SAE Release", status_summary["release"])
        c2.metric("Target Layer", f"Layer {status_summary['layer']}")
        c3.metric("SAE Dictionary Size (d_sae)", f"{status_summary['d_sae']:,}")
        c4.metric("Curated Testable Features", status_summary["curated_count"])
        st.caption(f"Hook point: `{status_summary['hook_name']}` | Corpus sentences: {status_summary['corpus_count']} | Raw MLP dimension (d_mlp): {status_summary['d_mlp']:,}")

    st.markdown("---")

    # 2. Feature Explorer Component
    st.subheader("🔍 Feature Explorer & Selection Bench")
    st.caption("Discover, search, or select from curated testable SAE features to audit.")

    curated_registry = get_curated_feature_registry(layer=layer)
    dropdown_options = {f"Feature {f['feature_id']} — {f['concept']} ({f['category']})": f['feature_id'] for f in curated_registry}

    col_exp1, col_exp2 = st.columns([1, 1])
    with col_exp1:
        selected_curated_label = st.selectbox(
            "Select from Curated Monosemantic Features",
            options=list(dropdown_options.keys()),
            index=0,
            key="mono_curated_dropdown"
        )
        curated_selected_id = dropdown_options[selected_curated_label]

    with col_exp2:
        direct_feature_id = st.number_input(
            "Or enter Feature ID directly",
            min_value=0,
            max_value=status_summary["d_sae"] - 1,
            value=curated_selected_id,
            step=1,
            key="mono_feature_id"
        )

    # Curated Features Metadata Table
    with st.expander("📋 **Browse Recommended Features Registry**", expanded=False):
        reg_df = pd.DataFrame([
            {
                "Feature ID": item["feature_id"],
                "Concept / Description": item["concept"],
                "Category": item["category"],
                "Interpretability": item["interpretability"],
                "Known Quality": item["known_quality"],
                "Details": item["description"]
            }
            for item in curated_registry
        ])
        st.dataframe(reg_df, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        neuron_index = st.number_input("Raw Model Neuron Dial Index", min_value=0, max_value=status_summary["d_mlp"] - 1, value=0, step=1, key="mono_neuron_index")
        corpus_source = st.radio("Corpus Source", options=["Bundled default", "Paste text", "Generate synthetic corpus (Groq AI)"], index=0, key="mono_corpus_source")
    with col2:
        use_groq = st.checkbox("Run autointerp scoring & dynamic AI explanations", value=bool(default_groq_key), key="mono_use_groq")
        groq_key = default_groq_key

    if corpus_source == "Bundled default":
        corpus = get_default_corpus()
        st.caption(f"Using bundled corpus with {len(corpus)} sentences across 10 categories.")
        with st.expander(f"👁️ **View Bundled Corpus Sentences ({len(corpus)} total)**", expanded=False):
            st.dataframe(pd.DataFrame({"Sentence #": range(1, len(corpus) + 1), "Text": corpus}), use_container_width=True)
    elif corpus_source == "Paste text":
        pasted = st.text_area("Paste a corpus, one sentence per line", height=220, key="mono_pasted_corpus")
        corpus = [line.strip() for line in pasted.splitlines() if line.strip()]
        if not corpus:
            st.info("Paste one or more sentences to analyze a custom corpus.")
    else:
        st.markdown("##### ✨ Synthetic Corpus Generator (Groq Agent)")
        synth_topic = st.text_input("Corpus Topic / Focus Domain", value="Cities, geography, science, and history", key="mono_synth_topic")
        synth_num = st.slider("Number of Sentences to Generate", min_value=5, max_value=30, value=15, key="mono_synth_num")
        
        if st.button("✨ Generate Synthetic Corpus via Groq Agent", key="btn_gen_synth"):
            if not default_groq_key:
                st.error("Groq API Key is missing in environment. Please ensure GROQ_API_KEY is set in your .env file.")
            else:
                with st.spinner("Generating diverse synthetic corpus using Groq AI agent..."):
                    try:
                        gen_corpus = generate_synthetic_corpus(topic=synth_topic, num_sentences=int(synth_num), api_key=default_groq_key)
                        st.session_state["mono_synthetic_corpus"] = gen_corpus
                        st.success(f"Generated {len(gen_corpus)} synthetic sentences!")
                    except Exception as e:
                        st.error(f"Error generating synthetic corpus: {e}")
                        
        corpus = st.session_state.get("mono_synthetic_corpus", [])
        if corpus:
            st.markdown(f"**Generated Corpus ({len(corpus)} sentences):**")
            st.dataframe(pd.DataFrame({"Sentence #": range(1, len(corpus) + 1), "Text": corpus}), use_container_width=True)
        else:
            st.info("Click 'Generate Synthetic Corpus via Groq Agent' to synthesize sentences on your chosen topic.")

    feature_id = int(direct_feature_id)

    # Input Validation checks
    is_f_valid, f_val_msg = validate_feature_id(sae, feature_id)
    is_n_valid, n_val_msg = validate_neuron_index(model, layer, int(neuron_index))

    if not is_f_valid:
        render_empty_state_card(get_empty_state_guidance("invalid_feature_id", {"feature_id": feature_id, "bounds_str": f"0 to {status_summary['d_sae'] - 1}"}))
    if not is_n_valid:
        render_empty_state_card(get_empty_state_guidance("invalid_neuron_index", {"neuron_index": neuron_index, "bounds_str": f"0 to {status_summary['d_mlp'] - 1}"}))

    if st.button("Run Monosemanticity Analysis & Feature Audit", key="btn_mono"):
        if not is_f_valid or not is_n_valid:
            st.error("Please provide valid Feature ID and Neuron Index bounds before running.")
            st.stop()
        if not corpus:
            st.warning("No corpus supplied.")
            st.stop()

        effective_groq_key = groq_key or groq_key_input

        with st.spinner("Scanning raw neuron & SAE dial activation patterns in a single fast pass..."):
            try:
                raw_neuron_examples, max_examples = scan_dual_activations(model, sae, int(neuron_index), feature_id, layer, corpus, top_n=10)
            except Exception as e:
                st.error(f"Error scanning activations: {e}")
                raw_neuron_examples, max_examples = [], []

        # Retrieve Neuronpedia explanation for the SAE feature dial
        sae_explanation = get_neuronpedia_explanation(feature_id, layer)

        # === RAW NEURON ANALYSIS ===
        st.subheader("Raw Neuron Analysis (Un-decomposed Baseline)")
        render_xai_guidance_card(get_xai_guidance_card("raw_neuron"))

        if raw_neuron_examples:
            top_raw_tokens = ", ".join(f"`{item['token']}`" for item in raw_neuron_examples[:6])
            st.markdown(f"**Top triggering tokens for Raw Neuron {neuron_index}:** {top_raw_tokens}")
            raw_rows = []
            for item in raw_neuron_examples:
                raw_rows.append({
                    "Activation": f"{item['activation']:.4f}",
                    "Trigger Word": item["token"],
                    "Position": item["token_position"],
                    "Sentence Context (Word in Bold)": item.get("highlighted_text", item["text"]),
                })
            st.dataframe(raw_rows, use_container_width=True)
        else:
            render_empty_state_card(get_empty_state_guidance("no_activations", {"feature_id": f"Raw Neuron {neuron_index}"}))

        # AI Polysemanticity Diagnosis Card
        raw_token_list = [item["token"] for item in raw_neuron_examples]
        sae_token_list = [item["token"] for item in max_examples]
        poly_xai = generate_polysemanticity_comparison_xai(int(neuron_index), feature_id, raw_token_list, sae_token_list, api_key=effective_groq_key)
        st.warning(f"🔬 **AI Polysemanticity Diagnosis:**\n\n{poly_xai}")

        st.markdown("---")

        # === SAE FEATURE ANALYSIS ===
        st.subheader("SAE Feature Analysis (Decomposed Feature Dial)")
        render_xai_guidance_card(get_xai_guidance_card("sae_feature"))

        if sae_explanation and sae_explanation != "Explanation unavailable":
            st.success(
                f"🧠 **Known Feature Concept (Neuronpedia):** `{sae_explanation}`\n\n"
                f"*(SAE Feature Dial ID: `{feature_id}` on Layer `{layer}`)*"
            )
        else:
            render_empty_state_card(get_empty_state_guidance("missing_neuronpedia", {"feature_id": feature_id, "layer": layer}))

        if max_examples:
            top_sae_tokens = ", ".join(f"`{item['token']}`" for item in max_examples[:6])
            st.markdown(f"**Top triggering tokens for SAE Dial {feature_id}:** {top_sae_tokens}")
            rows = []
            for item in max_examples:
                rows.append({
                    "Activation": f"{item['activation']:.4f}",
                    "Trigger Word": item["token"],
                    "Position": item["token_position"],
                    "Sentence Context (Word in Bold)": item.get("highlighted_text", item["text"]),
                })
            st.dataframe(rows, use_container_width=True)
        else:
            render_empty_state_card(get_empty_state_guidance("no_activations", {"feature_id": feature_id}))

        st.markdown("---")

        # === INTERPRETABILITY SCORE ===
        st.subheader("Interpretability Score (Autointerp Validation)")
        render_xai_guidance_card(get_xai_guidance_card("interpretability_score"))

        if use_groq and effective_groq_key:
            with st.spinner("Running Groq-based autointerp scoring..."):
                try:
                    score = score_feature_interpretability(model, sae, feature_id, corpus, effective_groq_key, held_out_fraction=0.2)
                except Exception as e:
                    st.warning(f"Failed to run autointerp scoring: {e}")
                    score = None

            if score and score.get("predictions"):
                st.metric("Autointerp Prediction Accuracy", f"{score['accuracy'] * 100:.1f}%")
                st.caption(f"Reference set size: {score['n_reference']}; Held-out validation set size: {score['n_held_out']}")

                st.markdown("#### (a) Reference examples shown to the AI evaluator")
                if score.get("reference_examples"):
                    ref_rows = []
                    for item in score["reference_examples"]:
                        ref_rows.append({
                            "Activation": f"{item['activation']:.4f}",
                            "Trigger Word": item["token"],
                            "Sentence": item.get("highlighted_text", item["text"]),
                        })
                    st.dataframe(ref_rows, use_container_width=True)

                st.markdown("#### (b) Held-out test cases validation")
                test_rows = []
                for item in score["predictions"]:
                    is_correct = item["correct"]
                    status_icon = "✅ Correct" if is_correct else "❌ Wrong"
                    test_rows.append({
                        "Status": status_icon,
                        "Prediction": item["predicted"],
                        "Actual": "HIGH" if item["actual_activation"] > 0.0 else "LOW",
                        "Actual Activation": f"{item['actual_activation']:.4f}",
                        "Sentence": item["text"],
                    })
                st.dataframe(test_rows, use_container_width=True)
        else:
            st.info("Provide a Groq API Key and check 'Run autointerp scoring' to run automated LLM prediction validation.")

        st.markdown("---")

        # === SPARSITY STATISTICS ===
        st.subheader("Sparsity Statistics (Representation Efficiency)")
        render_xai_guidance_card(get_xai_guidance_card("sparsity_stats"))

        with st.spinner("Computing sparsity stats..."):
            try:
                sparsity = compute_sparsity_stats(model, sae, corpus, max_corpus_items=200)
            except Exception as e:
                st.warning(f"Error computing sparsity stats: {e}")
                sparsity = {"mean_l0": 0.0, "l0_distribution": [], "cap": 200, "feature_firing_frequency": {}}

        st.metric("Mean L0 (Active Dials per Token)", f"{sparsity['mean_l0']:.4f}")
        st.caption(f"Corpus cap used: {sparsity['cap']} items")
        
        sparsity_xai = generate_sparsity_xai(sparsity["mean_l0"], len(corpus), api_key=effective_groq_key)
        st.info(f"📊 **Sparsity & L0 Analysis Verdict:**\n\n{sparsity_xai}")

        if sparsity["l0_distribution"]:
            st.bar_chart(pd.DataFrame({"L0 (Active Dials)": sparsity["l0_distribution"]}))
        if sparsity.get("feature_firing_frequency"):
            freq_rows = [{"Feature Dial ID": fid, "Firing Frequency": f"{freq:.4f}"} for fid, freq in sparsity["feature_firing_frequency"].items()]
            st.dataframe(freq_rows, use_container_width=True)

        st.markdown("---")

        # === MOST SIMILAR DECODER DIRECTIONS ===
        st.subheader("Most Similar Decoder Directions (Geometric Independence)")
        render_xai_guidance_card(get_xai_guidance_card("decoder_similarity"))

        with st.spinner("Comparing decoder directions..."):
            try:
                similar = find_most_similar_features(sae, feature_id, top_n=10)
            except Exception as e:
                st.warning(f"Error computing decoder similarity: {e}")
                similar = []

        if similar:
            sim_xai = generate_decoder_similarity_xai(feature_id, similar, api_key=effective_groq_key)
            st.info(f"📐 **Geometric Independence & Redundancy Verdict:**\n\n{sim_xai}")
            sim_rows = [{"Feature Dial ID": fid, "Cosine Similarity": f"{sim:.4f}"} for fid, sim in similar]
            st.dataframe(sim_rows, use_container_width=True)
        else:
            render_empty_state_card(get_empty_state_guidance("missing_decoder_similarity", {"feature_id": feature_id}))

# --- TAB 7: Session History & Benchmarks ---
with tab7:
    st.header("Session History & Benchmarks")
    st.caption("Every Hybrid Mute & Boost run in this session is captured in full — settings, every round's pools, every sweep step, rejections, rank progression, timings and AI explanations.")

    st.caption(f"Auto-saved to disk: `{st.session_state['history_file']}` (survives page refresh; older sessions stay in `outputs/session_history/`).")

    def _json_bytes(obj):
        return json.dumps(obj, indent=2, default=str).encode("utf-8")

    def _safe_df(rows):
        return pd.DataFrame(rows).astype(str) if rows else pd.DataFrame()

    uploaded_file = st.file_uploader("Upload a past session JSON to view/analyze", type=["json"], key="history_uploader")
    uploaded_history = None
    if uploaded_file is not None:
        try:
            uploaded_history = json.load(uploaded_file)
            if isinstance(uploaded_history, dict):
                uploaded_history = [uploaded_history]
            st.success(f"Loaded {len(uploaded_history)} run(s) from uploaded file.")
        except Exception as e:
            st.error(f"Error loading JSON file: {e}")

    history_to_show = uploaded_history if uploaded_history is not None else st.session_state.get("history", [])

    if not history_to_show:
        st.info("No runs logged in this session yet. Run the Hybrid Mute & Boost tab, or upload a previously downloaded session JSON file.")
    else:
        summary_rows = []
        for rec in history_to_show:
            best = rec.get("best_result") or {}
            summary_rows.append({
                "Run ID": rec.get("run_id"),
                "Timestamp": rec.get("timestamp"),
                "Mode": rec.get("mode"),
                "Layer": rec.get("layer"),
                "Prompt": rec.get("prompt"),
                "Target": rec.get("target"),
                "Rounds": len(rec.get("rounds") or []),
                "Steps": len(rec.get("hybrid_details") or []),
                "Baseline rank": rec.get("baseline_rank"),
                "Best rank": best.get("rank"),
                "Baseline Top-1": rec.get("baseline_top1"),
                "Baseline Target Prob": rec.get("baseline_target_prob"),
                "Final Top-1": rec.get("final_top1"),
                "Final Target Prob": rec.get("final_target_prob"),
                "Total time (s)": rec.get("total_time_s"),
                "Stop reason": rec.get("stop_reason"),
                "Success": "TRUE" if rec.get("success") else "FALSE",
            })
        df_summary = pd.DataFrame(summary_rows)
        st.subheader("Summary Table of Session Runs")
        st.dataframe(df_summary, use_container_width=True)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button("📥 Summary (CSV)", df_summary.to_csv(index=False).encode("utf-8"),
                               file_name="session_summary.csv", mime="text/csv", key="dl_hist_csv")
        with col_d2:
            st.download_button("📥 Full session, every detail (JSON)", _json_bytes(history_to_show),
                               file_name="session_full_details.json", mime="application/json", key="dl_hist_json")
        with col_d3:
            if uploaded_history is None and st.button("🗑️ Clear Session History", key="clear_hist"):
                st.session_state["history"] = []
                st.rerun()

        st.markdown("---")
        st.subheader("Run Viewer")
        for rec in reversed(history_to_show):
            rid = rec.get("run_id")
            best = rec.get("best_result") or {}
            with st.expander(f"Run #{rid} — {rec.get('mode')} (Layer {rec.get('layer')}) | '{rec.get('prompt')}' → '{rec.get('target')}' | {rec.get('timestamp')}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Baseline**")
                    st.write(f"Top-1: `{rec.get('baseline_top1')}`")
                    st.write(f"Target prob: `{rec.get('baseline_target_prob')}` · rank: `{rec.get('baseline_rank')}`")
                with c2:
                    st.markdown("**Best result**")
                    st.write(f"Top-1: `{rec.get('final_top1')}`")
                    st.write(f"Target prob: `{rec.get('final_target_prob')}` · rank: `{best.get('rank')}`")
                    if best:
                        st.write(f"Found at Mute {best.get('mute_size')} / Boost {best.get('boost_size')} (step {best.get('step')})")
                        st.write(f"Muted: `{best.get('mute_features')}` · Boosted: `{best.get('boost_features')}`")
                with c3:
                    st.markdown("**Status**")
                    (st.success if rec.get("success") else st.warning)("✅ Target reached Rank #1" if rec.get("success") else "⚠️ Target did not reach Rank #1")
                    st.write(f"Stop reason: {rec.get('stop_reason') or 'n/a'}")

                tabs_v = st.tabs(["Settings", "Rounds", "Sweep steps", "Rank progression", "Features & rejections", "Timing & log", "AI explanations", "Raw JSON"])

                with tabs_v[0]:
                    st.json(rec.get("settings") or {k: rec.get(k) for k in [
                        "mute_strength", "boost_strength", "cumulative_sweep", "pool_refill", "max_refill_rounds",
                        "mute_sizes", "boost_sizes", "top_n", "candidate_source", "use_batched"] if k in rec})

                with tabs_v[1]:
                    rounds = rec.get("rounds") or []
                    if rounds:
                        st.dataframe(_safe_df(rounds), use_container_width=True)
                        for r in rounds:
                            if r.get("safe_mute_ids") is not None:
                                st.markdown(f"**Round {r.get('round')}** — safe mutes `{r.get('safe_mute_ids')}` · safe boosts `{r.get('safe_boost_ids')}` · "
                                            f"rejected mutes `{r.get('rejected_mute_ids')}` · rejected boosts `{r.get('rejected_boost_ids')}`")
                    else:
                        st.caption("No round data recorded.")

                with tabs_v[2]:
                    steps = rec.get("hybrid_details") or []
                    if not steps:
                        st.caption("No sweep steps recorded.")
                    for i, step in enumerate(steps, start=1):
                        label = (f"Step {i} · Round {step.get('round', '?')} · Mute {step.get('mute_batch_size')} (-{step.get('mute_strength')}) / "
                                 f"Boost {step.get('boost_batch_size')} (+{step.get('boost_strength')}) · Target prob {step.get('target_prob')}")
                        with st.expander(label):
                            st.write(f"Muted: `{step.get('mute_features', [])}`")
                            st.write(f"Boosted: `{step.get('boost_features', [])}`")
                            st.write(f"New Top-1: `{step.get('new_top1')}`")
                            if isinstance(step.get("top5"), list):
                                st.table(step["top5"])
                            sc = step.get("combination_safety_check")
                            if sc:
                                st.write(f"**Combination Safety Check:** target clean rank `{sc.get('target_clean_rank')}` → new rank `{sc.get('target_new_rank')}`")
                                for blocker in sc.get("new_blockers") or []:
                                    c_rank = blocker.get("clean_rank_or_absent")
                                    was_str = f"was rank {c_rank}" if isinstance(c_rank, int) else "absent"
                                    st.write(f"⚠️ New blocker: {blocker.get('token')!r} rose to rank {blocker.get('new_rank')} ({was_str} in clean baseline)")
                                if not sc.get("new_blockers"):
                                    st.write("✅ No new blockers detected")

                with tabs_v[3]:
                    rp = rec.get("rank_progression") or []
                    if rp:
                        render_rank_progression_chart(rp, best_step=best.get("step"), refill_steps=rec.get("refill_steps"))
                        st.dataframe(pd.DataFrame(rp).set_index("Step"), use_container_width=True)
                    else:
                        st.caption("No rank progression recorded.")

                with tabs_v[4]:
                    st.write(f"**Final applied mutes:** `{rec.get('applied_mutes_final', [])}`")
                    st.write(f"**Final applied boosts:** `{rec.get('applied_boosts_final', [])}`")
                    ledger_rows = rec.get("applied_features_ledger") or []
                    if ledger_rows:
                        st.markdown("**Applied-features ledger**")
                        st.dataframe(_safe_df(ledger_rows), use_container_width=True)
                    rej = rec.get("rejections") or []
                    if rej:
                        st.markdown("**Safety-filter rejections**")
                        st.dataframe(_safe_df(rej), use_container_width=True)

                with tabs_v[5]:
                    st.json({"timing": rec.get("timing"), "total_time_s": rec.get("total_time_s"), "forward_passes": rec.get("forward_passes")})
                    for line in rec.get("event_log") or []:
                        st.markdown(f"- {line}")

                with tabs_v[6]:
                    if rec.get("xai_best_result_explanation"):
                        st.markdown("**Best-result explanation**")
                        st.markdown(rec["xai_best_result_explanation"])
                    if rec.get("xai_mechanistic_explanation"):
                        st.markdown("**Mechanistic explanation**")
                        st.info(rec["xai_mechanistic_explanation"])
                    if not (rec.get("xai_best_result_explanation") or rec.get("xai_mechanistic_explanation")):
                        st.caption("No AI explanations for this run (AI Explanations were off).")

                with tabs_v[7]:
                    st.json(rec, expanded=False)

                st.download_button(f"📥 Download Run #{rid} (JSON)", _json_bytes(rec),
                                   file_name=f"run_{rid}.json", mime="application/json", key=f"dl_run_{rid}_{rec.get('timestamp')}")

# --- TAB 11: Hybrid Mute & Boost — Original (Sequential) vs Optimized (GPU Batched) ---
with tab11:
    st.header("Hybrid Mute & Boost — Original vs Optimized, Side by Side")
    st.markdown(
        """
        This is **the same Hybrid Mute & Boost sweep as Tab 4** (mute competitors, boost target,
        try each mute/boost batch-size combination until the target reaches Rank #1), run **twice
        end-to-end on identical inputs**: once with the **original sequential method** (the one the
        app first shipped with — everything one candidate/one forward pass at a time), and once with
        the **new GPU-batched method**. Both stop early the moment the target hits Rank #1, exactly
        like Tab 4. The clock starts at the clean baseline pass and stops the instant a run finishes
        (either by hitting the target or exhausting the sweep), so the total time you see below is the
        real, whole-workflow time — not just one isolated piece of it.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        prompt_11 = st.text_input("Prompt", "Seiyu Group's headquarters are in", key="t11_prompt")
        target_11 = st.text_input("Target Completion", "Tokyo", key="t11_target")
        top_n_11 = st.number_input("Top N Candidate Features", value=30, min_value=1, step=1, key="t11_topn")
    with col2:
        cumulative_sweep_11 = st.checkbox(
            "🔁 Cumulative Sweep (pile on one feature at a time until Target reaches Rank 1)", value=False, key="t11_cumulative",
            help="Ignores the Mute/Boost Batch Sizes below. Instead runs Mute 1/Boost 1, then Mute 2/Boost 2, then Mute 3/Boost 3, and so on — one feature added to each side per step — stopping the moment the target reaches Rank #1 (or once the candidate pool runs out). Applied identically to both the sequential and batched runs."
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            strength_mute_11 = st.number_input("Mute Strength", value=0.3, step=0.1, key="t11_m_strength")
            mute_sizes_str_11 = st.text_input("Mute Batch Sizes", "1, 3, 5", key="t11_m_bs", disabled=cumulative_sweep_11)
        with col_m2:
            strength_boost_11 = st.number_input("Boost Strength", value=0.5, step=0.1, key="t11_b_strength")
            boost_sizes_str_11 = st.text_input("Boost Batch Sizes", "1, 3, 5", key="t11_b_bs", disabled=cumulative_sweep_11)
        use_safety_11 = st.checkbox("Enable Safety Filter (Target Protection)", value=True, key="t11_safety")
        stop_on_rank1_11 = st.checkbox("Stop sweep once Target reaches Rank 1", value=True, key="t11_stop_rank1")

    def _run_hybrid_workflow(prompt, target_token_id, top_n, mute_sizes, boost_sizes,
                              strength_mute, strength_boost, use_safety, stop_on_rank1, use_batched, cumulative=False):
        """Runs the full Tab-4-style hybrid sweep end-to-end and times it with a single stopwatch.
        use_batched=False reproduces the app's original sequential behaviour;
        use_batched=True uses the GPU-batched candidate ranking + safety filtering."""
        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        model.reset_hooks()
        clean_ctx = build_clean_context(model, sae, prompt, target_token_id)
        tokens = clean_ctx.tokens
        current_top1_id = torch.argmax(clean_ctx.clean_probs).item()
        current_top1_str = model.to_string([current_top1_id])
        baseline_target_prob = clean_ctx.clean_target_prob
        t_baseline_done = time.perf_counter()

        competitor_features = get_top_competitor_features(
            model, sae, prompt, current_top1_id, top_n=top_n, clean_ctx=clean_ctx, use_batched=use_batched
        )
        comp_ids = []
        comp_rejections = []
        if use_safety:
            comp_fids = [fid for fid, _ in competitor_features]
            if use_batched:
                batch_res = check_target_safe_batch(model, sae, clean_ctx, comp_fids, target_token_id, strength=strength_mute)
                for fid, (is_safe, delta) in zip(comp_fids, batch_res):
                    if is_safe:
                        comp_ids.append(fid)
                    else:
                        comp_rejections.append({"feature_id": fid, "reason": "harms target token", "target_prob_delta": f"{delta*100:.3f}%"})
            else:
                for fid in comp_fids:
                    is_safe, delta = check_target_safe(
                        model, sae, prompt, fid, target_token_id, strength=strength_mute,
                        clean_target_prob=baseline_target_prob, clean_rank=clean_ctx.clean_rank
                    )
                    if is_safe:
                        comp_ids.append(fid)
                    else:
                        comp_rejections.append({"feature_id": fid, "reason": "harms target token", "target_prob_delta": f"{delta*100:.3f}%"})
        else:
            comp_ids = [fid for fid, _ in competitor_features]

        target_features = get_top_target_features(
            model, sae, prompt, target_token_id, top_n=top_n, clean_ctx=clean_ctx, use_batched=use_batched
        )
        target_ids = []
        boost_rejections = []
        if use_safety:
            tgt_fids = [fid for fid, _ in target_features]
            if use_batched:
                batch_res = check_boost_safe_batch(model, sae, clean_ctx, tgt_fids, target_token_id, strength=strength_boost)
                for fid, (is_safe, delta, rank_imp) in zip(tgt_fids, batch_res):
                    if is_safe:
                        target_ids.append(fid)
                    else:
                        boost_rejections.append({"feature_id": fid, "reason": "degrades target rank", "target_prob_delta": f"{delta*100:.3f}%"})
            else:
                for fid in tgt_fids:
                    is_safe, delta, rank_imp = check_boost_safe(
                        model, sae, prompt, fid, target_token_id, strength=strength_boost,
                        clean_target_prob=baseline_target_prob, clean_rank=clean_ctx.clean_rank
                    )
                    if is_safe:
                        target_ids.append(fid)
                    else:
                        boost_rejections.append({"feature_id": fid, "reason": "degrades target rank", "target_prob_delta": f"{delta*100:.3f}%"})
        else:
            target_ids = [fid for fid, _ in target_features]

        t_filtering_done = time.perf_counter()

        steps = []
        reached_target = False
        stopped_at = None
        best_so_far = {
            "rank": clean_ctx.clean_rank,
            "prob": baseline_target_prob,
            "top1": current_top1_str,
            "mute_size": 0,
            "boost_size": 0,
            "mute_features": [],
            "boost_features": [],
            "step": 0,
        }
        rank_progression = [{
            "Step": 0, "Label": "Baseline",
            "Target Rank": clean_ctx.clean_rank, "Target Prob (%)": baseline_target_prob * 100,
        }]
        step_counter = 0

        if cumulative:
            max_pile = min(len(comp_ids), len(target_ids))
            combo_pairs = [(n, n) for n in range(1, max_pile + 1)]
        else:
            combo_pairs = [(m_n, b_n) for m_n in mute_sizes for b_n in boost_sizes]

        for m_n, b_n in combo_pairs:
                if reached_target:
                    break
                mute_batch = comp_ids[:m_n]
                boost_batch = target_ids[:b_n]

                model.reset_hooks()
                # Single combined hook (see Tab 4 for why): keeps this sweep's own rank/prob
                # numbers consistent with check_combination_safe's independently-computed rank.
                combined_fn = make_mute_and_boost_hook(mute_batch, strength_mute, boost_batch, strength_boost, sae)
                model.add_hook(hook_name, combined_fn)

                with torch.no_grad():
                    logits = model(tokens)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                top5_probs, top5_indices = torch.topk(probs, k=5)
                new_top1_id = top5_indices[0].item()
                new_top1_str = model.to_string([new_top1_id])
                target_prob = probs[target_token_id].item()

                top5_table = []
                for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                    top5_table.append({
                        "Rank": rank_idx,
                        "Token": model.to_string([idx.item()]),
                        "Probability": f"{p.item()*100:.2f}%",
                        "Is Target": "Yes (TARGET)" if idx.item() == target_token_id else "No",
                    })

                sorted_indices = torch.argsort(probs, descending=True)
                new_rank = (sorted_indices == target_token_id).nonzero().item() + 1

                steps.append({
                    "mute_size": m_n,
                    "boost_size": b_n,
                    "muted_features": mute_batch,
                    "boosted_features": boost_batch,
                    "new_top1": new_top1_str,
                    "target_prob": target_prob,
                    "target_rank": new_rank,
                    "target_reached": new_top1_id == target_token_id,
                    "top5_table": top5_table,
                })

                step_counter += 1
                rank_progression.append({
                    "Step": step_counter, "Label": f"M{m_n}/B{b_n}",
                    "Target Rank": new_rank, "Target Prob (%)": target_prob * 100,
                })
                if new_rank < best_so_far["rank"] or (new_rank == best_so_far["rank"] and target_prob > best_so_far["prob"]):
                    best_so_far = {
                        "rank": new_rank, "prob": target_prob, "top1": new_top1_str,
                        "mute_size": m_n, "boost_size": b_n,
                        "mute_features": mute_batch, "boost_features": boost_batch,
                        "step": step_counter,
                    }

                if stop_on_rank1 and new_top1_id == target_token_id:
                    reached_target = True
                    stopped_at = (m_n, b_n)
                    break

        model.reset_hooks()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        rows = [{
            "Mute Size": s["mute_size"],
            "Boost Size": s["boost_size"],
            "New Top-1": s["new_top1"],
            "Target Prob": f"{s['target_prob']*100:.2f}%",
            "Target Reached #1": "✅" if s["target_reached"] else "",
        } for s in steps]

        return {
            "elapsed": elapsed,
            "t_baseline": t_baseline_done - t_start,
            "t_filtering": t_filtering_done - t_baseline_done,
            "t_sweep": elapsed - (t_filtering_done - t_start),
            "rows": rows,
            "steps": steps,
            "comp_rejections": comp_rejections,
            "boost_rejections": boost_rejections,
            "n_combinations_run": len(steps),
            "n_combinations_total": len(combo_pairs),
            "reached_target": reached_target,
            "stopped_at": stopped_at,
            "baseline_top1": current_top1_str,
            "baseline_target_prob": baseline_target_prob,
            "final_top1": rows[-1]["New Top-1"] if rows else current_top1_str,
            "final_target_prob": rows[-1]["Target Prob"] if rows else f"{baseline_target_prob*100:.2f}%",
            "mute_pool_size": len(comp_ids),
            "boost_pool_size": len(target_ids),
            "best_so_far": best_so_far,
            "rank_progression": rank_progression,
        }

    if st.button("Run Both Versions & Compare", key="btn_t11"):
        try:
            mute_sizes_11 = [int(x.strip()) for x in mute_sizes_str_11.split(",") if x.strip()]
        except ValueError:
            mute_sizes_11 = [1, 3, 5]
            st.warning("Invalid mute batch sizes; using default [1, 3, 5].")

        try:
            boost_sizes_11 = [int(x.strip()) for x in boost_sizes_str_11.split(",") if x.strip()]
        except ValueError:
            boost_sizes_11 = [1, 3, 5]
            st.warning("Invalid boost batch sizes; using default [1, 3, 5].")

        prompt_11_s = prompt_11.strip()
        target_11_s = " " + target_11.strip()
        target_token_id_11 = get_target_token_id(model, target_11_s)

        wall_start_seq = time.perf_counter()
        with st.spinner("Running the ORIGINAL sequential version in the background..."):
            result_seq = _run_hybrid_workflow(
                prompt_11_s, target_token_id_11, top_n_11, mute_sizes_11, boost_sizes_11,
                strength_mute_11, strength_boost_11, use_safety_11, stop_on_rank1_11, use_batched=False,
                cumulative=cumulative_sweep_11
            )

        wall_start_batch = time.perf_counter()
        with st.spinner("Running the NEW GPU-batched version..."):
            result_batch = _run_hybrid_workflow(
                prompt_11_s, target_token_id_11, top_n_11, mute_sizes_11, boost_sizes_11,
                strength_mute_11, strength_boost_11, use_safety_11, stop_on_rank1_11, use_batched=True,
                cumulative=cumulative_sweep_11
            )

        speedup = result_seq["elapsed"] / result_batch["elapsed"] if result_batch["elapsed"] > 0 else float("inf")
        same_outcome = (
            result_seq["final_top1"] == result_batch["final_top1"]
            and result_seq["reached_target"] == result_batch["reached_target"]
            and result_seq["n_combinations_run"] == result_batch["n_combinations_run"]
        )

        st.markdown("---")
        st.subheader("⏱️ Stopwatch — Head-to-Head Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Original (sequential)", f"{result_seq['elapsed']:.3f}s")
        m2.metric("Optimized (GPU batched)", f"{result_batch['elapsed']:.3f}s")
        m3.metric("Speedup", f"{speedup:.2f}x")

        st.write("**Stage-by-stage breakdown** (where the time actually goes):")
        st.table([
            {
                "Stage": "Baseline pass",
                "Original (s)": f"{result_seq['t_baseline']:.3f}",
                "Optimized (s)": f"{result_batch['t_baseline']:.3f}",
            },
            {
                "Stage": "Candidate ranking + safety filter",
                "Original (s)": f"{result_seq['t_filtering']:.3f}",
                "Optimized (s)": f"{result_batch['t_filtering']:.3f}",
            },
            {
                "Stage": "Sweep (intervention passes)",
                "Original (s)": f"{result_seq['t_sweep']:.3f}",
                "Optimized (s)": f"{result_batch['t_sweep']:.3f}",
            },
            {
                "Stage": "TOTAL",
                "Original (s)": f"{result_seq['elapsed']:.3f}",
                "Optimized (s)": f"{result_batch['elapsed']:.3f}",
            },
        ])

        if same_outcome:
            st.success(f"Both versions reached the same final result — same Top-1 token (`{result_seq['final_top1']}`), same stop condition, same number of sweep combinations run. Only the runtime differs.")
        else:
            st.error("The two versions did NOT reach the same outcome — check the tables below.")

        def _render_side(label, caption, result, wall_start):
            st.write(f"**Model compute time:** `{result['elapsed']:.3f}s`  (baseline `{result['t_baseline']:.3f}s` + filtering `{result['t_filtering']:.3f}s` + sweep `{result['t_sweep']:.3f}s`)")
            st.write(f"**Baseline Top-1:** `{result['baseline_top1']}` | **Baseline Target Prob:** `{result['baseline_target_prob']*100:.2f}%`")
            st.write(f"**Mute pool (passed safety):** {result['mute_pool_size']} | **Boost pool (passed safety):** {result['boost_pool_size']}")
            st.write(f"**Combinations run:** {result['n_combinations_run']} / {result['n_combinations_total']}" + (f" (stopped early at mute={result['stopped_at'][0]}, boost={result['stopped_at'][1]})" if result["stopped_at"] else " (target never reached rank #1 — sweep exhausted)"))
            st.write(f"**Final Top-1:** `{result['final_top1']}` | **Final Target Prob:** `{result['final_target_prob']}`")

            with st.expander(f"❌ Candidates rejected by safety filter ({len(result['comp_rejections']) + len(result['boost_rejections'])})"):
                if result["comp_rejections"]:
                    st.write("**Mute pool rejections:**")
                    st.table(result["comp_rejections"])
                else:
                    st.write("No mute candidates rejected.")
                if result["boost_rejections"]:
                    st.write("**Boost pool rejections:**")
                    st.table(result["boost_rejections"])
                else:
                    st.write("No boost candidates rejected.")

            best = result["best_so_far"]
            st.write(
                f"**🏆 Best rank found in sweep:** `#{best['rank']}` at `{best['prob']*100:.2f}%` "
                + (f"(Mute {best['mute_size']} / Boost {best['boost_size']})" if best["step"] > 0 else "(baseline — no combo beat it)")
                + " — tracked across every row, not just the last one run."
            )
            render_rank_progression_chart(result["rank_progression"], best_step=best["step"], height=260)

            st.write("**Step-by-step sweep:**")
            for i, s in enumerate(result["steps"], start=1):
                title = f"Step {i}: Mute {s['mute_size']} | Boost {s['boost_size']} → Top-1 `{s['new_top1']}` (rank #{s['target_rank']}, {s['target_prob']*100:.2f}%)" + (" ✅ TARGET REACHED" if s["target_reached"] else "") + (" 🏆" if s["mute_size"] == best["mute_size"] and s["boost_size"] == best["boost_size"] and s["target_rank"] == best["rank"] else "")
                with st.expander(title):
                    st.write(f"**Muted Features:** `{s['muted_features']}`")
                    st.write(f"**Boosted Features:** `{s['boosted_features']}`")
                    st.table(s["top5_table"])

            wall_elapsed = time.perf_counter() - wall_start
            st.metric("⏱️ Total wall-clock time (this run, everything included)", f"{wall_elapsed:.3f}s")
            st.caption("Compute time plus any extra work spent building this display — the number that matches how long this side actually took, start to finish.")
            return wall_elapsed

        col_seq, col_batch = st.columns(2)
        with col_seq:
            st.markdown("### 🐢 Original (Sequential)")
            st.caption("The version the app first shipped with — evaluates candidate features and safety checks one at a time.")
            wall_seq = _render_side("Original", "sequential", result_seq, wall_start_seq)

        with col_batch:
            st.markdown("### ⚡ Optimized (GPU Batched)")
            st.caption("Candidate ranking + safety filtering run as batched GPU calls instead of one-by-one.")
            wall_batch = _render_side("Optimized", "batched", result_batch, wall_start_batch)

        st.markdown("---")
        st.subheader("⏱️ Wall-Clock Summary (entire run, including everything on screen)")
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Original — wall-clock", f"{wall_seq:.3f}s")
        wc2.metric("Optimized — wall-clock", f"{wall_batch:.3f}s")
        wc3.metric("Speedup (wall-clock)", f"{wall_seq / wall_batch:.2f}x" if wall_batch > 0 else "N/A")
        st.caption("This is the total time each side actually took, start to finish — not just the isolated model-compute numbers above.")


# --- TAB 12: Batch runner — Last token vs All prompt positions ---
with tab12:
    from src.batch_runner import parse_spec, run_batch

    st.header("Batch Runner — Last Token vs All Prompt Positions")
    st.markdown(
        "Paste or upload a JSON spec listing many prompts. Every prompt is run through the **Hybrid Mute & Boost** sweep "
        "in each selected candidate-source mode (*All prompt positions* and/or *Last token only*), with identical settings. "
        "Everything the Hybrid tab shows on screen is captured for each run, then a paired comparison is built. "
        "Download the result as one JSON file for analysis."
    )

    _spec_default_path = _os.path.join("data", "candidate_source_batch_spec.json")
    _spec_default = ""
    try:
        with open(_spec_default_path, encoding="utf-8") as _f:
            _spec_default = _f.read()
    except Exception:
        _spec_default = json.dumps({
            "name": "my batch", "settings": {"mute_strength": 0.6, "boost_strength": 0.5, "top_n": 120},
            "modes": ["all", "last"], "repeats": 1,
            "prompts": [{"prompt": "This is Sophia, she is a", "target": "woman"}],
        }, indent=2)

    with st.expander("Spec format and allowed settings", expanded=False):
        st.code(
            '{\n  "name": "my batch",\n  "settings": {"mute_strength": 0.6, "boost_strength": 0.5, "top_n": 120,\n'
            '               "cumulative_sweep": true, "pool_refill": true, "max_refill_rounds": 0,\n'
            '               "safety_filter": true, "stop_on_rank1": true, "use_batched": true},\n'
            '  "modes": ["all", "last"],          // "all" = all prompt positions, "last" = last token only\n'
            '  "repeats": 1,\n'
            '  "prompts": [ {"prompt": "This is Sophia, she is a", "target": "woman"} ]   // target WITHOUT leading space\n}',
            language="json",
        )
        st.caption("A bare list of {prompt, target} objects also works. Each prompt may carry its own \"settings\" overrides. "
                   "The SAE layer comes from the sidebar.")

    up_spec = st.file_uploader("Upload a batch spec JSON (optional — replaces the text box)", type=["json"], key="batch_spec_upload")
    if up_spec is not None:
        try:
            _spec_text = up_spec.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            _spec_text = _spec_default
    else:
        _spec_text = st.text_area("Batch spec (JSON)", value=_spec_default, height=320, key="batch_spec_text")

    spec_ok = None
    try:
        spec_ok = parse_spec(json.loads(_spec_text))
    except Exception as e:
        st.error(f"Spec problem: {e}")

    if spec_ok:
        n_jobs = len(spec_ok["prompts"]) * len(spec_ok["modes"]) * spec_ok["repeats"]
        st.info(f"**{spec_ok['name']}** — {len(spec_ok['prompts'])} prompts × {len(spec_ok['modes'])} mode(s) × {spec_ok['repeats']} repeat(s) = **{n_jobs} runs** "
                f"on layer {layer} ({device.upper()}). Expect roughly 5–15 s per steerable run; runs where the target is already rank #1 take under a second.")
        tok_rows = []
        for p_ in spec_ok["prompts"]:
            n_tok = int(model.to_tokens(" " + p_["target"], prepend_bos=False).numel())
            tok_rows.append({
                "Prompt": p_["prompt"], "Target": p_["target"],
                "Target tokens": n_tok, "Token used": model.to_string([get_target_token_id(model, " " + p_["target"])]),
                "OK": "✅" if n_tok == 1 else "⚠️ multi-token (only the LAST piece would be scored)",
            })
        df_tok = pd.DataFrame(tok_rows)
        with st.expander("Target token check (before you run)", expanded=any(r["Target tokens"] != 1 for r in tok_rows)):
            st.dataframe(df_tok, use_container_width=True, hide_index=True)
        if any(r["Target tokens"] != 1 for r in tok_rows):
            st.warning("Some targets split into several GPT-2 tokens. Their results are recorded but excluded from the comparison; replace them with single-token words.")

    def _flatten_pairs(res):
        rows = []
        for r in res["paired_summary"]:
            row = {"Prompt": r["prompt"], "Target": r["target"], "Repeat": r["repeat"], "Baseline rank": r["baseline_rank"],
                   "Baseline prob %": None if r["baseline_prob_pct"] is None else round(r["baseline_prob_pct"], 2),
                   "Single token": r["target_single_token"], "Winner": r["winner"]}
            for key, lab in (("all", "All"), ("last", "Last")):
                v = r.get(key) or {}
                row[f"{lab}: final rank"] = v.get("final_rank")
                row[f"{lab}: success"] = v.get("success")
                row[f"{lab}: final prob %"] = None if v.get("final_prob_pct") is None else round(v["final_prob_pct"], 2)
                row[f"{lab}: steps"] = v.get("steps")
                row[f"{lab}: rounds"] = v.get("rounds")
                row[f"{lab}: features used"] = v.get("features_at_best")
                row[f"{lab}: round-0 candidates"] = v.get("round0_candidates_available")
                row[f"{lab}: round-0 safe mute/boost"] = (f"{v.get('round0_safe_mute')}/{v.get('round0_safe_boost')}" if v.get("round0_safe_mute") is not None else None)
                row[f"{lab}: time s"] = v.get("time_s")
                row[f"{lab}: stop reason"] = v.get("stop_reason")
            rows.append(row)
        return pd.DataFrame(rows)

    def render_batch_result(res, key_prefix):
        meta, agg = res["meta"], res.get("aggregate", {})
        st.subheader("Result")
        st.caption(f"{meta.get('name')} — {meta.get('total_runs')} runs, layer {meta.get('layer')}, {meta.get('elapsed_s')} s, "
                   f"{meta.get('errors', 0)} error(s). Started {meta.get('started')}, finished {meta.get('finished')}.")
        if agg:
            m1, m2, m3, m4 = st.columns(4)
            wc = agg.get("winner_counts", {})
            m1.metric("Steerable pairs", agg.get("steerable_pairs (baseline rank > 1, single-token target, both modes ran)"))
            m2.metric("Wins: all / last / tie", f"{wc.get('all', 0)} / {wc.get('last', 0)} / {wc.get('tie', 0)}")
            sa, sl = agg.get("success_rate_all"), agg.get("success_rate_last")
            m3.metric("Reached rank #1 (all)", "n/a" if sa is None else f"{sa*100:.0f}%")
            m4.metric("Reached rank #1 (last)", "n/a" if sl is None else f"{sl*100:.0f}%")
            with st.expander("Aggregate numbers"):
                st.json(agg)
        df = _flatten_pairs(res)
        st.dataframe(df, use_container_width=True, hide_index=True)

        valid = df[(df["Baseline rank"] > 1) & (df["Single token"] != False)]
        if not valid.empty:
            long_rows = []
            for _, r_ in valid.iterrows():
                long_rows += [{"Prompt": r_["Prompt"], "Series": "Baseline", "Rank": r_["Baseline rank"]},
                              {"Prompt": r_["Prompt"], "Series": "All positions (final)", "Rank": r_["All: final rank"]},
                              {"Prompt": r_["Prompt"], "Series": "Last token (final)", "Rank": r_["Last: final rank"]}]
            chart = alt.Chart(pd.DataFrame(long_rows)).mark_bar().encode(
                y=alt.Y("Prompt:N", sort=None, title=None),
                x=alt.X("Rank:Q", scale=alt.Scale(type="symlog"), title="Target rank (lower is better, log scale)"),
                yOffset="Series:N", color=alt.Color("Series:N", scale=alt.Scale(range=["#9ca3af", "#2563eb", "#ea580c"])),
                tooltip=["Prompt", "Series", "Rank"],
            ).properties(height=max(240, 34 * len(valid)))
            st.altair_chart(chart, use_container_width=True)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button("📥 Download FULL batch result (JSON)", json.dumps(res, indent=2, default=str).encode("utf-8"),
                               file_name=f"batch_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json",
                               key=f"{key_prefix}_dl_full")
        with d2:
            st.download_button("📥 Download paired summary (CSV)", df.to_csv(index=False).encode("utf-8"),
                               file_name="batch_paired_summary.csv", mime="text/csv", key=f"{key_prefix}_dl_csv")

        st.markdown("---")
        st.subheader("Inspect one run")
        runs = res["runs"]
        labels = [f"#{r['batch_index']} · {r['candidate_source']} · {r['prompt'][:50]} → {r['target']} (repeat {r['repeat']})" for r in runs]
        pick = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i], key=f"{key_prefix}_pick")
        rec = runs[pick]
        if rec.get("error"):
            st.error(rec["error"])
            st.code(rec.get("traceback", ""))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baseline rank", f"#{rec['baseline_rank']}", help=f"top-1 {rec['baseline_top1']} · target prob {rec['baseline_target_prob']}")
            c2.metric("Best rank", f"#{rec['best_result']['rank']}", delta=f"{rec['baseline_rank'] - rec['best_result']['rank']:+d}")
            c3.metric("Steps / rounds", f"{len(rec['hybrid_details'])} / {len(rec['rounds'])}")
            c4.metric("Time", f"{rec['total_time_s']} s")
            st.write(f"**Stop reason:** {rec['stop_reason']} · **Final top-1:** `{rec['final_top1']}` at {rec['final_target_prob']} · **Success:** {rec['success']}")
            st.write("**Baseline top-5**")
            st.table(rec["baseline_top5"])
            for rd in rec.get("round_details", []):
                with st.expander(f"Round {rd['round']} — pools and sweep ({len(rd['steps'])} steps)"):
                    st.write(f"Blocking token: `{rd['blocking_token']['token']}` ({rd['blocking_token']['prob_pct']:.2f}%). "
                             f"{rd['n_active_distinct']} distinct active features, {rd['available_unapplied']} unapplied; Top N {rd['top_n_requested']} → {rd['top_n_effective']} evaluated per side.")
                    st.write("Active features per prompt token: " + " · ".join(f"`{t['token']}` **{t['n_active']}**" for t in rd["active_features_per_token"]))
                    st.table(rd["pool_table"])
                    if rd["overlap"]:
                        st.write("Overlap assigned to one side")
                        st.dataframe(pd.DataFrame(rd["overlap"]), use_container_width=True, hide_index=True)
                    if rd["retested_features"]:
                        st.write("Previously rejected, re-tested")
                        st.dataframe(pd.DataFrame(rd["retested_features"]), use_container_width=True, hide_index=True)
                    if rd["steps"]:
                        st.write("Sweep steps")
                        st.dataframe(pd.DataFrame([{
                            "Step": s_["step"], "Mutes": s_["mute_batch_size"], "Boosts": s_["boost_batch_size"], "Target rank": s_["target_rank"],
                            "Target prob %": round(s_["target_prob_pct"], 3), "Top-1": s_["new_top1"], "New best": s_["is_new_best"],
                            "vs round baseline": s_["rank_change_vs_round_baseline"], "vs clean prompt": s_["rank_change_vs_clean_prompt"],
                            "New blockers": len(s_["combination_safety_check"].get("new_blockers") or []), "Note": s_["note"],
                        } for s_ in rd["steps"]]), use_container_width=True, hide_index=True)
            if rec.get("rank_progression"):
                st.write("Rank progression")
                st.dataframe(pd.DataFrame(rec["rank_progression"]).set_index("Step"), use_container_width=True)
            if rec.get("event_log"):
                st.write("Event log")
                for line in rec["event_log"]:
                    st.markdown(f"- {line}")
            st.write("Timing")
            st.json(rec["timing"])
            with st.expander("Raw JSON for this run"):
                st.json(rec, expanded=False)

    run_col, note_col = st.columns([1, 3])
    with run_col:
        go = st.button("▶ Run batch", key="btn_run_batch", disabled=spec_ok is None, type="primary")
    with note_col:
        st.caption("Results are auto-saved after every run to `outputs/batch_runs/`, so a refresh or crash won't lose finished runs. "
                   "Keep this tab open until it finishes.")

    if go and spec_ok:
        out_path = _os.path.join("outputs", "batch_runs", f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        prog = st.progress(0.0, text="Starting…")
        live = st.empty()
        live_rows = []

        def _cb(done, total, rec_):
            best = (rec_.get("best_result") or {}).get("rank")
            live_rows.append({
                "#": done, "Mode": rec_.get("candidate_source"), "Prompt": rec_.get("prompt"), "Target": rec_.get("target"),
                "Baseline rank": rec_.get("baseline_rank"), "Final rank": best,
                "Success": rec_.get("success"), "Steps": len(rec_.get("hybrid_details") or []),
                "Time s": rec_.get("total_time_s"), "Error": rec_.get("error"),
            })
            prog.progress(done / total, text=f"{done}/{total} runs finished")
            live.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)

        with st.spinner("Running batch…"):
            batch_res = run_batch(model, sae, hook_name, layer, device, spec_ok, progress_cb=_cb, save_path=out_path)
        prog.progress(1.0, text="Done")
        st.session_state["batch_result"] = batch_res
        st.session_state["batch_result_path"] = out_path
        st.success(f"Batch finished. Full result auto-saved to `{out_path}`.")

    if st.session_state.get("batch_result"):
        st.markdown("---")
        st.caption(f"Latest batch in this session (also on disk: `{st.session_state.get('batch_result_path')}`)")
        render_batch_result(st.session_state["batch_result"], "cur")

    st.markdown("---")
    up_res = st.file_uploader("Or open a previously saved batch result JSON", type=["json"], key="batch_result_upload")
    if up_res is not None:
        try:
            _old = json.load(up_res)
            if isinstance(_old, dict) and "runs" in _old:
                render_batch_result(_old, "old")
            else:
                st.error("That file is not a batch result (no 'runs' list).")
        except Exception as e:
            st.error(f"Could not open that file: {e}")
