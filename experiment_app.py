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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🧪 Single-trace iterative ablation",
    "📦 Compound batch test",
    "⚡ Target feature boost",
    "🔄 Hybrid mute and boost",
    "🛡️ Safety-filtered ablation",
    "📚 Monosemanticity Analysis",
    "📊 Session history and benchmarks",
    "⚖️ Weighted multi-competitor reduction",
    "🎛️ Weighted multi-feature competitor reduction",
    "🧠 Towards Monosemanticity",
    "⏱️ Sequential vs Batched Proof"
])



# --- TAB 1: Single-Trace Iterative Ablation ---
with tab1:
    st.header("Single-Trace Iterative Ablation")
    col1, col2 = st.columns(2)
    with col1:
        prompt_1 = st.text_input("Prompt", "Seiyu Group's headquarters are in", key="t1_prompt")
        target_1 = st.text_input("Target Completion", "Tokyo", key="t1_target")
    with col2:
        strength_1 = st.number_input("Ablation Strength", value=0.3, step=0.1, key="t1_strength")
        max_rounds_1 = st.number_input("Max Rounds", value=5, min_value=1, step=1, key="t1_rounds")
        top_n_1 = st.number_input("Top N Candidate Features", value=30, min_value=1, step=1, key="t1_topn")
        use_safety_1 = st.checkbox("Enable Safety Filter (Target Protection)", value=True, key="t1_safety")

        
    if st.button("Run Single Trace", key="btn_t1"):
        target_str = target_1 if target_1.startswith(" ") else " " + target_1
        tokens = model.to_tokens(prompt_1)
        target_token_id = get_target_token_id(model, target_str)
        
        features_ablated = []
        model.reset_hooks()
        
        st.write(f"**Target:** `{target_str}` (Token ID: `{target_token_id}`)")
        
        rounds_detail = []
        baseline_top1 = ""
        baseline_target_prob = 0.0
        final_top1 = ""
        final_target_prob = 0.0
        is_success = False
        
        for r in range(0, max_rounds_1 + 1):
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            top5_probs, top5_indices = torch.topk(probs, k=5)
            current_top1_id = top5_indices[0].item()
            current_top1_str = model.to_string([current_top1_id])
            target_prob = probs[target_token_id].item()
            
            if r == 0:
                baseline_top1 = current_top1_str
                baseline_target_prob = target_prob
                
            final_top1 = current_top1_str
            final_target_prob = target_prob
            
            st.subheader(f"Round {r}")
            st.write(f"**Ablated Features:** `{features_ablated if r > 0 else 'None (Clean Baseline)'}`")
            
            table_data = []
            for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                tok_str = model.to_string([idx.item()])
                is_target = "Yes (TARGET)" if idx.item() == target_token_id else "No"
                table_data.append({
                    "Rank": rank_idx,
                    "Token": tok_str,
                    "Token ID": idx.item(),
                    "Probability": f"{p.item()*100:.2f}%",
                    "Is Target": is_target
                })
            st.table(table_data)
            st.write(f"**Target '{target_str}' Probability:** `{target_prob*100:.2f}%`")
            
            rounds_detail.append({
                "round": r,
                "ablated_features": list(features_ablated),
                "top5": table_data,
                "target_prob": f"{target_prob*100:.2f}%"
            })
            
            if current_top1_id == target_token_id:
                is_success = True
                st.success(f"SUCCESS! Target token '{target_str}' reached Top-1 in Round {r}!")
                break
                
            if r == max_rounds_1:
                st.warning(f"REACHED MAX ROUNDS ({max_rounds_1}). Stopping trace.")
                break
                
            competitor_features = get_top_competitor_features(model, sae, prompt_1, current_top1_id, top_n=top_n_1)
            selected_feature = None
            selected_delta = None
            for fid, delta in competitor_features:
                if fid in features_ablated:
                    continue
                if use_safety_1:
                    is_safe, t_delta = check_target_safe(model, sae, prompt_1, fid, target_token_id, strength=strength_1)
                    if not is_safe:
                        st.markdown(f"❌ {make_feature_hover_link(fid, layer)} skipped: harms target token by `{t_delta*100:.2f}%`.", unsafe_allow_html=True)
                        continue
                selected_feature = fid
                selected_delta = delta
                break
                    
            if selected_feature is None:
                st.info("No more safe features found to ablate.")
                break

                
            features_ablated.append(selected_feature)
            st.write(f"**Selected Feature for Round {r+1}:** Feature `{selected_feature}` (Drops '{current_top1_str}' prob by `{selected_delta*100:.2f}%` at strength 1.0)")
            
            model.reset_hooks()
            for fid in features_ablated:
                hook_fn = make_ablation_hook(fid, sae, strength=strength_1)
                model.add_hook(hook_name, hook_fn)
                
        model.reset_hooks()
        
        # Save run to session history
        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Single-Trace Iterative Ablation",
            "layer": layer,
            "prompt": prompt_1,
            "target": target_1,
            "strength": strength_1,
            "max_rounds": max_rounds_1,
            "top_n": top_n_1,
            "baseline_top1": baseline_top1,
            "baseline_target_prob": f"{baseline_target_prob*100:.2f}%",
            "final_top1": final_top1,
            "final_target_prob": f"{final_target_prob*100:.2f}%",
            "success": is_success,
            "rounds_detail": rounds_detail
        }
        st.session_state["history"].append(run_record)
        st.success("Run saved to Session History!")

# --- TAB 2: Compound Batch Test ---
with tab2:
    st.header("Compound Batch Test")
    col1, col2 = st.columns(2)
    with col1:
        prompt_2 = st.text_input("Prompt", "Seiyu Group's headquarters are in", key="t2_prompt")
        target_2 = st.text_input("Target Completion", "Tokyo", key="t2_target")
    with col2:
        strength_2 = st.number_input("Ablation Strength", value=0.3, step=0.1, key="t2_strength")
        batch_sizes_str_2 = st.text_input("Batch Sizes (comma-separated)", "1, 3, 5, 10, 15, 20", key="t2_bs")
        top_n_2 = st.number_input("Top N Candidate Features", value=30, min_value=1, step=1, key="t2_topn")
        use_safety_2 = st.checkbox("Enable Safety Filter (Target Protection)", value=True, key="t2_safety")


    if st.button("Run Compound Batch Test", key="btn_t2"):
        try:
            batch_sizes_2 = [int(x.strip()) for x in batch_sizes_str_2.split(",") if x.strip()]
        except ValueError:
            batch_sizes_2 = [1, 3, 5, 10, 15, 20]
            st.warning("Invalid batch sizes format; using default [1, 3, 5, 10, 15, 20].")
            
        target_str = target_2 if target_2.startswith(" ") else " " + target_2
        tokens = model.to_tokens(prompt_2)
        target_token_id = get_target_token_id(model, target_str)
        
        model.reset_hooks()
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        current_top1_id = torch.argmax(probs).item()
        current_top1_str = model.to_string([current_top1_id])
        baseline_target_prob = probs[target_token_id].item()
        
        st.write(f"**Baseline Top-1:** `{current_top1_str}` | **Target '{target_str}' Prob:** `{baseline_target_prob*100:.2f}%`")
        
        competitor_features = get_top_competitor_features(model, sae, prompt_2, current_top1_id, top_n=top_n_2)
        safe_ranked_ids = []
        for fid, _ in competitor_features:
            if use_safety_2:
                is_safe, t_delta = check_target_safe(model, sae, prompt_2, fid, target_token_id, strength=strength_2)
                if is_safe:
                    safe_ranked_ids.append(fid)
                else:
                    st.markdown(f"❌ {make_feature_hover_link(fid, layer)} excluded from batch candidate pool (harms target by `{t_delta*100:.2f}%`).", unsafe_allow_html=True)
            else:
                safe_ranked_ids.append(fid)
        ranked_ids = safe_ranked_ids

        
        batch_details = []
        is_any_success = False
        best_final_top1 = current_top1_str
        best_final_target_prob = baseline_target_prob
        
        for n in batch_sizes_2:
            if n > len(ranked_ids):
                st.warning(f"Requested batch size {n} exceeds {len(ranked_ids)} available active features — using all {len(ranked_ids)} instead.")
            batch = ranked_ids[:n]
            
            model.reset_hooks()
            joint_hook_fn = make_joint_ablation_hook(batch, sae, strength=strength_2)
            model.add_hook(hook_name, joint_hook_fn)
            
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            top5_probs, top5_indices = torch.topk(probs, k=5)
            new_top1_id = top5_indices[0].item()
            new_top1_str = model.to_string([new_top1_id])
            target_prob = probs[target_token_id].item()
            
            if new_top1_id == target_token_id:
                is_any_success = True
            best_final_top1 = new_top1_str
            best_final_target_prob = target_prob
            
            st.subheader(f"Batch Size {n} (Features: {batch})")
            st.write(f"**New Top-1:** `{new_top1_str}` | **Target Prob:** `{target_prob*100:.2f}%`")
            
            table_data = []
            for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                tok_str = model.to_string([idx.item()])
                is_target = "Yes (TARGET)" if idx.item() == target_token_id else "No"
                table_data.append({
                    "Rank": rank_idx,
                    "Token": tok_str,
                    "Probability": f"{p.item()*100:.2f}%",
                    "Is Target": is_target
                })
            st.table(table_data)
            
            batch_details.append({
                "batch_size": n,
                "features_used": batch,
                "new_top1": new_top1_str,
                "target_prob": f"{target_prob*100:.2f}%",
                "top5": table_data
            })
            
        model.reset_hooks()
        
        # Save run to session history
        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Compound Batch Test",
            "layer": layer,
            "prompt": prompt_2,
            "target": target_2,
            "strength": strength_2,
            "batch_sizes": batch_sizes_str_2,
            "top_n": top_n_2,
            "baseline_top1": current_top1_str,
            "baseline_target_prob": f"{baseline_target_prob*100:.2f}%",
            "final_top1": best_final_top1,
            "final_target_prob": f"{best_final_target_prob*100:.2f}%",
            "success": is_any_success,
            "batch_details": batch_details
        }
        st.session_state["history"].append(run_record)
        st.success("Run saved to Session History!")

# --- TAB 3: Target Feature Boost ---
with tab3:
    st.header("Target Feature Boost")
    col1, col2 = st.columns(2)
    with col1:
        prompt_3 = st.text_input("Prompt", "Seiyu Group's headquarters are in", key="t3_prompt")
        target_3 = st.text_input("Target Completion", "Tokyo", key="t3_target")
    with col2:
        boost_strengths_str_3 = st.text_input("Boost Strengths (comma-separated)", "0.3, 0.5, 0.7, 1.0", key="t3_strengths")
        batch_sizes_str_3 = st.text_input("Batch Sizes (comma-separated)", "1, 3, 5", key="t3_bs")
        top_n_3 = st.number_input("Top N Candidate Features", value=30, min_value=1, step=1, key="t3_topn")
        use_safety_3 = st.checkbox("Enable Safety Filter (Amplification Protection)", value=True, key="t3_safety")


    if st.button("Run Target Boost Test", key="btn_t3"):
        try:
            boost_strengths_3 = [float(x.strip()) for x in boost_strengths_str_3.split(",") if x.strip()]
        except ValueError:
            boost_strengths_3 = [0.3, 0.5, 0.7, 1.0]
            st.warning("Invalid boost strengths; using default [0.3, 0.5, 0.7, 1.0].")

        try:
            batch_sizes_3 = [int(x.strip()) for x in batch_sizes_str_3.split(",") if x.strip()]
        except ValueError:
            batch_sizes_3 = [1, 3, 5]
            st.warning("Invalid batch sizes; using default [1, 3, 5].")
            
        target_str = target_3 if target_3.startswith(" ") else " " + target_3
        tokens = model.to_tokens(prompt_3)
        target_token_id = get_target_token_id(model, target_str)
        
        model.reset_hooks()
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        current_top1_id = torch.argmax(probs).item()
        current_top1_str = model.to_string([current_top1_id])
        baseline_target_prob = probs[target_token_id].item()
        
        st.write(f"**Baseline Top-1:** `{current_top1_str}` | **Target '{target_str}' Prob:** `{baseline_target_prob*100:.2f}%`")
        
        target_features = get_top_target_features(model, sae, prompt_3, target_token_id, top_n=top_n_3)
        safe_target_ids = []
        for fid, _ in target_features:
            if use_safety_3:
                first_strength = boost_strengths_3[0] if boost_strengths_3 else 0.5
                is_safe, t_delta, r_imp = check_boost_safe(model, sae, prompt_3, fid, target_token_id, strength=first_strength)
                if is_safe:
                    safe_target_ids.append(fid)
                else:
                    st.write(f"❌ Target Feature `{fid}` excluded from boost pool (amplifies competitor, degrading target rank).")
            else:
                safe_target_ids.append(fid)
        ranked_ids = safe_target_ids

        
        boost_details = []
        is_any_success = False
        best_final_top1 = current_top1_str
        best_final_target_prob = baseline_target_prob
        
        for strength in boost_strengths_3:
            for n in batch_sizes_3:
                if n > len(ranked_ids):
                    st.warning(f"Requested batch size {n} exceeds {len(ranked_ids)} available active features — using all {len(ranked_ids)} instead.")
                batch = ranked_ids[:n]
                
                model.reset_hooks()
                signed_hook_fn = make_signed_ablation_hook(batch, sae, strength=strength)
                model.add_hook(hook_name, signed_hook_fn)
                
                with torch.no_grad():
                    logits = model(tokens)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                top5_probs, top5_indices = torch.topk(probs, k=5)
                new_top1_id = top5_indices[0].item()
                new_top1_str = model.to_string([new_top1_id])
                target_prob = probs[target_token_id].item()
                
                if new_top1_id == target_token_id:
                    is_any_success = True
                best_final_top1 = new_top1_str
                best_final_target_prob = target_prob
                
                st.subheader(f"Boost Strength +{strength} | Batch Size {n} (Features: {batch})")
                st.write(f"**New Top-1:** `{new_top1_str}` | **Target Prob:** `{target_prob*100:.2f}%`")
                
                table_data = []
                for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                    tok_str = model.to_string([idx.item()])
                    is_target = "Yes (TARGET)" if idx.item() == target_token_id else "No"
                    table_data.append({
                        "Rank": rank_idx,
                        "Token": tok_str,
                        "Probability": f"{p.item()*100:.2f}%",
                        "Is Target": is_target
                    })
                st.table(table_data)
                
                boost_details.append({
                    "boost_strength": strength,
                    "batch_size": n,
                    "features_used": batch,
                    "new_top1": new_top1_str,
                    "target_prob": f"{target_prob*100:.2f}%",
                    "top5": table_data
                })
                
        model.reset_hooks()
        
        # Save run to session history
        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Target Feature Boost",
            "layer": layer,
            "prompt": prompt_3,
            "target": target_3,
            "boost_strengths": boost_strengths_str_3,
            "batch_sizes": batch_sizes_str_3,
            "top_n": top_n_3,
            "baseline_top1": current_top1_str,
            "baseline_target_prob": f"{baseline_target_prob*100:.2f}%",
            "final_top1": best_final_top1,
            "final_target_prob": f"{best_final_target_prob*100:.2f}%",
            "success": is_any_success,
            "boost_details": boost_details
        }
        st.session_state["history"].append(run_record)
        st.success("Run saved to Session History!")

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
        }
        st.session_state["history"].append(run_record)
        st.success("Run saved to Session History!")

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

# --- TAB 5: Safety-Filtered Ablation (Filter) ---
with tab5:
    st.header("Safety-Filtered Iterative Ablation")
    st.markdown("Performs round-by-round competitor ablation, but **safety-checks** each candidate feature to ensure it does not hurt the target token's probability. Features that harm the target are skipped.")
    
    col1, col2 = st.columns(2)
    with col1:
        prompt_5 = st.text_input("Prompt", "Seiyu Group's headquarters are in", key="t5_prompt")
        target_5 = st.text_input("Target Completion", "Tokyo", key="t5_target")
    with col2:
        strength_5 = st.number_input("Ablation Strength", value=0.3, step=0.1, key="t5_strength")
        max_rounds_5 = st.number_input("Max Rounds", value=5, min_value=1, step=1, key="t5_rounds")
        top_n_5 = st.number_input("Top N Candidate Features", value=30, min_value=1, step=1, key="t5_topn")

    if st.button("Run Safety-Filtered Trace", key="btn_t5"):
        target_str = target_5 if target_5.startswith(" ") else " " + target_5
        tokens = model.to_tokens(prompt_5)
        target_token_id = get_target_token_id(model, target_str)
        
        features_ablated = []
        model.reset_hooks()
        
        st.write(f"**Target:** `{target_str}` (Token ID: `{target_token_id}`)")
        
        rounds_detail = []
        baseline_top1 = ""
        baseline_target_prob = 0.0
        final_top1 = ""
        final_target_prob = 0.0
        is_success = False
        
        for r in range(0, max_rounds_5 + 1):
            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            top5_probs, top5_indices = torch.topk(probs, k=5)
            current_top1_id = top5_indices[0].item()
            current_top1_str = model.to_string([current_top1_id])
            target_prob = probs[target_token_id].item()
            
            if r == 0:
                baseline_top1 = current_top1_str
                baseline_target_prob = target_prob
                
            final_top1 = current_top1_str
            final_target_prob = target_prob
            
            st.subheader(f"Round {r}")
            st.write(f"**Active Ablated Features:** `{features_ablated if r > 0 else 'None (Clean Baseline)'}`")
            
            table_data = []
            for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                tok_str = model.to_string([idx.item()])
                is_target = "Yes (TARGET)" if idx.item() == target_token_id else "No"
                table_data.append({
                    "Rank": rank_idx,
                    "Token": tok_str,
                    "Token ID": idx.item(),
                    "Probability": f"{p.item()*100:.2f}%",
                    "Is Target": is_target
                })
            st.table(table_data)
            st.write(f"**Target '{target_str}' Probability:** `{target_prob*100:.2f}%`")
            
            rounds_detail.append({
                "round": r,
                "ablated_features": list(features_ablated),
                "top5": table_data,
                "target_prob": f"{target_prob*100:.2f}%"
            })
            
            if current_top1_id == target_token_id:
                is_success = True
                st.success(f"SUCCESS! Target token '{target_str}' reached Top-1 in Round {r}!")
                break
                
            if r == max_rounds_5:
                st.warning(f"REACHED MAX ROUNDS ({max_rounds_5}). Stopping trace.")
                break
                
            competitor_features = get_top_competitor_features(model, sae, prompt_5, current_top1_id, top_n=top_n_5)
            
            selected_feature = None
            selected_delta = None
            skipped_log = []
            
            for fid, delta in competitor_features:
                if fid in features_ablated:
                    continue
                
                # Safety check on feature
                is_safe, t_delta = check_target_safe(model, sae, prompt_5, fid, target_token_id, strength=strength_5)
                
                if is_safe:
                    selected_feature = fid
                    selected_delta = delta
                    skipped_log.append(f"✅ **{make_feature_hover_link(fid, layer)}**: SAFE (target prob change: `+{t_delta*100:.2f}%`). Selected.")
                    break
                else:
                    skipped_log.append(f"❌ **{make_feature_hover_link(fid, layer)}**: SKIPPED (harms target token: `{t_delta*100:.2f}%`).")
                    
            for log_entry in skipped_log:
                st.markdown(log_entry, unsafe_allow_html=True)
                
            if selected_feature is None:
                st.info("No more safe competitor features available to ablate.")
                break
                
            features_ablated.append(selected_feature)
            
            model.reset_hooks()
            for fid in features_ablated:
                hook_fn = make_ablation_hook(fid, sae, strength=strength_5)
                model.add_hook(hook_name, hook_fn)
                
        model.reset_hooks()
        
        # Save run to session history
        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Safety-Filtered Ablation",
            "layer": layer,
            "prompt": prompt_5,
            "target": target_5,
            "strength": strength_5,
            "max_rounds": max_rounds_5,
            "top_n": top_n_5,
            "baseline_top1": baseline_top1,
            "baseline_target_prob": f"{baseline_target_prob*100:.2f}%",
            "final_top1": final_top1,
            "final_target_prob": f"{final_target_prob*100:.2f}%",
            "success": is_success,
            "rounds_detail": rounds_detail
        }
        st.session_state["history"].append(run_record)
        st.success("Run saved to Session History!")

# --- TAB 8: Weighted Multi-Competitor Reduction ---
with tab8:
    st.header("Weighted Multi-Competitor Reduction")
    st.markdown("Identifies all tokens ranked above the target and weakens their principal driving features in proportion to their threat level (probability).")
    
    col1, col2 = st.columns(2)
    with col1:
        prompt_7 = st.text_input("Prompt", "The location of Massachusetts Institute of Technology is in", key="t7_prompt")
        target_7 = st.text_input("Target Completion", "Cambridge", key="t7_target")
    with col2:
        max_strength_7 = st.slider("Max Mute Strength", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="t7_max_strength")
        top_n_7 = st.number_input("Top N Candidate Features", value=20, min_value=1, step=1, key="t7_topn")
        
    if st.button("Run Weighted Reduction", key="btn_t7"):
        target_str = target_7 if target_7.startswith(" ") else " " + target_7
        tokens = model.to_tokens(prompt_7)
        target_token_id = get_target_token_id(model, target_str)
        
        with st.spinner("Running weighted multi-competitor reduction..."):
            res = run_weighted_multi_competitor_reduction(
                model, sae, prompt_7, target_token_id,
                max_strength=max_strength_7, top_n_candidates=int(top_n_7)
            )
            
        st.subheader("Target Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Target Clean Rank", value=res["target_clean_rank"])
            st.metric(label="Target Clean Prob", value=f"{res['target_clean_prob']*100:.4f}%")
        with res_col2:
            st.metric(label="Target New Rank", value=res["target_new_rank"])
            st.metric(label="Target New Prob", value=f"{res['target_new_prob']*100:.4f}%")
            
        if res["is_safe"]:
            st.success("✅ Combination is SAFE: Target rank did not regress and no new blockers detected.")
        else:
            st.warning("⚠️ Combination is UNSAFE: Target rank regressed or new blockers detected.")
            
        st.subheader("Competitor Features & Weights")
        comp_rows = []
        for comp in res["competitors"]:
            weight = comp["probability"] / sum(c["probability"] for c in res["competitors"]) if res["competitors"] else 0.0
            comp_rows.append({
                "Competitor Token": comp["token"],
                "Baseline Prob": f"{comp['probability']*100:.2f}%",
                "Normalized Weight": f"{weight*100:.2f}%",
                "Driving Feature ID": comp["top_feature"],
                "Mute Strength": f"{res['feature_to_strength'].get(comp['top_feature'], 0.0):.4f}"
            })
        if comp_rows:
            df_comp = pd.DataFrame(comp_rows)
            df_comp["Driving Feature ID"] = df_comp["Driving Feature ID"].apply(
                lambda fid: make_feature_hover_link(fid, layer) if fid is not None else "None"
            )
            st.markdown(df_comp.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("No competitors ranked above the target.")
            
        if res["new_blockers"]:
            st.subheader("⚠️ New Blocker Tokens")
            blocker_data = []
            for blocker in res["new_blockers"]:
                c_prob = f"{blocker['clean_prob_or_absent']*100:.2f}%" if isinstance(blocker['clean_prob_or_absent'], float) else blocker['clean_prob_or_absent']
                blocker_data.append({
                    "Token": blocker["token"],
                    "Baseline Prob": c_prob,
                    "Baseline Rank": blocker["clean_rank_or_absent"],
                    "New Prob": f"{blocker['new_prob']*100:.2f}%",
                    "New Rank": blocker["new_rank"]
                })
            st.table(blocker_data)
        else:
            st.write("✅ No new blocker tokens detected.")
            
        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Weighted Multi-Competitor Reduction",
            "layer": layer,
            "prompt": prompt_7,
            "target": target_7,
            "max_strength": max_strength_7,
            "top_n": top_n_7,
            "baseline_top1": res["competitors"][0]["token"] if res["competitors"] else target_str,
            "baseline_target_prob": f"{res['target_clean_prob']*100:.2f}%",
            "final_top1": res["competitors"][0]["token"] if res["competitors"] else target_str,
            "final_target_prob": f"{res['target_new_prob']*100:.2f}%",
            "success": res["is_safe"],
            "weighted_reduction_details": res
        }
        st.session_state["history"].append(run_record)
        st.success("Run saved to Session History!")

        if enable_xai:
            st.markdown("---")
            st.subheader("🤖 Explainable AI Analysis")
            with st.spinner("Generating mechanistic explanation..."):
                from src.explain import generate_mechanistic_explanation
                interventions = []
                for fid, strength in res["feature_to_strength"].items():
                    interventions.append({
                        "feature_id": fid,
                        "description": get_neuronpedia_explanation(fid, layer),
                        "action": "muted",
                        "strength": strength
                    })
                explanation = generate_mechanistic_explanation(
                    prompt=prompt_7,
                    target=target_7,
                    baseline_prob=res["target_clean_prob"],
                    baseline_rank=res["target_clean_rank"],
                    final_prob=res["target_new_prob"],
                    final_rank=res["target_new_rank"],
                    interventions=interventions,
                    api_key=groq_key_input
                )
                st.info(explanation, icon=":material/psychology:")

# --- TAB 9: Weighted Multi-Feature Competitor Reduction ---
with tab9:
    st.header("Weighted Multi-Feature Competitor Reduction")
    st.markdown("Distributes competitor weights across their Top-K causal features and aggregates them to test distributed representations.")
    
    col1, col2 = st.columns(2)
    with col1:
        prompt_8 = st.text_input("Prompt", "The location of Massachusetts Institute of Technology is in", key="t8_prompt")
        target_8 = st.text_input("Target Completion", "Cambridge", key="t8_target")
        max_strength_8 = st.slider("Max Mute Strength", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="t8_max_strength")
    with col2:
        top_n_8 = st.number_input("Top N Candidate Features", value=20, min_value=1, step=1, key="t8_topn")
        top_k_8 = st.number_input("Top-K Features Per Competitor", value=3, min_value=1, step=1, key="t8_topk")
        weighting_method_8 = st.selectbox("Weighting Method", options=["equal", "delta_normalized", "softmax"], index=1, key="t8_weighting_method")
        
        softmax_temp_8 = 1.0
        if weighting_method_8 == "softmax":
            softmax_temp_8 = st.number_input("Softmax Temperature", value=1.0, step=0.1, key="t8_temp")
            
    if st.button("Run Multi-Feature Reduction", key="btn_t8"):
        target_str = target_8 if target_8.startswith(" ") else " " + target_8
        tokens = model.to_tokens(prompt_8)
        target_token_id = get_target_token_id(model, target_str)
        
        with st.spinner("Running weighted multi-feature competitor reduction..."):
            res = run_weighted_multi_feature_competitor_reduction(
                model, sae, prompt_8, target_token_id,
                max_strength=max_strength_8, top_n_candidates=int(top_n_8),
                top_k_features_per_competitor=int(top_k_8),
                feature_weighting_method=weighting_method_8,
                softmax_temperature=softmax_temp_8
            )
            
        st.subheader("Target Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Target Clean Rank", value=res["target_clean_rank"])
            st.metric(label="Target Clean Prob", value=f"{res['target_clean_prob']*100:.4f}%")
        with res_col2:
            st.metric(label="Target New Rank", value=res["target_new_rank"])
            st.metric(label="Target New Prob", value=f"{res['target_new_prob']*100:.4f}%")
            
        if res["is_safe"]:
            st.success("✅ Combination is SAFE: Target rank did not regress and no new blockers detected.")
        else:
            st.warning("⚠️ Combination is UNSAFE: Target rank regressed or new blockers detected.")
            
        st.subheader("Per-Competitor Breakdown")
        for comp in res["competitors"]:
            comp_weight = comp["probability"] / sum(c["probability"] for c in res["competitors"]) if res["competitors"] else 0.0
            with st.expander(f"Competitor '{comp['token']}' | Prob: {comp['probability']*100:.2f}% | Comp Weight: {comp_weight:.4f}"):
                feat_rows = []
                for feat in comp.get("features", []):
                    feat_rows.append({
                        "Feature ID": feat["feature_id"],
                        "Delta Prob": f"{feat['delta']:.6f}",
                        "Within-Competitor Weight": f"{feat['within_competitor_weight']:.4f}",
                        "Joint Weight": f"{feat['joint_weight']:.4f}"
                    })
                if feat_rows:
                    df_feat = pd.DataFrame(feat_rows)
                    df_feat["Feature ID"] = df_feat["Feature ID"].apply(
                        lambda fid: make_feature_hover_link(fid, layer) if fid is not None else "None"
                    )
                    st.markdown(df_feat.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.write("No features driving this competitor.")

        st.subheader("Merged Feature Table")
        merged_rows = []
        for fid, strength in res["feature_to_strength"].items():
            contributors = res["feature_to_contributors"].get(fid, [])
            sat_info = res["saturated"].get(fid, {"saturated": False, "before": strength, "after": strength})
            merged_rows.append({
                "Feature ID": fid,
                "Number of Contributors": len(contributors),
                "Final Accumulated Strength": f"{strength:.6f}",
                "Saturated": "True" if sat_info["saturated"] else "False",
                "Raw Strength (Before Clamp)": f"{sat_info['before']:.6f}"
            })
        if merged_rows:
            df_merged = pd.DataFrame(merged_rows)
            df_merged["Feature ID"] = df_merged["Feature ID"].apply(
                lambda fid: make_feature_hover_link(fid, layer) if fid is not None else "None"
            )
            st.markdown(df_merged.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("No muted features.")
            
        st.subheader("Contributor Details per Merged Feature")
        for fid, contributors in res["feature_to_contributors"].items():
            with st.expander(f"Feature {fid} | {len(contributors)} Contributor(s)"):
                contrib_rows = []
                for c in contributors:
                    contrib_rows.append({
                        "Competitor Token": c["token"],
                        "Competitor Probability": f"{c['prob']*100:.2f}%",
                        "Competitor Weight": f"{c['competitor_weight']:.4f}",
                        "Feature Weight (Within-Comp)": f"{c['feature_weight']:.4f}",
                        "Joint Weight": f"{c['joint_weight']:.4f}",
                        "Contribution Strength": f"{c['contrib_strength']:.6f}"
                    })
                st.table(contrib_rows)

        if res["new_blockers"]:
            st.subheader("⚠️ New Blocker Tokens")
            blocker_data = []
            for blocker in res["new_blockers"]:
                c_prob = f"{blocker['clean_prob_or_absent']*100:.2f}%" if isinstance(blocker['clean_prob_or_absent'], float) else blocker['clean_prob_or_absent']
                blocker_data.append({
                    "Token": blocker["token"],
                    "Baseline Prob": c_prob,
                    "Baseline Rank": blocker["clean_rank_or_absent"],
                    "New Prob": f"{blocker['new_prob']*100:.2f}%",
                    "New Rank": blocker["new_rank"]
                })
            st.table(blocker_data)
        else:
            st.write("✅ No new blocker tokens detected.")

        # Display Resulting Top Predictions (Resulting Top-5 Table)
        model.reset_hooks()
        weighted_hook = make_weighted_ablation_hook(res["feature_to_strength"], sae)
        model.add_hook(hook_name, weighted_hook)
        with torch.no_grad():
            new_logits = model(tokens)
            new_probs = F.softmax(new_logits[0, -1, :], dim=-1)
        model.reset_hooks()
        new_sorted_indices = torch.argsort(new_probs, descending=True)
        table_data = []
        for rank_idx, idx in enumerate(new_sorted_indices[:5], 1):
            tok_str = model.to_string([idx.item()])
            tok_prob = new_probs[idx].item()
            table_data.append({
                "Rank": rank_idx,
                "Prediction": tok_str,
                "Probability": f"{tok_prob*100:.2f}%"
            })
        st.subheader("Post-Intervention Predictions")
        st.table(table_data)

        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Weighted Multi-Feature Competitor Reduction",
            "layer": layer,
            "prompt": prompt_8,
            "target": target_8,
            "max_strength": max_strength_8,
            "top_n": top_n_8,
            "top_k": top_k_8,
            "feature_weighting_method": weighting_method_8,
            "softmax_temperature": softmax_temp_8,
            "baseline_top1": res["competitors"][0]["token"] if res["competitors"] else target_str,
            "baseline_target_prob": f"{res['target_clean_prob']*100:.2f}%",
            "final_top1": table_data[0]["Prediction"] if table_data else target_str,
            "final_target_prob": f"{res['target_new_prob']*100:.2f}%",
            "success": res["is_safe"],
            "weighted_reduction_details": res
        }
        st.session_state["history"].append(run_record)
        st.success("Run saved to Session History!")

        if enable_xai:
            st.markdown("---")
            st.subheader("🤖 Explainable AI Analysis")
            with st.spinner("Generating mechanistic explanation..."):
                from src.explain import generate_mechanistic_explanation
                interventions = []
                for fid, strength in res["feature_to_strength"].items():
                    interventions.append({
                        "feature_id": fid,
                        "description": get_neuronpedia_explanation(fid, layer),
                        "action": "muted",
                        "strength": strength
                    })
                explanation = generate_mechanistic_explanation(
                    prompt=prompt_8,
                    target=target_8,
                    baseline_prob=res["target_clean_prob"],
                    baseline_rank=res["target_clean_rank"],
                    final_prob=res["target_new_prob"],
                    final_rank=res["target_new_rank"],
                    interventions=interventions,
                    api_key=groq_key_input
                )
                st.info(explanation, icon=":material/psychology:")

# --- TAB 10: Towards Monosemanticity Base Paper Implementation ---
with tab10:
    st.header("Towards Monosemanticity — Base Paper Baseline Implementation")
    st.markdown(
        """
        Direct implementation of the 3 fundamental diagnostic measurements from **Anthropic (Bricken et al., 2023)**:
        1. **Feature Activation Spectrum** ($f(x) = \\text{ReLU}(W_{dec}^T(x - b_{dec}) + b_{enc})$)
        2. **Direct Logit Attribution** ($f_i \\cdot (W_{dec}[:, i] \\cdot W_U)$)
        3. **SAE Feature Activation Clamping** ($f_i \\leftarrow 0$ or $f_i \\leftarrow C$)
        """
    )
    
    st.markdown("---")
    
    mono_prompt = st.text_input("Input Prompt", "The location of Massachusetts Institute of Technology is in", key="base_paper_prompt")
    
    if st.button("Run Base Paper Diagnostics", key="btn_base_paper_diag"):
        tokens = model.to_tokens(mono_prompt)
        
        # 1. Clean forward pass & SAE encoding
        model.reset_hooks()
        with torch.no_grad():
            clean_logits, cache = model.run_with_cache(tokens)
            clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
            
            resid_pre = cache[hook_name] # (batch, seq, d_model)
            sae_acts = sae(resid_pre[0, -1, :]) # (d_sae,)
            
        active_mask = sae_acts > 0
        l0_norm = active_mask.sum().item()
        
        st.subheader("1. Feature Activation Spectrum (Sparsity Measurement)")
        st.metric(label="Active Features Count (L0 Norm)", value=l0_norm, delta=f"out of {sae.cfg.d_sae} total features")
        
        top_acts, top_fids = torch.topk(sae_acts, k=10)
        
        feat_spectrum = []
        for fid, act in zip(top_fids.tolist(), top_acts.tolist()):
            feat_spectrum.append({
                "Feature ID": fid,
                "Raw Activation f_i": f"{act:.4f}",
                "Description": get_neuronpedia_explanation(fid, layer)
            })
        df_spectrum = pd.DataFrame(feat_spectrum)
        df_spectrum["Feature ID"] = df_spectrum["Feature ID"].apply(lambda fid: make_feature_hover_link(fid, layer))
        st.markdown(df_spectrum.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        st.session_state["base_paper_top_fids"] = top_fids.tolist()
        st.session_state["base_paper_top_acts"] = top_acts.tolist()

    # --- Step 2 & 3: Logit Attribution & Feature Clamping ---
    if "base_paper_top_fids" in st.session_state:
        st.markdown("---")
        st.subheader("2. Direct Logit Attribution (W_dec · W_U)")
        st.markdown("Measures the exact mathematical projection of an SAE feature's decoder vector onto the model's vocabulary unembedding matrix $W_U$.")
        
        selected_fid = st.selectbox(
            "Select Feature to Inspect Logit Attribution & Clamp",
            options=st.session_state["base_paper_top_fids"],
            key="base_selected_fid"
        )
        
        # Calculate W_dec[:, fid] @ W_U
        # sae.W_dec is (d_sae, d_model), model.W_U is (d_model, d_vocab)
        w_dec_i = sae.W_dec[selected_fid, :] # (d_model,)
        with torch.no_grad():
            logit_promotions = torch.matmul(w_dec_i, model.W_U) # (d_vocab,)
            
        top_positive_vals, top_positive_ids = torch.topk(logit_promotions, k=5)
        top_negative_vals, top_negative_ids = torch.topk(logit_promotions, k=5, largest=False)
        
        col_attr1, col_attr2 = st.columns(2)
        with col_attr1:
            st.markdown("##### ⬆️ Tokens Promoted by Feature (Positive Logit Weight)")
            pos_rows = []
            for tok_id, val in zip(top_positive_ids.tolist(), top_positive_vals.tolist()):
                pos_rows.append({"Token": model.to_string([tok_id]), "Logit Weight Contribution": f"+{val:.4f}"})
            st.table(pos_rows)
            
        with col_attr2:
            st.markdown("##### ⬇️ Tokens Suppressed by Feature (Negative Logit Weight)")
            neg_rows = []
            for tok_id, val in zip(top_negative_ids.tolist(), top_negative_vals.tolist()):
                neg_rows.append({"Token": model.to_string([tok_id]), "Logit Weight Contribution": f"{val:.4f}"})
            st.table(neg_rows)

        st.markdown("---")
        st.subheader("3. SAE Activation Clamping (Base Paper Steering)")
        
        c1, c2 = st.columns(2)
        with c1:
            clamp_mode = st.radio("Clamp Mode", options=["Ablate (Clamp f_i = 0)", "Excitatory Pinch (Clamp f_i = C)"], key="base_clamp_mode")
        with c2:
            clamp_val = st.slider("Clamping Value (C)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, key="base_clamp_val")
            
        if st.button("Run Exact Paper Clamping", key="btn_run_paper_clamp"):
            tokens = model.to_tokens(mono_prompt)
            target_val = 0.0 if clamp_mode == "Ablate (Clamp f_i = 0)" else clamp_val
            
            # SAE Space Clamping Hook
            def sae_clamp_hook(module, input, output):
                # input[0] or output is residual stream tensor (batch, seq, d_model)
                # Apply SAE encoding, modify feature, decode back
                x = output[0] if isinstance(output, tuple) else output
                # Compute feature acts
                acts = sae(x[0, -1, :])
                current_act = acts[selected_fid].item()
                # Difference between target clamped val and current act
                diff = target_val - current_act
                # Modify residual vector by diff * W_dec[selected_fid]
                x[0, -1, :] = x[0, -1, :] + diff * sae.W_dec[selected_fid, :]
                return output
                
            model.reset_hooks()
            with torch.no_grad():
                clean_logits = model(tokens)
                clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)
                
            # Add hook
            model.add_hook(hook_name, sae_clamp_hook)
            with torch.no_grad():
                steered_logits = model(tokens)
                steered_probs = F.softmax(steered_logits[0, -1, :], dim=-1)
            model.reset_hooks()
            
            st.markdown("#### Clamping Logit & Probability Impact")
            
            clean_top5_indices = torch.topk(clean_probs[0, -1, :], k=5).indices.tolist()
            steered_top5_indices = torch.topk(steered_probs[0, -1, :], k=5).indices.tolist()
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.markdown("##### Clean Top 5 Predictions")
                clean_table = []
                for rank, idx in enumerate(clean_top_k_indices if 'clean_top_k_indices' in locals() else clean_top5_indices, 1):
                    clean_table.append({"Rank": rank, "Token": model.to_string([idx]), "Probability": f"{clean_probs[0, -1, idx].item()*100:.2f}%"})
                st.table(clean_table)
                
            with c_res2:
                st.markdown("##### Steered Top 5 Predictions")
                steered_table = []
                for rank, idx in enumerate(steered_top5_indices, 1):
                    steered_table.append({"Rank": rank, "Token": model.to_string([idx]), "Probability": f"{steered_probs[0, -1, idx].item()*100:.2f}%"})
                st.table(steered_table)

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

    st.header("Session History & Benchmarks")

    # File uploader to load past session JSONs
    uploaded_file = st.file_uploader("Upload a past session JSON to view/analyze", type=["json"], key="history_uploader")
    uploaded_history = None
    if uploaded_file is not None:
        try:
            uploaded_history = json.load(uploaded_file)
            st.success(f"Successfully loaded session with {len(uploaded_history)} runs from uploaded file.")
        except Exception as e:
            st.error(f"Error loading JSON file: {e}")

    # Fallback to st.session_state["history"] if no file is uploaded
    history_to_show = uploaded_history if uploaded_history is not None else st.session_state.get("history", [])

    if not history_to_show:
        st.info("No runs logged in this session yet. Run tests in other tabs, or upload a previously downloaded session JSON file.")
    else:
        st.subheader("Summary Table of Session Runs")
        summary_rows = []
        for rec in history_to_show:
            summary_rows.append({
                "Run ID": rec["run_id"],
                "Timestamp": rec["timestamp"],
                "Mode": rec["mode"],
                "Layer": rec["layer"],
                "Prompt": rec["prompt"],
                "Target": rec["target"],
                "Baseline Top-1": rec["baseline_top1"],
                "Baseline Target Prob": rec["baseline_target_prob"],
                "Final Top-1": rec["final_top1"],
                "Final Target Prob": rec["final_target_prob"],
                "Success": "TRUE" if rec["success"] else "FALSE"
            })
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_data = df_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Session Summary (CSV)",
                data=csv_data,
                file_name="session_benchmark_summary.csv",
                mime="text/csv"
            )
        with col_d2:
            json_data = json.dumps(history_to_show, indent=2).encode('utf-8')
            st.download_button(
                label="📥 Download Detailed Trace Session (JSON)",
                data=json_data,
                file_name="session_benchmark_full_details.json",
                mime="application/json"
            )
            
        if uploaded_history is None:
            if st.button("🗑️ Clear Session History"):
                st.session_state["history"] = []
                st.rerun()

        st.markdown("---")
        st.subheader("Formatted Session Viewer")
        for rec in reversed(history_to_show):
            with st.container(border=True):
                st.markdown(f"### Run #{rec['run_id']} — **{rec['mode']}** (Layer {rec['layer']})")
                st.write(f"**Prompt:** `{rec['prompt']}` | **Target:** `{rec['target']}` | **Time:** `{rec['timestamp']}`")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Baseline State**")
                    st.write(f"Top-1: `{rec['baseline_top1']}`")
                    st.write(f"Target Prob: `{rec['baseline_target_prob']}`")
                with c2:
                    st.markdown("**Final State**")
                    st.write(f"Top-1: `{rec['final_top1']}`")
                    st.write(f"Target Prob: `{rec['final_target_prob']}`")
                with c3:
                    st.markdown("**Status**")
                    if rec.get("success"):
                        st.success("✅ SUCCESS")
                    else:
                        st.warning("⚠️ FAILURE / NOT MET")
                
                # Check for detail list
                detail_key = None
                for k in ["rounds_detail", "batch_details", "boost_details", "hybrid_details", "weighted_reduction_details"]:
                    if k in rec:
                        detail_key = k
                        break
                
                if detail_key and rec[detail_key]:
                    st.markdown("**Detailed Steps / Variations:**")
                    steps_list = rec[detail_key] if isinstance(rec[detail_key], list) else [rec[detail_key]]
                    for i, step in enumerate(steps_list):
                        if detail_key == "rounds_detail":
                            label = f"Round {step.get('round', i)} | Target Prob: {step.get('target_prob')}"
                            exp_desc = f"Ablated features: `{step.get('ablated_features', [])}`"
                        elif detail_key == "batch_details":
                            label = f"Batch Size {step.get('batch_size', i)} | Target Prob: {step.get('target_prob')}"
                            exp_desc = f"Features ablated: `{step.get('features_ablated', [])}`"
                        elif detail_key == "boost_details":
                            label = f"Strength +{step.get('boost_strength', '')} | Batch {step.get('batch_size', '')} | Target Prob: {step.get('target_prob')}"
                            exp_desc = f"Features boosted: `{step.get('features_used', [])}`"
                        elif detail_key == "hybrid_details":
                            label = f"Mute {step.get('mute_batch_size', '')} features | Boost {step.get('boost_batch_size', '')} features | Target Prob: {step.get('target_prob')}"
                            exp_desc = f"Muted: `{step.get('mute_features', [])}` | Boosted: `{step.get('boost_features', [])}`"
                        elif detail_key == "weighted_reduction_details":
                            label = f"Weighted Reduction | Clean Rank: {step.get('target_clean_rank')} -> New Rank: {step.get('target_new_rank')}"
                            exp_desc = f"Max Strength: {rec.get('max_strength')} | Muted features: `{step.get('feature_to_strength', {})}`"
                        else:
                            label = f"Step {i}"
                            exp_desc = ""
                            
                        with st.expander(label):
                            if exp_desc:
                                st.write(exp_desc)
                            if "top5" in step and isinstance(step["top5"], list):
                                st.table(step["top5"])
                            if "combination_safety_check" in step:
                                sc = step["combination_safety_check"]
                                st.write(f"**Combination Safety Check:** Target clean rank: `{sc.get('target_clean_rank')}` -> New rank: `{sc.get('target_new_rank')}`")
                                if sc.get("new_blockers"):
                                    for blocker in sc["new_blockers"]:
                                        c_rank = blocker.get("clean_rank_or_absent")
                                        was_str = f"was rank {c_rank}" if isinstance(c_rank, int) else "absent"
                                        st.write(f"⚠️ New blocker: {blocker.get('token')!r} rose to rank {blocker.get('new_rank')} ({was_str} in clean baseline)")
                                else:
                                    st.write("✅ No new blockers detected")

        st.markdown("---")
        st.subheader("Detailed Run Inspector")
        for rec in reversed(history_to_show):
            with st.expander(f"Run #{rec['run_id']} — {rec['mode']} (Layer {rec['layer']}) | Prompt: '{rec['prompt']}' | Target: '{rec['target']}'"):
                st.write(f"**Timestamp:** `{rec['timestamp']}`")
                st.write(f"**Baseline Top-1:** `{rec['baseline_top1']}` | **Baseline Target Prob:** `{rec['baseline_target_prob']}`")
                st.write(f"**Final Top-1:** `{rec['final_top1']}` | **Final Target Prob:** `{rec['final_target_prob']}`")
                st.write(f"**Success Status:** `{'Success' if rec['success'] else 'Failure'}`")
                st.json(rec)

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
