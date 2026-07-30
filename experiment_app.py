import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import json
from datetime import datetime

from src.sae_utils import load_model_and_sae
from src.editing import (
    get_target_token_id,
    get_top_competitor_features,
    get_top_target_features,
    check_target_safe,
    check_boost_safe,
    check_combination_safe,
    run_weighted_multi_competitor_reduction,
    run_weighted_multi_feature_competitor_reduction,
    make_weighted_ablation_hook,
)

from src.hooks import (
    make_ablation_hook,
    make_joint_ablation_hook,
    make_signed_ablation_hook,
)
from src.monosemanticity import (
    find_max_activating_examples,
    score_feature_interpretability,
    compute_sparsity_stats,
    compute_feature_similarity,
    find_most_similar_features,
    get_default_corpus,
)

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

# Layer selection with cached model loading
@st.cache_resource
def get_model_and_sae(layer: int):
    return load_model_and_sae(device="cpu", layer=layer)

layer = st.sidebar.selectbox("Select Model Layer", options=list(range(12)), index=8)

with st.spinner(f"Loading Model & SAE for Layer {layer}..."):
    model, sae = get_model_and_sae(layer)

hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Explainable AI Layer")
enable_xai = st.sidebar.checkbox("Enable AI Explanations (Groq)", value=False)
groq_key_input = st.sidebar.text_input("Groq API Key", type="password", value="")

# Tabs setup
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🧪 Single-trace iterative ablation",
    "📦 Compound batch test",
    "⚡ Target feature boost",
    "🔄 Hybrid mute and boost",
    "🛡️ Safety-filtered ablation",
    "📚 Monosemanticity Analysis",
    "📊 Session history and benchmarks",
    "⚖️ Weighted multi-competitor reduction",
    "🎛️ Weighted multi-feature competitor reduction",
    "🧠 Towards Monosemanticity"
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
        top_n_4 = st.number_input("Top N Candidate Features", value=30, min_value=1, step=1, key="t4_topn")
    with col2:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            strength_mute_4 = st.number_input("Mute Strength", value=0.3, step=0.1, key="t4_m_strength")
            mute_sizes_str_4 = st.text_input("Mute Batch Sizes", "1, 3, 5", key="t4_m_bs")
        with col_m2:
            strength_boost_4 = st.number_input("Boost Strength", value=0.5, step=0.1, key="t4_b_strength")
            boost_sizes_str_4 = st.text_input("Boost Batch Sizes", "1, 3, 5", key="t4_b_bs")
        use_safety_4 = st.checkbox("Enable Safety Filter (Target Protection)", value=True, key="t4_safety")


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
            
        target_str = target_4 if target_4.startswith(" ") else " " + target_4
        tokens = model.to_tokens(prompt_4)
        target_token_id = get_target_token_id(model, target_str)
        
        # Clean baseline pass
        model.reset_hooks()
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        current_top1_id = torch.argmax(probs).item()
        current_top1_str = model.to_string([current_top1_id])
        baseline_target_prob = probs[target_token_id].item()
        
        st.write(f"**Baseline Top-1:** `{current_top1_str}` | **Target '{target_str}' Prob:** `{baseline_target_prob*100:.2f}%`")
        
        # Candidate feature sets with safety filtering
        competitor_features = get_top_competitor_features(model, sae, prompt_4, current_top1_id, top_n=top_n_4)
        comp_ids = []
        for fid, _ in competitor_features:
            if use_safety_4:
                is_safe, t_delta = check_target_safe(model, sae, prompt_4, fid, target_token_id, strength=strength_mute_4)
                if is_safe:
                    comp_ids.append(fid)
                else:
                    st.markdown(f"❌ Competitor {make_feature_hover_link(fid, layer)} excluded from mute pool (harms target token).", unsafe_allow_html=True)
            else:
                comp_ids.append(fid)
        
        target_features = get_top_target_features(model, sae, prompt_4, target_token_id, top_n=top_n_4)
        target_ids = []
        for fid, _ in target_features:
            if use_safety_4:
                is_safe, t_delta, r_imp = check_boost_safe(model, sae, prompt_4, fid, target_token_id, strength=strength_boost_4)
                if is_safe:
                    target_ids.append(fid)
                else:
                    st.markdown(f"❌ Target {make_feature_hover_link(fid, layer)} excluded from boost pool (amplifies competitor, degrading target rank).", unsafe_allow_html=True)
            else:
                target_ids.append(fid)

        
        hybrid_details = []
        is_any_success = False
        best_final_top1 = current_top1_str
        best_final_target_prob = baseline_target_prob
        
        for m_n in mute_sizes_4:
            for b_n in boost_sizes_4:
                mute_batch = comp_ids[:m_n]
                boost_batch = target_ids[:b_n]
                
                model.reset_hooks()
                # Apply joint mute hook
                joint_mute_fn = make_joint_ablation_hook(mute_batch, sae, strength=strength_mute_4)
                model.add_hook(hook_name, joint_mute_fn)
                
                # Apply signed boost hook
                signed_boost_fn = make_signed_ablation_hook(boost_batch, sae, strength=+strength_boost_4)
                model.add_hook(hook_name, signed_boost_fn)
                
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
                
                st.subheader(f"Mute: {m_n} features (-{strength_mute_4}) | Boost: {b_n} features (+{strength_boost_4})")
                st.write(f"**Muted Features:** `{mute_batch}` | **Boosted Features:** `{boost_batch}`")
                
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
                
                # Run the whole-combination safety check
                safety_res = check_combination_safe(
                    model, sae, prompt_4,
                    mute_feature_ids=mute_batch, mute_strength=strength_mute_4,
                    boost_feature_ids=boost_batch, boost_strength=strength_boost_4,
                    target_token_id=target_token_id, top_k=10
                )
                
                # Display safety results
                st.write(f"**Combination Safety Check:** Target clean rank: `{safety_res['target_clean_rank']}` -> New rank: `{safety_res['target_new_rank']}`")
                if safety_res["new_blockers"]:
                    for blocker in safety_res["new_blockers"]:
                        c_rank = blocker["clean_rank_or_absent"]
                        was_str = f"was rank {c_rank}" if isinstance(c_rank, int) else "absent"
                        st.write(f"⚠️ New blocker: {blocker['token']!r} rose to rank {blocker['new_rank']} ({was_str} in clean baseline)")
                else:
                    st.write("✅ No new blockers detected")
                
                hybrid_details.append({
                    "mute_batch_size": m_n,
                    "mute_features": mute_batch,
                    "mute_strength": strength_mute_4,
                    "boost_batch_size": b_n,
                    "boost_features": boost_batch,
                    "boost_strength": strength_boost_4,
                    "new_top1": new_top1_str,
                    "target_prob": f"{target_prob*100:.2f}%",
                    "top5": table_data,
                    "combination_safety_check": safety_res
                })
                
        model.reset_hooks()
        
        # Save run to session history
        run_record = {
            "run_id": len(st.session_state["history"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "Hybrid Mute & Boost",
            "layer": layer,
            "prompt": prompt_4,
            "target": target_4,
            "mute_strength": strength_mute_4,
            "mute_sizes": mute_sizes_str_4,
            "boost_strength": strength_boost_4,
            "boost_sizes": boost_sizes_str_4,
            "top_n": top_n_4,
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
    st.header("Monosemanticity Analysis")
    st.markdown("Inspect whether a feature behaves like a coherent, interpretable concept using the core evaluation methodology from Anthropic’s monosemanticity work, applied to the already-loaded pretrained SAE.")

    col1, col2 = st.columns([1, 1])
    with col1:
        feature_id = st.number_input("Feature ID", min_value=0, value=313, step=1, key="mono_feature_id")
        corpus_source = st.radio("Corpus Source", options=["Bundled default", "Paste text"], index=0, key="mono_corpus_source")
    with col2:
        groq_key = st.text_input("Groq API Key (optional)", type="password", value="", key="mono_groq_key")
        use_groq = st.checkbox("Run autointerp scoring", value=False, key="mono_use_groq")

    if corpus_source == "Bundled default":
        corpus = get_default_corpus()
        st.caption(f"Using bundled corpus with {len(corpus)} sentences.")
    else:
        pasted = st.text_area("Paste a corpus, one sentence per line", height=220, key="mono_pasted_corpus")
        corpus = [line.strip() for line in pasted.splitlines() if line.strip()]
        if not corpus:
            st.info("Paste one or more sentences to analyze a custom corpus.")

    if st.button("Run Monosemanticity Analysis", key="btn_mono"):
        if not corpus:
            st.warning("No corpus supplied.")
            st.stop()

        with st.spinner("Scanning feature activation patterns..."):
            max_examples = find_max_activating_examples(model, sae, int(feature_id), corpus, top_n=10)

        st.subheader("1. Max-activating examples")
        if max_examples:
            rows = []
            for item in max_examples:
                rows.append({
                    "Activation": f"{item['activation']:.4f}",
                    "Token Position": item["token_position"],
                    "Token": item["token"],
                    "Snippet": item["text"],
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No activating examples found for this corpus.")

        if use_groq and groq_key:
            st.subheader("2. Interpretability score")
            with st.spinner("Running Groq-based autointerp scoring..."):
                score = score_feature_interpretability(model, sae, int(feature_id), corpus, groq_key, held_out_fraction=0.2)
            st.metric("Accuracy", f"{score['accuracy'] * 100:.1f}%")
            st.caption(f"Reference set size: {score['n_reference']}; Held-out set size: {score['n_held_out']}")
            if score["predictions"]:
                preview_rows = []
                for item in score["predictions"][:10]:
                    preview_rows.append({
                        "Prediction": item["predicted"],
                        "Actual": "HIGH" if item["actual_activation"] > 0.0 else "LOW",
                        "Actual Activation": f"{item['actual_activation']:.4f}",
                        "Text": item["text"],
                    })
                st.dataframe(preview_rows, use_container_width=True)

        st.subheader("3. Sparsity statistics")
        with st.spinner("Computing sparsity stats..."):
            sparsity = compute_sparsity_stats(model, sae, corpus, max_corpus_items=200)
        st.metric("Mean L0", f"{sparsity['mean_l0']:.4f}")
        st.caption(f"Corpus cap used: {sparsity['cap']} items")
        if sparsity["l0_distribution"]:
            st.bar_chart(pd.DataFrame({"L0": sparsity["l0_distribution"]}))
        if sparsity.get("feature_firing_frequency"):
            freq_rows = [{"Feature ID": fid, "Firing Frequency": f"{freq:.4f}"} for fid, freq in sparsity["feature_firing_frequency"].items()]
            st.dataframe(freq_rows, use_container_width=True)

        st.subheader("4. Most similar decoder directions")
        with st.spinner("Comparing decoder directions..."):
            similar = find_most_similar_features(sae, int(feature_id), top_n=10)
        if similar:
            sim_rows = [{"Feature ID": fid, "Cosine Similarity": f"{sim:.4f}"} for fid, sim in similar]
            st.dataframe(sim_rows, use_container_width=True)
        else:
            st.info("No similar features found.")

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
