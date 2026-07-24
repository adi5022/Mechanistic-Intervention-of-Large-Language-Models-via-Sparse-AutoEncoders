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
)

from src.hooks import (
    make_ablation_hook,
    make_joint_ablation_hook,
    make_signed_ablation_hook,
)


st.set_page_config(page_title="FeatureScalpel — Experimentation Bench", layout="wide")

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

# Tabs setup (6 tabs including Session History)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Single-Trace Iterative Ablation (Mute)",
    "Compound Batch Test (Mute)",
    "Target Feature Boost (Amplify)",
    "Hybrid Mute & Boost (Dual)",
    "Safety-Filtered Ablation (Filter)",
    "Session History & Benchmarks"
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
                        st.write(f"❌ Feature `{fid}` skipped: harms target token by `{t_delta*100:.2f}%`.")
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
                    st.write(f"❌ Feature `{fid}` excluded from batch candidate pool (harms target by `{t_delta*100:.2f}%`).")
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
                    st.write(f"❌ Competitor Feature `{fid}` excluded from mute pool (harms target token).")
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
                    st.write(f"❌ Target Feature `{fid}` excluded from boost pool (amplifies competitor, degrading target rank).")
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
                    skipped_log.append(f"✅ **Feature {fid}**: SAFE (target prob change: `+{t_delta*100:.2f}%`). Selected.")
                    break
                else:
                    skipped_log.append(f"❌ **Feature {fid}**: SKIPPED (harms target token: `{t_delta*100:.2f}%`).")
                    
            for log_entry in skipped_log:
                st.markdown(log_entry)
                
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

# --- TAB 6: Session History & Benchmarks ---
with tab6:
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
                for k in ["rounds_detail", "batch_details", "boost_details", "hybrid_details"]:
                    if k in rec:
                        detail_key = k
                        break
                
                if detail_key and rec[detail_key]:
                    st.markdown("**Detailed Steps / Variations:**")
                    for i, step in enumerate(rec[detail_key]):
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
