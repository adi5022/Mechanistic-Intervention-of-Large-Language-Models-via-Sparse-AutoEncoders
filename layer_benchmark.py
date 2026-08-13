"""
Layer Intervention Benchmark UI
Dedicated Streamlit application for characterizing how Hybrid Mute & Boost intervention
effectiveness changes across transformer layer depth across single or multiple prompt-target pairs.
"""

import streamlit as st
import pandas as pd
import json
import os
import torch
from src.sae_utils import load_base_model, load_sae_for_layer, load_model_and_sae
from src.benchmark.layer_benchmark_runner import run_layer_benchmark

# 1. Page Config
st.set_page_config(
    page_title="Layer Intervention Benchmark",
    page_icon=":material/analytics:",
    layout="wide"
)

# 2. Caching Strategy
@st.cache_resource
def get_cached_base_model(device: str = "cuda"):
    """Cached loader for GPT-2 base model on specified device."""
    return load_base_model(device=device)

@st.cache_resource
def get_cached_sae(layer: int, device: str = "cuda"):
    """Cached loader for Layer-specific SAE on specified device."""
    return load_sae_for_layer(layer=layer, device=device)

def get_cached_model_and_sae(layer: int, device: str = "cuda"):
    """Composes cached GPT-2 base model and cached layer SAE on specified device."""
    model = get_cached_base_model(device=device)
    sae = get_cached_sae(layer=layer, device=device)
    return model, sae


def main():
    st.title("Layer Intervention Benchmark")
    st.markdown("Characterize how **Hybrid Mute & Boost** intervention effectiveness varies across transformer layer depth across multiple prompt datasets.")

    # Initialize prompts dataset in session state if missing
    if "prompts_dataset" not in st.session_state:
        st.session_state["prompts_dataset"] = [
            {"prompt": "The location of Massachusetts Institute of Technology is in", "target": "Cambridge"}
        ]

    # 3. Sidebar inputs & Prompts Dataset Editor
    st.sidebar.header("Configuration")

    # Hardware / Compute Device Selection
    st.sidebar.subheader("Compute Device Target")
    device_options = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_options.append(f"⚡ GPU ({gpu_name})")
    device_options.append("💻 CPU Host")

    selected_device_label = st.sidebar.radio(
        "Select Compute Hardware:",
        options=device_options,
        index=0,
        help="Switch between GPU and CPU execution to compare benchmark processing speeds."
    )
    selected_device = "cuda" if "GPU" in selected_device_label else "cpu"

    st.sidebar.subheader("Prompts Dataset")

    # BATCH JSON DATASET UPLOADER & COPY/PASTE INPUT
    with st.sidebar.expander("📁 Batch Import Prompts (Upload / Copy-Paste JSON)", expanded=False):
        st.markdown("**Option 1: Copy-Paste JSON String**")
        pasted_json_str = st.text_area(
            "Paste JSON dataset string",
            placeholder='[\n  {"prompt": "The capital of France is", "target": "Paris"}\n]',
            height=110,
            key="pasted_json_input"
        )
        if pasted_json_str.strip():
            try:
                p_data = json.loads(pasted_json_str)
                if isinstance(p_data, list):
                    parsed_p = p_data
                elif isinstance(p_data, dict) and "prompts" in p_data:
                    parsed_p = p_data["prompts"]
                else:
                    parsed_p = []

                valid_paste_items = [
                    {"prompt": str(item.get("prompt", "")), "target": str(item.get("target", ""))}
                    for item in parsed_p
                    if isinstance(item, dict) and "prompt" in item and "target" in item
                ]

                if valid_paste_items:
                    if st.button(f"📋 Import {len(valid_paste_items)} Prompts from Text"):
                        st.session_state["prompts_dataset"] = valid_paste_items
                        st.success(f"Successfully imported {len(valid_paste_items)} prompt(s)!")
                        st.rerun()
                else:
                    st.warning("Pasted JSON must contain objects with 'prompt' and 'target' fields.")
            except Exception as e:
                st.error(f"Invalid JSON string: {e}")

        st.markdown("---")
        st.markdown("**Option 2: Upload JSON File**")
        uploaded_file = st.file_uploader("Upload JSON Dataset File", type=["json"], key="batch_json_uploader")
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    parsed_prompts = data
                elif isinstance(data, dict) and "prompts" in data:
                    parsed_prompts = data["prompts"]
                else:
                    parsed_prompts = []

                valid_items = [
                    {"prompt": str(item.get("prompt", "")), "target": str(item.get("target", ""))}
                    for item in parsed_prompts
                    if isinstance(item, dict) and "prompt" in item and "target" in item
                ]

                if valid_items:
                    if st.button(f"📥 Import {len(valid_items)} Prompts from File"):
                        st.session_state["prompts_dataset"] = valid_items
                        st.success(f"Successfully imported {len(valid_items)} prompt(s)!")
                        st.rerun()
                else:
                    st.warning("Uploaded JSON must contain objects with 'prompt' and 'target' fields.")
            except Exception as e:
                st.error(f"Error parsing JSON file: {e}")

        # Sample Template Download
        sample_json_template = json.dumps([
            {"prompt": "The location of Massachusetts Institute of Technology is in", "target": "Cambridge"},
            {"prompt": "The capital of France is", "target": "Paris"},
            {"prompt": "The currency used in Japan is", "target": "Yen"}
        ], indent=4)

        st.download_button(
            label="📥 Download Sample Prompts JSON Template",
            data=sample_json_template,
            file_name="sample_prompts_dataset.json",
            mime="application/json",
            key="dl_sample_template"
        )

    st.sidebar.markdown(f"**Configured Prompts ({len(st.session_state['prompts_dataset'])} total)**")
    
    # Render prompt-target input rows
    for idx, item in enumerate(st.session_state["prompts_dataset"]):
        with st.sidebar.expander(f"Prompt #{idx + 1}", expanded=(idx < 3)):
            p_val = st.text_area(
                "Prompt",
                value=item["prompt"],
                key=f"prompt_{idx}",
                height=70
            )
            t_val = st.text_input(
                "Target Completion",
                value=item["target"],
                key=f"target_{idx}"
            )
            st.session_state["prompts_dataset"][idx]["prompt"] = p_val
            st.session_state["prompts_dataset"][idx]["target"] = t_val

    col_add, col_rem = st.sidebar.columns(2)
    with col_add:
        if st.sidebar.button("➕ Add Prompt"):
            st.session_state["prompts_dataset"].append({"prompt": "", "target": ""})
            st.rerun()
    with col_rem:
        if len(st.session_state["prompts_dataset"]) > 1:
            if st.sidebar.button("➖ Remove Prompt"):
                st.session_state["prompts_dataset"].pop()
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Intervention Parameters")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        mute_strength = st.sidebar.number_input("Mute Strength", value=0.3, step=0.1)
        mute_batch_size = st.sidebar.number_input("Mute Batch Size", value=3, min_value=1, step=1)
    with col2:
        boost_strength = st.sidebar.number_input("Boost Strength", value=0.5, step=0.1)
        boost_batch_size = st.sidebar.number_input("Boost Batch Size", value=3, min_value=1, step=1)

    use_safety = st.sidebar.checkbox(
        "Enable Safety Filter (Target Protection)",
        value=True,
        help="Excludes features that harm target token probability or rank when muted/boosted individually."
    )

    layer_options = list(range(12))
    selected_layers = st.sidebar.multiselect(
        "Layers to Benchmark",
        options=layer_options,
        default=[2, 5, 8, 10]
    )

    if st.sidebar.button("Run Layer Benchmark", type="primary"):
        # Validate prompts dataset
        valid_pairs = [
            p for p in st.session_state["prompts_dataset"]
            if p["prompt"].strip() and p["target"].strip()
        ]
        
        if not valid_pairs:
            st.error("Please provide at least one valid prompt and target completion pair.")
            return
        if not selected_layers:
            st.error("Please select at least one layer to benchmark.")
            return

        total_prompts = len(valid_pairs)
        total_layers = len(selected_layers)

        st.markdown("### Execution Progress")
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        # Detailed Stage Progress Callback
        def layer_callback(p_idx, total_p, l_idx, total_l, current_layer, prompt_text, current_stage_text):
            frac = ((l_idx - 1) + (p_idx / total_p)) / total_l
            progress_bar.progress(min(max(frac, 0.0), 1.0))
            status_box.markdown(
                f"⏳ **Layer {current_layer}** ({l_idx} / {total_l}) | "
                f"**Prompt {p_idx} / {total_p}** (`{prompt_text[:30]}...`) | **Stage:** `{current_stage_text}`"
            )

        benchmark_output = run_layer_benchmark(
            prompts=valid_pairs,
            layers=sorted(selected_layers),
            mute_strength=mute_strength,
            boost_strength=boost_strength,
            mute_batch_size=mute_batch_size,
            boost_batch_size=boost_batch_size,
            use_safety=use_safety,
            algorithm="hybrid",
            model_sae_loader=lambda l: get_cached_model_and_sae(layer=l, device=selected_device),
            results_dir="benchmark_results",
            layer_callback=layer_callback
        )

        progress_bar.progress(1.0)
        status_box.success(f"✅ Completed benchmark across {total_prompts} prompt(s)! Saved to single artifact.")
        st.session_state["consolidated_benchmark_results"] = benchmark_output

    # 4. Display Results
    if "consolidated_benchmark_results" in st.session_state and st.session_state["consolidated_benchmark_results"]:
        master_results = st.session_state["consolidated_benchmark_results"]
        prompt_runs = master_results.get("prompts", [])
        
        st.markdown("---")
        st.header("Benchmark Results & Research Artifacts")

        device_str = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU (torch+cpu)"
        # Experiment Metadata Display
        st.caption(
            f"**Model:** `{master_results.get('model', 'gpt2')}` | "
            f"**Device:** `{device_str}` | "
            f"**SAE Release:** `{master_results.get('sae_release', 'gpt2-small-res-jb')}` | "
            f"**Hook:** `{master_results.get('hook_location', 'hook_resid_pre')}` | "
            f"**Safety:** `{'Enabled' if master_results.get('safety_enabled', True) else 'Disabled'}` | "
            f"**Version:** `{master_results.get('benchmark_version', '0.3')}`"
        )

        if "saved_filepath" in master_results:
            st.info(f"💾 Master Benchmark JSON automatically saved to: `{master_results['saved_filepath']}`")

        prompt_options = [
            f"Prompt #{i+1}: '{r['prompt'][:35]}...' → Target: '{r['target']}'"
            for i, r in enumerate(prompt_runs)
        ]

        selected_prompt_idx = st.selectbox(
            "Select Benchmark Prompt to Inspect:",
            options=list(range(len(prompt_runs))),
            format_func=lambda i: prompt_options[i]
        )

        selected_run = prompt_runs[selected_prompt_idx]
        layers_data = selected_run.get("layers", [])

        # Formatted results DataFrame
        table_rows = []
        chart_rows = []
        profile_rows = []

        for r in layers_data:
            table_rows.append({
                "Layer": f"Layer {r['layer']}",
                "Clean Rank": r["clean_rank"],
                "Final Rank": r["final_rank"],
                "Rank Improvement": r["rank_improvement"],
                "Clean Probability": f"{r['clean_probability']*100:.3f}%",
                "Final Probability": f"{r['final_probability']*100:.3f}%",
                "Probability Gain": f"{r['probability_gain']*100:.3f}%",
                "Runtime (ms)": f"{r['runtime_ms']:.1f}",
                "Success": "Yes" if r["success"] else "No"
            })
            chart_rows.append({
                "Layer": f"Layer {r['layer']}",
                "Probability Gain": r["probability_gain"],
                "Rank Improvement": r["rank_improvement"],
                "Runtime (ms)": r["runtime_ms"]
            })
            prof = r.get("profile", {})
            tot_ms = prof.get("total_layer_ms", 1.0)
            if tot_ms <= 0:
                tot_ms = 1.0

            profile_rows.append({
                "Layer": f"Layer {r['layer']}",
                "SAE Load (ms / %)": f"{prof.get('sae_loading_ms', 0.0):.1f} ms ({prof.get('sae_loading_ms', 0.0)/tot_ms*100.0:.1f}%)",
                "Clean Baseline (ms / %)": f"{prof.get('clean_baseline_ms', 0.0):.1f} ms ({prof.get('clean_baseline_ms', 0.0)/tot_ms*100.0:.1f}%)",
                "Feature Select (ms / %)": f"{prof.get('feature_selection_ms', 0.0):.1f} ms ({prof.get('feature_selection_ms', 0.0)/tot_ms*100.0:.1f}%)",
                "Safety Filter (ms / %)": f"{prof.get('safety_filtering_ms', 0.0):.1f} ms ({prof.get('safety_filtering_ms', 0.0)/tot_ms*100.0:.1f}%)",
                "Intervention (ms / %)": f"{prof.get('intervention_ms', 0.0):.1f} ms ({prof.get('intervention_ms', 0.0)/tot_ms*100.0:.1f}%)",
                "Total (ms)": f"{tot_ms:.1f} ms"
            })

        df_table = pd.DataFrame(table_rows)
        df_chart = pd.DataFrame(chart_rows)
        df_profile = pd.DataFrame(profile_rows)

        # Summary Top Cards Metrics
        st.subheader("Benchmark Execution Summary")
        total_time_sec = sum(r["runtime_ms"] for r in layers_data) / 1000.0
        avg_time_ms = total_time_sec * 1000.0 / len(layers_data) if layers_data else 0.0
        max_prob_gain = max(r["probability_gain"] for r in layers_data) * 100.0 if layers_data else 0.0
        best_layer = max(layers_data, key=lambda x: x["probability_gain"])["layer"] if layers_data else 0

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Layer Execution Time", f"{total_time_sec:.2f} s")
        m_col2.metric("Avg Time per Layer", f"{avg_time_ms:.1f} ms")
        m_col3.metric("Peak Target Gain", f"+{max_prob_gain:.2f}%")
        m_col4.metric("Best Performing Layer", f"Layer {best_layer}")

        st.subheader(f"Results for Prompt #{selected_prompt_idx + 1}")
        st.caption(f"Prompt: `{selected_run.get('prompt', '')}` | Target: `{selected_run.get('target', '')}`")
        
        st.dataframe(df_table, hide_index=True)

        # 1. Expandable Execution Cost / Forward-Pass Accounting
        with st.expander("🔍 Execution Cost / Forward-Pass Accounting", expanded=True):
            fwd_rows = []
            cnt_rows = []
            for r in layers_data:
                fwd = r.get("forward_passes", {})
                cnt = r.get("counts", {})
                fwd_rows.append({
                    "Layer": f"Layer {r['layer']}",
                    "Clean Baseline": fwd.get("clean_baseline", 1),
                    "Candidate Screening": fwd.get("candidate_screening", 2),
                    "Competitor Ranking": fwd.get("competitor_ranking", 31),
                    "Target Ranking": fwd.get("target_ranking", 31),
                    "Competitor Safety": fwd.get("competitor_safety", 30),
                    "Target Safety": fwd.get("target_safety", 30),
                    "Final Intervention": fwd.get("final_intervention", 1),
                    "Total Model Forwards": fwd.get("total_model_forwards", 127)
                })
                cnt_rows.append({
                    "Layer": f"Layer {r['layer']}",
                    "Competitor Candidates": cnt.get("competitor_candidates_evaluated", 30),
                    "Target Candidates": cnt.get("target_candidates_evaluated", 30),
                    "Competitor Safety Checks": cnt.get("competitor_safety_checks", 30),
                    "Target Safety Checks": cnt.get("target_safety_checks", 30),
                    "Selected Mute Features": cnt.get("selected_mute_features_count", len(r.get("mute_features", []))),
                    "Selected Boost Features": cnt.get("selected_boost_features_count", len(r.get("boost_features", [])))
                })

            st.markdown("**Model Forward Passes Breakdown per Layer:**")
            st.dataframe(pd.DataFrame(fwd_rows), hide_index=True)

            st.markdown("**Candidates & Safety Evaluation Counts per Layer:**")
            st.dataframe(pd.DataFrame(cnt_rows), hide_index=True)

        # 2. Display Stage Profiling Breakdown Table with Timing & Percentage Highlights
        with st.expander("⏱️ Detailed Stage Timing Breakdown (Stage vs Time)", expanded=True):
            st.markdown("**Stage-by-Stage Execution Times for Selected Layer:**")
            
            stage_table_rows = []
            for r in layers_data:
                prof = r.get("profile", {})
                tot_ms = prof.get("total_layer_ms", 1.0)
                sae_ms = prof.get("sae_loading_ms", 0.0)
                clean_ms = prof.get("clean_baseline_ms", 0.0)
                feat_ms = prof.get("feature_selection_ms", 0.0)
                safe_ms = prof.get("safety_filtering_ms", 0.0)
                int_ms = prof.get("intervention_ms", 0.0)

                # Format in seconds / ms cleanly like research journal
                stage_table_rows.append({
                    "Layer": f"Layer {r['layer']}",
                    "SAE Loading": f"{sae_ms/1000.0:.2f} s ({sae_ms:.1f} ms)",
                    "Clean Baseline": f"{clean_ms/1000.0:.3f} s ({clean_ms:.1f} ms)",
                    "Feature Selection": f"{feat_ms/1000.0:.3f} s ({feat_ms:.1f} ms)",
                    "Safety Filtering": f"{safe_ms/1000.0:.3f} s ({safe_ms:.1f} ms)",
                    "Final Intervention": f"{int_ms/1000.0:.3f} s ({int_ms:.1f} ms)",
                    "Total Layer Time": f"{tot_ms/1000.0:.2f} s ({tot_ms:.1f} ms)"
                })

            st.dataframe(pd.DataFrame(stage_table_rows), hide_index=True)
            
            # Show single vertical Stage vs Time breakdown for primary layer
            if layers_data:
                first_r = layers_data[0]
                f_prof = first_r.get("profile", {})
                f_tot = f_prof.get("total_layer_ms", 1.0)
                vertical_stage_data = [
                    {"Stage": "SAE Loading", "Time (seconds)": f"{f_prof.get('sae_loading_ms', 0.0)/1000.0:.3f} s", "Time (ms)": f"{f_prof.get('sae_loading_ms', 0.0):.1f} ms", "% of Total": f"{f_prof.get('sae_loading_ms', 0.0)/f_tot*100.0:.1f}%"},
                    {"Stage": "Clean Baseline", "Time (seconds)": f"{f_prof.get('clean_baseline_ms', 0.0)/1000.0:.3f} s", "Time (ms)": f"{f_prof.get('clean_baseline_ms', 0.0):.1f} ms", "% of Total": f"{f_prof.get('clean_baseline_ms', 0.0)/f_tot*100.0:.1f}%"},
                    {"Stage": "Feature Selection", "Time (seconds)": f"{f_prof.get('feature_selection_ms', 0.0)/1000.0:.3f} s", "Time (ms)": f"{f_prof.get('feature_selection_ms', 0.0):.1f} ms", "% of Total": f"{f_prof.get('feature_selection_ms', 0.0)/f_tot*100.0:.1f}%"},
                    {"Stage": "Safety Filtering", "Time (seconds)": f"{f_prof.get('safety_filtering_ms', 0.0)/1000.0:.3f} s", "Time (ms)": f"{f_prof.get('safety_filtering_ms', 0.0):.1f} ms", "% of Total": f"{f_prof.get('safety_filtering_ms', 0.0)/f_tot*100.0:.1f}%"},
                    {"Stage": "Final Intervention", "Time (seconds)": f"{f_prof.get('intervention_ms', 0.0)/1000.0:.3f} s", "Time (ms)": f"{f_prof.get('intervention_ms', 0.0):.1f} ms", "% of Total": f"{f_prof.get('intervention_ms', 0.0)/f_tot*100.0:.1f}%"},
                    {"Stage": "Total Layer Execution", "Time (seconds)": f"{f_tot/1000.0:.3f} s", "Time (ms)": f"{f_tot:.1f} ms", "% of Total": "100.0%"}
                ]
                st.markdown(f"**Stage vs Time Breakdown for Layer {first_r['layer']}:**")
                st.table(pd.DataFrame(vertical_stage_data))

            st.caption("💡 *Note: Step 2 Optimization is active. Redundant clean-baseline model forward passes in safety filtering have been removed.*")

        # Bar charts
        st.subheader("Performance Visualizations")
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown("**Probability Gain vs Layer**")
            st.bar_chart(df_chart, x="Layer", y="Probability Gain")

        with col_c2:
            st.markdown("**Rank Improvement vs Layer**")
            st.bar_chart(df_chart, x="Layer", y="Rank Improvement")

        with col_c3:
            st.markdown("**Runtime (ms) vs Layer**")
            st.bar_chart(df_chart, x="Layer", y="Runtime (ms)")

        # Benchmark Artifact Visibility & Download Workflow
        st.markdown("---")
        st.subheader("Benchmark Artifacts & Export Manager")

        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            if "saved_filepath" in master_results:
                st.markdown(f"**Saved Filepath:** `{master_results['saved_filepath']}`")
                st.caption(f"Total Prompts: {master_results.get('total_prompts', 1)} | Timestamp: `{master_results.get('timestamp')}`")
        with col_d2:
            master_json_str = json.dumps(master_results, indent=4)
            st.download_button(
                label="Download Master Benchmark JSON",
                data=master_json_str,
                file_name=os.path.basename(master_results.get("saved_filepath", "master_layer_benchmark.json")),
                mime="application/json",
                type="primary"
            )

        st.subheader("Generated Prompt Artifacts")
        for i, pr in enumerate(prompt_runs):
            with st.expander(f"📄 Artifact #{i+1}: '{pr['prompt'][:40]}...' → Target: '{pr['target']}'", expanded=(i==0)):
                single_prompt_data = {
                    "benchmark_name": master_results.get("benchmark_name"),
                    "benchmark_version": master_results.get("benchmark_version"),
                    "timestamp": master_results.get("timestamp"),
                    "model": master_results.get("model"),
                    "sae_release": master_results.get("sae_release"),
                    "hook_location": master_results.get("hook_location"),
                    "safety_enabled": master_results.get("safety_enabled"),
                    "parameters": master_results.get("parameters"),
                    "prompt": pr["prompt"],
                    "target": pr["target"],
                    "layers": pr["layers"]
                }
                single_json_str = json.dumps(single_prompt_data, indent=4)
                st.download_button(
                    label=f"Download Prompt #{i+1} JSON",
                    data=single_json_str,
                    file_name=f"layer_benchmark_prompt_{i+1}.json",
                    mime="application/json",
                    key=f"dl_prompt_{i}"
                )
                st.code(single_json_str, language="json")

        with st.expander("📄 View Full Master Benchmark JSON", expanded=False):
            st.code(master_json_str, language="json")

if __name__ == "__main__":
    main()
