"""
Layer Intervention Benchmark UI
Dedicated Streamlit application for characterizing how Hybrid Mute & Boost intervention
effectiveness changes across transformer layer depth across single or multiple prompt-target pairs.
"""

import streamlit as st
import pandas as pd
import json
import os
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
def get_cached_base_model():
    """Cached loader for GPT-2 base model (loaded ONCE per Streamlit session)."""
    return load_base_model()

@st.cache_resource
def get_cached_sae(layer: int):
    """Cached loader for Layer-specific SAE (loaded ONCE per layer index)."""
    return load_sae_for_layer(layer=layer)

def get_cached_model_and_sae(layer: int):
    """Composes cached GPT-2 base model and cached layer SAE without re-initializing GPT-2."""
    model = get_cached_base_model()
    sae = get_cached_sae(layer=layer)
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
            frac = ((p_idx - 1) + (l_idx / total_l)) / total_p
            progress_bar.progress(min(frac, 1.0))
            status_box.markdown(
                f"⏳ **Prompt {p_idx} / {total_p}** (`{prompt_text[:30]}...`) | "
                f"**Layer {current_layer}** ({l_idx} / {total_l}) | **Stage:** `{current_stage_text}`"
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
            model_sae_loader=get_cached_model_and_sae,
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

        # Experiment Metadata Display
        st.caption(
            f"**Model:** `{master_results.get('model', 'gpt2')}` | "
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
            profile_rows.append({
                "Layer": f"Layer {r['layer']}",
                "SAE Load (ms)": f"{prof.get('sae_loading_ms', 0.0):.1f}",
                "Clean Baseline (ms)": f"{prof.get('clean_baseline_ms', 0.0):.1f}",
                "Feature Select (ms)": f"{prof.get('feature_selection_ms', 0.0):.1f}",
                "Safety Filter (ms)": f"{prof.get('safety_filtering_ms', 0.0):.1f}",
                "Intervention (ms)": f"{prof.get('intervention_ms', 0.0):.1f}",
                "Total (ms)": f"{prof.get('total_layer_ms', 0.0):.1f}"
            })

        df_table = pd.DataFrame(table_rows)
        df_chart = pd.DataFrame(chart_rows)
        df_profile = pd.DataFrame(profile_rows)

        st.subheader(f"Results for Prompt #{selected_prompt_idx + 1}")
        st.caption(f"Prompt: `{selected_run.get('prompt', '')}` | Target: `{selected_run.get('target', '')}`")
        
        st.dataframe(df_table, hide_index=True)

        # Display Stage Profiling Breakdown Table
        with st.expander("⏱️ View Layer Stage Profiling Breakdown", expanded=False):
            st.dataframe(df_profile, hide_index=True)

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
