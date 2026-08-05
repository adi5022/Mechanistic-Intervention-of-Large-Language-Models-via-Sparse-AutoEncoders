"""
Layer Intervention Benchmark UI
Dedicated Streamlit application for characterizing how Hybrid Mute & Boost intervention
effectiveness changes across transformer layer depth across single or multiple prompt-target pairs.
"""

import streamlit as st
import pandas as pd
import json
import os
from src.sae_utils import load_model_and_sae
from src.benchmark.layer_benchmark_runner import run_layer_benchmark

# 1. Page Config
st.set_page_config(
    page_title="Layer Intervention Benchmark",
    page_icon=":material/analytics:",
    layout="wide"
)

# 2. Caching Strategy
@st.cache_resource
def get_cached_model_and_sae(layer: int):
    """Cached loader for HookedTransformer and Layer-specific SAE."""
    return load_model_and_sae(layer=layer)

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
    
    # Render prompt-target input rows
    to_remove = None
    for idx, item in enumerate(st.session_state["prompts_dataset"]):
        with st.sidebar.expander(f"Prompt #{idx + 1}", expanded=True):
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
        if st.button("➕ Add Prompt"):
            st.session_state["prompts_dataset"].append({"prompt": "", "target": ""})
            st.rerun()
    with col_rem:
        if len(st.session_state["prompts_dataset"]) > 1:
            if st.button("➖ Remove Prompt"):
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

        results_list = []

        for p_idx, p_item in enumerate(valid_pairs, start=1):
            prompt_text = p_item["prompt"].strip()
            target_text = p_item["target"].strip()

            def layer_callback(layer_step, current_layer):
                # Calculate progress fraction
                frac = ((p_idx - 1) + (layer_step / total_layers)) / total_prompts
                progress_bar.progress(min(frac, 1.0))
                status_box.markdown(
                    f"⏳ **Benchmarking Prompt {p_idx} / {total_prompts}** (`{prompt_text[:30]}...`) | "
                    f"**Layer {current_layer}** ({layer_step} / {total_layers})"
                )

            res = run_layer_benchmark(
                prompt=prompt_text,
                target=target_text,
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
            results_list.append(res)

        progress_bar.progress(1.0)
        status_box.success(f"✅ Completed benchmark across {len(results_list)} prompt(s)!")
        st.session_state["multi_benchmark_results"] = results_list

    # 4. Display Results
    if "multi_benchmark_results" in st.session_state and st.session_state["multi_benchmark_results"]:
        all_results = st.session_state["multi_benchmark_results"]
        
        st.markdown("---")
        st.header("Benchmark Results & Research Artifacts")

        prompt_options = [
            f"Prompt #{i+1}: '{r['prompt'][:35]}...' → Target: '{r['target']}'"
            for i, r in enumerate(all_results)
        ]

        selected_prompt_idx = st.selectbox(
            "Select Benchmark Prompt to Inspect:",
            options=list(range(len(all_results))),
            format_func=lambda i: prompt_options[i]
        )

        results = all_results[selected_prompt_idx]
        layers_data = results.get("layers", [])

        if "saved_filepath" in results:
            st.info(f"💾 Benchmark JSON automatically saved to: `{results['saved_filepath']}`")

        # Formatted results DataFrame
        table_rows = []
        chart_rows = []
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

        df_table = pd.DataFrame(table_rows)
        df_chart = pd.DataFrame(chart_rows)

        st.subheader(f"Results for Prompt #{selected_prompt_idx + 1}")
        st.caption(f"Prompt: `{results.get('prompt', '')}` | Target: `{results.get('target', '')}`")
        st.caption(f"Safety Filter: {'Enabled (Target Protected)' if results.get('use_safety', True) else 'Disabled (Raw Feature Selection)'}")
        
        st.dataframe(df_table, hide_index=True)

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

        # Raw JSON output & Download Workflow
        st.subheader("Research Artifact (Benchmark JSON)")
        json_str = json.dumps(results, indent=4)

        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            if "saved_filepath" in results:
                st.caption(f"Auto-saved artifact location: `{results['saved_filepath']}`")
        with col_d2:
            st.download_button(
                label="Download Benchmark JSON",
                data=json_str,
                file_name=os.path.basename(results.get("saved_filepath", f"layer_benchmark_prompt_{selected_prompt_idx+1}.json")),
                mime="application/json",
                type="primary"
            )

        with st.expander("📄 View Complete Raw Benchmark JSON", expanded=True):
            st.code(json_str, language="json")

if __name__ == "__main__":
    main()
