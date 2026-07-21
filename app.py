import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from src.sae_utils import load_model_and_sae
from src.editing import run_causal_selector
from src.hooks import make_ablation_hook
from src.evaluation import classify_fact

# Page config
st.set_page_config(page_title="FeatureScalpel MVP", layout="wide")

# Page title
st.title("🔬 FeatureScalpel — MVP Prototype")

# 1. Load the model and SAE (cached so it only runs once)
@st.cache_resource
def get_cached_model_and_sae():
    return load_model_and_sae(device="cpu")

with st.spinner("Loading model and SAE..."):
    model, sae = get_cached_model_and_sae()

st.success("Model and SAE loaded successfully!")

# Section 0 — Factual Eligibility Scanner
st.header("Section 0 — Factual Eligibility Scanner")
st.markdown(
    "Before performing an intervention, we check if the fact is **eligible** (i.e. present in the model's weights but suppressed)."
)

col1, col2 = st.columns(2)
with col1:
    prompt_input = st.text_input(
        "Type a sentence (Prompt):", 
        value="The Eiffel Tower is in the city of"
    )
with col2:
    target_input = st.text_input(
        "Target token (include leading space):", 
        value=" Paris"
    )

# Run classification on change
try:
    classification = classify_fact(model, prompt_input, target_input)
    label = classification["label"]
    clean_prob = classification["clean_prob"]
    rank = classification["rank"]
    
    if label == "already_correct":
        st.success(
            f"✅ **Baseline Correct**: The model already predicts `{repr(target_input)}` as its top-1 guess "
            f"(Rank {rank}, Probability: {clean_prob*100:.2f}%). No intervention is needed!"
        )
    elif label == "suppressed":
        st.info(
            f"⚡ **Suppressed Knowledge Detected**: The model knows `{repr(target_input)}` (Rank {rank}, "
            f"Probability: {clean_prob*100:.2f}%) but it is outcompeted at the output layer. "
            f"This is an **eligible** candidate for intervention!"
        )
    else:
        st.error(
            f"❌ **Absent Knowledge**: The model does not know `{repr(target_input)}` (Rank {rank}, "
            f"Probability: {clean_prob*100:.2f}%). Steering is unlikely to work."
        )
except Exception as e:
    st.error(f"Error scanning fact: {e}")

st.markdown("---")

# Section 1 — Feature Selector (the new, automated part)
st.header("Section 1 — Automated Causal Feature Selector")
st.markdown(
    "Identify which sparse autoencoder features causally drive predictions. "
    "We extract the top active features on the final token and ablate each one to observe the probability drop."
)

if "causal_results" not in st.session_state:
    st.session_state.causal_results = None

if st.button("Run Causal Feature Selector"):
    with st.spinner("Running causal analysis (fully ablating top active features)..."):
        try:
            results = run_causal_selector(model, sae, prompt_input, target_input, top_n=20)
            st.session_state.causal_results = results
        except Exception as e:
            st.error(f"Error running selector: {e}")

if st.session_state.causal_results is not None:
    results = st.session_state.causal_results
    st.subheader("Ranked Causal Features")
    
    # Format table for user display
    df_data = []
    for r in results:
        df_data.append({
            "Rank": len(df_data) + 1,
            "Feature ID": r["feature_id"],
            "Activation": f"{r['activation']:.4f}",
            "Clean Prob": f"{r['clean_prob'] * 100:.2f}%",
            "Ablated Prob": f"{r['ablated_prob'] * 100:.2f}%",
            "Prob Delta": f"{r['prob_delta'] * 100:.2f}%",
            "Top-1 Ablated": repr(r["top1_prediction"]),
        })
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Check if 11149 is in top features
    top_feature_ids = [r["feature_id"] for r in results]
    if 11149 in top_feature_ids:
        rank_11149 = top_feature_ids.index(11149) + 1
        st.info(f"🎉 **Validation Confirmed**: Feature `11149` was automatically discovered and ranked **#{rank_11149}** in causal effect!")
    else:
        st.warning("Feature `11149` was not found in the top 20 active features.")

st.markdown("---")

# Section 2 — The Slider (manual intervention, already validated)
st.header("Section 2 — Live Soft Ablation Slider")
st.markdown(
    "Select a feature to ablate and adjust the strength slider to see the live impact on the model's next-token predictions."
)

# Auto-fill selected feature with top ranked causal feature or default to 11149
default_feature = 11149
if st.session_state.causal_results is not None and len(st.session_state.causal_results) > 0:
    default_feature = st.session_state.causal_results[0]["feature_id"]

col3, col4 = st.columns([1, 3])
with col3:
    selected_feature_id = st.number_input(
        "Feature ID to ablate:", 
        min_value=0, 
        max_value=sae.cfg.d_sae - 1, 
        value=default_feature,
        step=1
    )
with col4:
    ablation_strength = st.slider(
        "Ablation Strength (θ):", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.05
    )

# Helper function to run inference and return top 5 predictions
def get_predictions(prompt_str, feature_id=None, strength=0.0):
    tokens = model.to_tokens(prompt_str)
    if feature_id is not None and strength > 0.0:
        hook_fn = make_ablation_hook(feature_id, sae, strength)
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("blocks.8.hook_resid_pre", hook_fn)]
            )
    else:
        with torch.no_grad():
            logits = model(tokens)
            
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_k = probs.topk(5)
    return [(model.to_string(idx), val.item()) for val, idx in zip(top_k.values, top_k.indices)]

# Show info box for Eiffel Tower specific demo
if prompt_input == "The Eiffel Tower is in the city of" and selected_feature_id == 11149:
    st.info(
        "💡 **Eiffel Tower Demo Guide**:\n"
        "- At **θ = 0.0** (untouched), the model predicts `' London'` (wrong).\n"
        "- Try moving the slider to **θ = 0.3**; you'll see `' Paris'` become the top prediction!\n"
        "- Keep this in mind for the Specificity check in Section 3."
    )

# Side-by-side comparison
col_clean, col_ablated = st.columns(2)

with col_clean:
    st.subheader("Before Ablation (Baseline)")
    clean_preds = get_predictions(prompt_input, strength=0.0)
    for token, prob in clean_preds:
        st.write(f"Word: `{repr(token)}` | Probability: {prob * 100:.2f}%")

with col_ablated:
    st.subheader(f"After Ablation (Feature {selected_feature_id} @ θ={ablation_strength:.2f})")
    ablated_preds = get_predictions(prompt_input, feature_id=selected_feature_id, strength=ablation_strength)
    for token, prob in ablated_preds:
        st.write(f"Word: `{repr(token)}` | Probability: {prob * 100:.2f}%")
