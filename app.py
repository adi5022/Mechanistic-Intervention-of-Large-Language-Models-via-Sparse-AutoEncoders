import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import urllib.request
import json
from src.sae_utils import load_model_and_sae
from src.editing import run_causal_selector, get_target_token_id
from src.hooks import make_ablation_hook

# Set page configuration with a premium look
st.set_page_config(
    page_title="FeatureScalpel MVP",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_neuronpedia_explanation(feature_id: int) -> str:
    url = f"https://www.neuronpedia.org/api/feature/gpt2-small/8-res-jb/{feature_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode())
            explanations = data.get("explanations", [])
            if explanations:
                return explanations[0].get("description", "No description available")
            return "No explanation found"
    except Exception:
        return "Lookup failed"

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f111a;
    }
    h1 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #cbd5e1;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Core Resource Caching -----------------
@st.cache_resource
def get_cached_model_and_sae():
    # Returns loaded model and SAE, cached so it only loads once per app session
    return load_model_and_sae(device="cpu")

# Load models
with st.spinner("Loading GPT-2 Small and Sparse Autoencoder... (takes a moment on first load)"):
    model, sae = get_cached_model_and_sae()

st.title("🔬 FeatureScalpel MVP Prototype")
st.markdown("Mechanistic Activation Interventions via Sparse Autoencoders without weight-editing.")

# Initialize session state
if "ranked_features" not in st.session_state:
    st.session_state.ranked_features = None
if "selected_feature_id" not in st.session_state:
    st.session_state.selected_feature_id = 11149

# ----------------- Section 1: Feature Selector -----------------
st.header("1️⃣ Causal Feature Selector")
st.markdown("Analyze which SAE features are causally responsible for suppressing a correct token.")

col1, col2 = st.columns([3, 1])
with col1:
    prompt_input = st.text_input(
        "Enter Prompt:", 
        value="The Eiffel Tower is in the city of"
    )
with col2:
    target_input = st.text_input(
        "Expected Target Token:", 
        value=" Paris"
    )

if st.button("Run Causal Feature Selector"):
    with st.spinner("Running causal interventions & querying Neuronpedia..."):
        # Run selector for top 20 active features
        results = run_causal_selector(model, sae, prompt_input, target_input, top_n=20)
        # Fetch description for each feature
        for res in results:
            res["description"] = get_neuronpedia_explanation(res["feature_id"])
        st.session_state.ranked_features = results
        if results:
            st.session_state.selected_feature_id = results[0]["feature_id"]

if st.session_state.ranked_features is not None:
    df = pd.DataFrame(st.session_state.ranked_features)
    
    # Highlight feature 11149 specifically if present
    def highlight_target_feature(row):
        if row["feature_id"] == 11149:
            return ["background-color: rgba(59, 130, 246, 0.2)"] * len(row)
        return [""] * len(row)
        
    styled_df = df.style.apply(highlight_target_feature, axis=1)
    
    col_table, col_chart = st.columns([3, 2])
    with col_table:
        st.subheader("Causal Rankings")
        st.dataframe(
            styled_df, 
            column_config={
                "feature_id": "Feature ID",
                "description": "Neuronpedia Explanation",
                "activation": "Activation",
                "clean_prob": st.column_config.NumberColumn("Baseline Prob", format="%.4f"),
                "ablated_prob": st.column_config.NumberColumn("Ablated Prob", format="%.4f"),
                "prob_delta": st.column_config.NumberColumn("Prob Delta", format="%.4f"),
                "kl_divergence": st.column_config.NumberColumn("KL Div", format="%.4f"),
                "top1_prediction": "New Top-1 Output"
            },
            hide_index=True,
            use_container_width=True
        )
    with col_chart:
        st.subheader("Causal Influence (Prob Delta)")
        chart_data = df[["feature_id", "prob_delta"]].copy()
        chart_data["feature_id"] = chart_data["feature_id"].astype(str)
        st.bar_chart(chart_data.set_index("feature_id"))

# ----------------- Section 2: Intervention Slider -----------------
st.header("2️⃣ Soft Ablation Control")
st.markdown("Manually tune the ablation strength of a specific feature and see live predictions.")

# Prep default feature ID list
feature_options = [11149]
if st.session_state.ranked_features is not None:
    feature_options = [r["feature_id"] for r in st.session_state.ranked_features]

selected_feature = st.selectbox(
    "Select Feature ID to Intervene On:",
    options=feature_options,
    index=feature_options.index(st.session_state.selected_feature_id) if st.session_state.selected_feature_id in feature_options else 0
)

# Fetch explanation for selected feature
feat_explanation = get_neuronpedia_explanation(selected_feature)
st.markdown(f"**Neuronpedia Explanation:** *\"{feat_explanation}\"*")
st.markdown(f"[View Feature on Neuronpedia 🔗](https://www.neuronpedia.org/gpt2-small/8-res-jb/{selected_feature})")

ablation_strength = st.slider(
    "Ablation Strength (0.0 = untouched, 1.0 = fully ablated)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

# Helper function to get top predictions
def get_predictions(prompt_str, feat_id, strength):
    tokens = model.to_tokens(prompt_str)
    if strength > 0.0:
        hook_fn = make_ablation_hook(feat_id, sae, strength)
        logits = model.run_with_hooks(tokens, fwd_hooks=[("blocks.8.hook_resid_pre", hook_fn)])
    else:
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_k = probs.topk(5)
    return [(model.to_string(idx), val.item()) for val, idx in zip(top_k.values, top_k.indices)]

# Display side-by-side predictions
st.subheader(f"Effect on Prompt: '{prompt_input}'")

# Dynamic Warning/Information Tooltips for Eiffel/Colosseum
if "Eiffel Tower" in prompt_input and selected_feature == 11149:
    # Get live predictions for Eiffel and Colosseum at current strength
    eiffel_top = get_predictions("The Eiffel Tower is in the city of", 11149, ablation_strength)[0][0].strip().lower()
    colosseum_top = get_predictions("The Colosseum is in the city of", 11149, ablation_strength)[0][0].strip().lower()
    
    eiffel_ok = (eiffel_top == "paris")
    colosseum_ok = (colosseum_top == "rome")
    
    if not eiffel_ok:
        st.info(f"💡 **Observation (Strength {ablation_strength:.2f}):** Ablation is too weak. The model still predicts '{eiffel_top.capitalize()}' instead of Paris.")
    elif eiffel_ok and colosseum_ok:
        st.success(f"🎯 **Sweet Spot (Strength {ablation_strength:.2f}):** The correct prediction ('Paris') becomes top-1 while the control case ('Rome') remains preserved!")
    else:
        st.warning(f"⚠️ **Side Effects Detected (Strength {ablation_strength:.2f}):** The Eiffel Tower is fixed ('Paris'), but the control prompt (The Colosseum) is now broken (predicts '{colosseum_top.capitalize()}').")

col_before, col_after = st.columns(2)

with col_before:
    st.markdown("### Clean Baseline (0.0)")
    baseline_preds = get_predictions(prompt_input, selected_feature, 0.0)
    for token, prob in baseline_preds:
        st.progress(prob, text=f"**{token.strip()}**: {prob:.4f}")

with col_after:
    st.markdown(f"### Intervened ({ablation_strength:.2f})")
    intervened_preds = get_predictions(prompt_input, selected_feature, ablation_strength)
    for token, prob in intervened_preds:
        st.progress(prob, text=f"**{token.strip()}**: {prob:.4f}")

# ----------------- Section 3: Specificity / Side-Effect Check -----------------
st.header("3️⃣ Specificity Check")
st.markdown("Verify if your intervention fixes target errors without breaking unrelated correct predictions.")

stored_prompts = [
    {"prompt": "The Eiffel Tower is in the city of", "target": " Paris", "tag": "Target (Error)"},
    {"prompt": "The Colosseum is in the city of", "target": " Rome", "tag": "Control (Correct)"},
    {"prompt": "Big Ben is in the city of", "target": " London", "tag": "Control (Error)"},
    {"prompt": "The Sagrada Familia is in the city of", "target": " Barcelona", "tag": "Control (Error)"}
]

cols_spec = st.columns(len(stored_prompts))

for i, test in enumerate(stored_prompts):
    with cols_spec[i]:
        st.subheader(f"{test['tag']}")
        st.write(f"*{test['prompt']}*")
        
        # Get baseline vs ablated top predictions
        base_top = get_predictions(test["prompt"], selected_feature, 0.0)[0][0]
        abl_top = get_predictions(test["prompt"], selected_feature, ablation_strength)[0][0]
        
        target_clean = test["target"].strip().lower()
        top_ablated_clean = abl_top.strip().lower()
        
        is_correct = (top_ablated_clean == target_clean)
        
        # Visual indicator card
        if is_correct:
            st.markdown(f"""
            <div style="background-color:rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 10px; border-radius: 4px;">
                <b style="color:#10b981;">🟢 Correct</b><br>
                Baseline Top-1: <b>{base_top.strip()}</b><br>
                Intervened Top-1: <b>{abl_top.strip()}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Check if this intervention broke a correct baseline or kept it wrong
            base_correct = (base_top.strip().lower() == target_clean)
            if base_correct:
                status_text = "🔴 Broken by Intervention"
            else:
                status_text = "⚪ Unchanged / Still Wrong"
                
            st.markdown(f"""
            <div style="background-color:rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 10px; border-radius: 4px;">
                <b style="color:#ef4444;">{status_text}</b><br>
                Baseline Top-1: <b>{base_top.strip()}</b><br>
                Intervened Top-1: <b>{abl_top.strip()}</b>
            </div>
            """, unsafe_allow_html=True)

# Sidebar helper instructions
with st.sidebar:
    st.markdown("### How to Run the Demo")
    st.markdown("""
    1. **Run Causal Feature Selector** to find feature `11149`.
    2. Adjust the **Ablation Slider** to `0.3`.
    3. Observe that the Eiffel Tower prompt prediction switches to **Paris**.
    4. Check the **Specificity Panel** at `0.3` to see that **The Colosseum** remains correctly predicted as **Rome**, but degrades and breaks at `0.5+`.
    """)
