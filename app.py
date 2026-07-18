import streamlit as st
import torch
import torch.nn.functional as F
from src.sae_utils import load_model_and_sae

# Page title
st.title("🔬 FeatureScalpel — Step 1: Baseline Predictor")

# 1. Load the model and SAE (cached so it only runs once)
@st.cache_resource
def get_cached_model_and_sae():
    return load_model_and_sae(device="cpu")

with st.spinner("Loading model..."):
    model, sae = get_cached_model_and_sae()

st.success("Model loaded successfully!")

# 2. User Input
prompt_input = st.text_input(
    "Type a sentence (Prompt):", 
    value="The Eiffel Tower is in the city of"
)

# Helper function to get top predictions
def get_predictions(prompt_str):
    tokens = model.to_tokens(prompt_str)
    with torch.no_grad():
        logits = model(tokens)
    # Get probabilities for the very last token
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_k = probs.topk(5)
    
    # Return list of (token_string, probability)
    return [(model.to_string(idx), val.item()) for val, idx in zip(top_k.values, top_k.indices)]

# 3. Display Predictions
st.subheader("Model's Top 5 Predictions for the next word:")
predictions = get_predictions(prompt_input)

for token, prob in predictions:
    # Use repr() to show exactly what the model predicts, including spaces (e.g. ' Paris')
    st.write(f"Word: `{repr(token)}` | Probability: {prob * 100:.2f}%")
