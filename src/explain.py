import os
import requests
import streamlit as st

def generate_mechanistic_explanation(
    prompt: str,
    target: str,
    baseline_prob: float,
    baseline_rank: int,
    final_prob: float,
    final_rank: int,
    interventions: list[dict],
    api_key: str = None
) -> str:
    """
    Calls the Groq API to translate complex feature activations and interventions
    into a plain-English layperson explanation.
    """
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key and "groq" in st.secrets:
        api_key = st.secrets["groq"].get("api_key")
        
    if not api_key:
        return "Configure your GROQ_API_KEY in environment variables or the sidebar to enable AI explanations."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    intervention_desc = ""
    for inv in interventions:
        action = inv.get("action", "modified")
        fid = inv.get("feature_id")
        desc = inv.get("description", "Unknown concept")
        strength = inv.get("strength", 1.0)
        intervention_desc += f"- Feature {fid} ({desc}): {action} at strength {strength:.2f}\n"
        
    user_prompt = f"""
Analyze this mechanistic intervention on GPT-2's internal activations.

Context:
- Prompt: "{prompt}"
- Target Token: "{target}"
- Baseline Model State (Before): Target Prob = {baseline_prob*100:.2f}%, Target Rank = {baseline_rank}
- Post-Intervention State (After): Target Prob = {final_prob*100:.2f}%, Target Rank = {final_rank}

Interventions Applied:
{intervention_desc}

Task:
Write a concise, plain-English summary (2-3 sentences max) explaining what this intervention did and why it worked or failed. Focus on translating the feature descriptions into how they influenced the model's behavior. Do not use complex neural network jargon.
"""

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful mechanistic interpretability researcher who explains complex neural activations in simple, clear layperson terms."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 150
    }
    
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Groq API Error: {r.status_code} - {r.text}"
    except Exception as e:
        return f"Failed to connect to Groq: {e}"
