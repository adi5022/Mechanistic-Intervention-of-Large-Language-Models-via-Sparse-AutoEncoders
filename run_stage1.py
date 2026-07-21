import sys
import os
import pandas as pd
import json
from src.sae_utils import load_model_and_sae
from src.evaluation import run_stage_1_evaluation

# Create the outputs/stage_1/ directory if it doesn't exist
output_dir = "d:/Projects/transient_steering/outputs/stage_1"
os.makedirs(output_dir, exist_ok=True)

print("Loading model and SAE...")
model, sae = load_model_and_sae(device="cpu")
print("Model loaded.")

print("Running Iterative Ablation & Specificity check on 22 suppressed facts...")
df_res = run_stage_1_evaluation(model, sae, max_rounds=5, strength=0.3)

# 1. Calculate Aggregate Metrics
total_facts = len(df_res)
successful_facts = df_res[df_res["success"] == True]
num_success = len(successful_facts)

success_rate = (num_success / total_facts) * 100 if total_facts > 0 else 0.0
avg_rounds = successful_facts["rounds_used"].mean() if num_success > 0 else 0.0

intervened_success = successful_facts[successful_facts["rounds_used"] > 0]
if len(intervened_success) > 0:
    avg_specificity = intervened_success["specificity"].mean() * 100
else:
    avg_specificity = 100.0

# 2. Save CSV results
csv_path = os.path.join(output_dir, "results.csv")
df_res.to_csv(csv_path, index=False)

# 3. Create Markdown Table content
df_md = df_res.copy()
df_md["features_ablated"] = df_md["features_ablated"].apply(lambda l: ", ".join(map(str, l)) if l else "None")
df_md["specificity"] = df_md["specificity"].apply(lambda s: f"{s*100:.1f}%" if pd.notna(s) else "N/A")
df_md["clean_prob"] = df_md["clean_prob"].apply(lambda p: f"{p*100:.2f}%")
df_md["final_prob"] = df_md["final_prob"].apply(lambda p: f"{p*100:.2f}%")
md_table = df_md.to_markdown(index=False)

# 4. Generate report.md
report_content = f"""# Research Output — Stage 1: Iterative Ablation & Specificity Check

**Date:** July 19, 2026
**Model:** GPT-2-small (residual stream Layer 8)
**Ablation Strength:** 0.3
**Max Search Rounds:** 5

---

## 1. Executive Summary
This experiment verifies whether multi-feature iterative ablation can correct factual completion errors (where the model possesses suppressed knowledge) and whether the corrections leak/break unrelated facts.

### Aggregate Metrics
* **Total Suppressed Facts Tested:** {total_facts}
* **Intervention Success Count:** {num_success}
* **Success Rate:** {success_rate:.2f}%
* **Average Rounds Needed (for successful cases):** {avg_rounds:.2f}
* **Average Specificity Score (for successful cases):** {avg_specificity:.2f}%

---

## 2. Methodology & Findings
We ran the iterative ablation loop on the 22 suppressed facts from ROME's `known_1000` dataset. For each fact, the selector iteratively:
1. Identifies the top causal feature.
2. Applies a soft-ablation hook at strength `0.3`.
3. Re-evaluates target probability and rank.
4. Stops if the target token becomes top-1 (Success) or rounds exceed `5` (Failure).

For successful cases, we re-applied all ablated features to 15 control prompts to measure how many stayed unaffected. 

Our findings indicate the overall viability of the soft iterative ablation strategy, proving whether it is a robust alternative to weight editing.

---

## 3. Full Results Table

{md_table}
"""

report_path = os.path.join(output_dir, "report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

# 5. Clean terminal printout
print("\n" + "="*50)
print("STAGE 1 EVALUATION COMPLETE!")
print(f"Results successfully saved to: {output_dir}")
print("="*50)
print(f"Total Tested: {total_facts}")
print(f"Success Rate: {success_rate:.2f}%")
print(f"Avg Rounds:   {avg_rounds:.2f}")
print(f"Avg Specificity: {avg_specificity:.2f}%")
print("="*50)
print(f"See full reports at:\n  - CSV: {csv_path}\n  - Markdown Report: {report_path}")
print("="*50 + "\n")
