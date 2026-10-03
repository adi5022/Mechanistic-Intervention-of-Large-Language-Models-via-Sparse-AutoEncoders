"""
Control for the additive edit (docs/Research_Journal/30.md): can it push words that have NOTHING to do with the prompt to rank 1?

20 validation prompts (seeded), each paired with a random unrelated single-token word drawn from the whole vocabulary
(' ' + lowercase letters, longer than 3 letters). Three arms, 100 steps, Top N 200: multipliers only, + additive (cap factor 1.0),
+ additive (cap factor 0.25). If the additive edit reaches rank 1 on nearly all of them, rank 1 is not evidence that it recovers
what the model knows; it is evidence the edit is strong enough to force any word.

    .venv\Scripts\python.exe tools/control_random_targets.py
Writes docs/Research_Journal/packs/additive_control/results.json
"""
import os, sys, json, random, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from src.sae_utils import load_base_model, load_sae_for_layer
from src.editing import build_clean_context, get_target_token_id
from src.gradient_editing import run_gradient_descent_edit

model = load_base_model(); layer = 8
sae = load_sae_for_layer(layer)
hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")

cf = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "datasets", "counterfact.json"), encoding="utf-8"))}
split = json.load(open(os.path.join(ROOT, "data", "counterfact_split.json"), encoding="utf-8"))
rng = random.Random(11)
prompts = []
for cid in rng.sample(split["val"], 20):
    rec = cf[cid]["requested_rewrite"]
    prompts.append(rec["prompt"].format(rec["subject"]))

# random single-token WORDS: ' ' + lowercase letters, from the whole vocabulary (so mostly rare, unrelated words)
cands = [i for i in range(model.cfg.d_vocab) if (s := model.to_string([i])).startswith(" ") and s[1:].isalpha() and s[1:].islower() and len(s) > 4]
targets = [model.to_string([i]) for i in rng.sample(cands, 20)]

arms = [("multipliers only", {}), ("+ additive (cap 1.0)", {"additive": True}), ("+ additive (cap 0.25)", {"additive": True, "add_cap_factor": 0.25})]
res = {a: [] for a, _ in arms}
rows = []
for p, t in zip(prompts, targets):
    tid = get_target_token_id(model, t)
    ctx = build_clean_context(model, sae, p, tid)
    line = f"{p[:38]!r:<42} -> {t!r:<14} start #{ctx.clean_rank:<6}"
    row = {"prompt": p, "target": t, "start_rank": ctx.clean_rank}
    for name, kw in arms:
        g = run_gradient_descent_edit(model, sae, ctx, tid, p, layer, hook_name, top_n=200, steps=100, **kw)
        rp = g["real_path"]
        res[name].append((rp["rank"], rp["kl"], g["edit_size_frac_norm"], rp["prob"]))
        row[name] = {"rank": rp["rank"], "kl": rp["kl"], "edit_size_frac_norm": g["edit_size_frac_norm"], "prob": rp["prob"]}
        line += f" | {name.split('(')[-1].strip(')') if '(' in name else 'mult'}: #{rp['rank']} sz {100 * g['edit_size_frac_norm']:.0f}%"
    rows.append(row)
    print(line, flush=True)
print("\nRANDOM unrelated target words, 20 prompts (validation split):")
for name, _ in arms:
    R = res[name]
    ok = sum(1 for r in R if r[0] == 1)
    print(f"  {name:<24} rank 1 on {ok}/20 | median end rank {st.median([r[0] for r in R]):.0f} | median KL {st.median([r[1] for r in R]):.2f} | median edit size {100 * st.median([r[2] for r in R]):.0f}% of norm | median target prob at end {100 * st.median([r[3] for r in R]):.1f}%")

os.makedirs(os.path.join(ROOT, "docs", "Research_Journal", "packs", "additive_control"), exist_ok=True)
summary = {name: {"rank1": sum(1 for r in R if r[0] == 1), "n": len(R), "median_end_rank": st.median([r[0] for r in R]), "median_kl": st.median([r[1] for r in R]),
                  "median_edit_size_frac_norm": st.median([r[2] for r in R]), "median_prob": st.median([r[3] for r in R])} for name, R in res.items()}
json.dump({"seed": 11, "split": "val", "rows": rows, "summary": summary}, open(os.path.join(ROOT, "docs", "Research_Journal", "packs", "additive_control", "results.json"), "w"), indent=1)
