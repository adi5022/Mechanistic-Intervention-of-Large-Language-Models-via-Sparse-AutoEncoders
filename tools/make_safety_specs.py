"""
Create the batch specs for the safety-filter study from data/safety_filter_prompts_full.json.

  data/safety_filter_spec_pilot.json  - 45 steerable prompts (stratified by baseline rank) + 5 controls, all-positions mode
  data/safety_filter_spec_full.json   - every steerable prompt with baseline rank <= 1000 + 25 controls, all-positions mode
  data/safety_filter_spec_full_last.json - same prompts, last-token mode (does relaxing the filter help when the pool is small?)
  data/safety_filter_spec_everything.json - full prompts in BOTH modes (the whole study in one run)
  data/safety_filter_spec_quick.json  - 6 prompts + 2 controls in both modes, a quick end-to-end check

Arms compare the safety filter variants on identical prompts:
  strict         the original filter (control)
  off            no safety filter at all (ablation)
  tol5_after     accept features whose harm is within 5% of the target's probability; used after the strictly safe ones
  graded5_after  same budget, but over-budget features are admitted at a reduced strength; used after the strictly safe ones
  graded5_inter  as above, but admitted features keep their natural (blocker/target effect) order
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS = [
    {"label": "strict", "settings": {"safety_mode": "strict"}},
    {"label": "off", "settings": {"safety_filter": False}},
    {"label": "tol5_after", "settings": {"safety_mode": "tolerance", "tolerance": 0.05, "rescued_order": "after"}},
    {"label": "graded5_after", "settings": {"safety_mode": "graded", "tolerance": 0.05, "rescued_order": "after"}},
    {"label": "graded5_inter", "settings": {"safety_mode": "graded", "tolerance": 0.05, "rescued_order": "interleaved"}},
]
BASE_SETTINGS = {"mute_strength": 0.6, "boost_strength": 0.5, "top_n": 120, "cumulative_sweep": True, "pool_refill": True,
                 "max_refill_rounds": 0, "stop_on_rank1": True, "use_batched": True, "safety_filter": True,
                 "max_steps": 250, "collateral": True, "combination_check": False, "record_detail": "compact"}


def prompt_entry(o):
    return {"prompt": o["prompt"], "target": o["target"], "meta": o["meta"]}


def write(name, title, prompts, modes):
    spec = {"name": title, "settings": BASE_SETTINGS, "modes": modes, "repeats": 1, "arms": ARMS,
            "prompts": [prompt_entry(o) for o in prompts]}
    path = os.path.join(ROOT, "data", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    print(f"{name}: {len(prompts)} prompts x {len(ARMS)} arms x {len(modes)} mode(s) = {len(prompts) * len(ARMS) * len(modes)} runs")


def main():
    with open(os.path.join(ROOT, "data", "safety_filter_prompts_full.json"), encoding="utf-8") as f:
        allp = json.load(f)
    steer = [o for o in allp if 1 < o["meta"]["baseline_rank"] <= 1000]
    ctrl = [o for o in allp if o["meta"]["baseline_rank"] == 1]
    rnd = random.Random(0)

    bins = [(2, 3, 10), (4, 10, 10), (11, 50, 10), (51, 300, 10), (301, 1000, 5)]
    pilot = []
    for lo, hi, n in bins:
        pool = [o for o in steer if lo <= o["meta"]["baseline_rank"] <= hi]
        rnd.shuffle(pool)
        pilot += pool[:n]
    pilot += rnd.sample(ctrl, 5)
    write("safety_filter_spec_pilot.json", "Safety filter study - PILOT (all prompt positions)", pilot, ["all"])

    full = steer + rnd.sample(ctrl, min(25, len(ctrl)))
    write("safety_filter_spec_full.json", "Safety filter study - FULL (all prompt positions)", full, ["all"])
    write("safety_filter_spec_full_last.json", "Safety filter study - FULL (last token only)", full, ["last"])
    write("safety_filter_spec_everything.json", "Safety filter study - EVERYTHING (all positions + last token)", full, ["all", "last"])

    quick_names = ["This is Sophia, she is a", "The doctor said that", "Mount Everest is located in", "Ferrari is a car company from",
                   "Albert Einstein was born in", "Diwali is celebrated in"]
    quick = [o for o in steer if o["prompt"] in quick_names] + rnd.sample(ctrl, 2)
    write("safety_filter_spec_quick.json", "Safety filter study - QUICK CHECK (sanity test, a few minutes)", quick, ["all", "last"])


if __name__ == "__main__":
    main()
