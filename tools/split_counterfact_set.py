"""
Step 2 of the learned-strength study (docs/Research_Journal/23.md, Phase 1 / Section 13).

Splits data/counterfact_hard_set.json (the 16,360 prompts GPT-2 gets wrong) into

    train        a BALANCED ORDER over the whole training pool; use the first N (250, 500, 1000, 2000 ...).
                 Every prefix is balanced across relations and starting-rank bands, so the sets are nested.
    val          prompts for choosing settings (never used for the final score)
    test_seen    new subjects of relations that appear in training
    test_unseen  subjects of relations held out of training and validation completely

Rules
  * A subject never appears in two splits (subjects of the held-out relations are removed from the training pool).
  * The held-out relations are disjoint from every training relation.
  * Selection is seeded; the seed and the held-out relations are written into the output.

Output: data/counterfact_split.json (case_ids only, small) and data/counterfact_split_summary.json.

    .venv\\Scripts\\python.exe tools/split_counterfact_set.py --dry-run     (prints the plan, writes nothing)
    .venv\\Scripts\\python.exe tools/split_counterfact_set.py               (writes the split)
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Five relations with very different kinds of answers, none of them the language / country relations that
# dominate the rest of the data:  P140 religion, P641 sport, P1303 instrument, P176 manufacturer, P190 twinned city.
DEFAULT_HELDOUT = ["P140", "P641", "P1303", "P176", "P190"]
MIN_RELATION_SIZE = 50                      # relations with fewer kept prompts are left out entirely (P264 has 6)
BANDS = [(2, 5), (6, 20), (21, 100), (101, 1000)]
RELATION_NAMES = {   # Wikidata property ids, for reading the report
    "P1412": "languages spoken", "P495": "country of origin", "P937": "work location", "P27": "citizenship",
    "P30": "continent", "P17": "country", "P449": "original network", "P19": "place of birth",
    "P740": "location of formation", "P37": "official language", "P159": "headquarters location",
    "P413": "position played", "P20": "place of death", "P103": "native language", "P131": "located in",
    "P364": "original language of work", "P276": "location", "P136": "genre", "P1303": "instrument",
    "P176": "manufacturer", "P140": "religion", "P39": "position held", "P127": "owned by", "P108": "employer",
    "P178": "developer", "P641": "sport", "P138": "named after", "P106": "occupation", "P101": "field of work",
    "P407": "language of work", "P190": "twinned city", "P463": "member of", "P36": "capital", "P264": "record label"}


def band_of(rank):
    for lo, hi in BANDS:
        if lo <= rank <= hi:
            return f"{lo}-{hi}"
    return "other"


def balanced_order(items, rng):
    """Order items so that every prefix is balanced across (relation, rank band) strata (round robin over strata)."""
    strata = defaultdict(list)
    for it in items:
        strata[(it["relation_id"], band_of(it["baseline_rank"]))].append(it)
    keys = sorted(strata)
    for k in keys:
        rng.shuffle(strata[k])
    rng.shuffle(keys)
    out, r = [], 0
    while len(out) < len(items):
        for k in keys:
            if r < len(strata[k]):
                out.append(strata[k][r])
        r += 1
    return out


def describe(name, items):
    rels = Counter(i["relation_id"] for i in items)
    bands = Counter(band_of(i["baseline_rank"]) for i in items)
    top = ", ".join(f"{r} {c}" for r, c in rels.most_common(4))
    print(f"  {name:<12}{len(items):>6} prompts | {len(rels):>2} relations | bands " +
          " ".join(f"{b}:{bands.get(b, 0)}" for b in [f'{lo}-{hi}' for lo, hi in BANDS]) + f" | most common: {top}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "counterfact_hard_set.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "counterfact_split.json"))
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--heldout", default=",".join(DEFAULT_HELDOUT), help="comma-separated relation ids kept out of training")
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test-seen", type=int, default=150)
    ap.add_argument("--test-unseen", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with open(a.src, encoding="utf-8") as f:
        prompts = json.load(f)["prompts"]
    rng = random.Random(a.seed)
    heldout = [r.strip() for r in a.heldout.split(",") if r.strip()]

    rel_counts = Counter(p["relation_id"] for p in prompts)
    small = sorted(r for r, c in rel_counts.items() if c < MIN_RELATION_SIZE)
    prompts = [p for p in prompts if p["relation_id"] not in small]
    unknown = [r for r in heldout if r not in rel_counts]
    if unknown:
        raise SystemExit(f"Held-out relation(s) not in the data: {unknown}")

    print(f"{len(prompts)} prompts across {len({p['relation_id'] for p in prompts})} relations "
          f"(dropped relations with fewer than {MIN_RELATION_SIZE} prompts: {small or 'none'})")
    print("Held-out relations (never seen in training or validation):")
    for r in heldout:
        print(f"  {r:<7}{RELATION_NAMES.get(r, '?'):<22}{rel_counts[r]:>5} prompts")

    # 1. held-out pool, and every subject that appears in it is removed from the seen pool (no subject leakage)
    held_pool = [p for p in prompts if p["relation_id"] in heldout]
    held_subjects = {p["subject"] for p in held_pool}
    seen_pool = [p for p in prompts if p["relation_id"] not in heldout and p["subject"] not in held_subjects]
    removed = len([p for p in prompts if p["relation_id"] not in heldout]) - len(seen_pool)
    print(f"Seen pool: {len(seen_pool)} prompts ({removed} removed because their subject also occurs in a held-out relation)")

    # 2. split SUBJECTS of the seen pool into train / val / test_seen buckets (a subject stays in one bucket)
    subjects = sorted({p["subject"] for p in seen_pool})
    rng.shuffle(subjects)
    n = len(subjects)
    bucket = {}
    for i, s in enumerate(subjects):
        bucket[s] = "test_seen" if i < 0.15 * n else ("val" if i < 0.30 * n else "train")
    pools = defaultdict(list)
    for p in seen_pool:
        pools[bucket[p["subject"]]].append(p)

    # 3. balanced selections
    val = balanced_order(pools["val"], rng)[:a.val]
    test_seen = balanced_order(pools["test_seen"], rng)[:a.test_seen]
    test_unseen = balanced_order(held_pool, rng)[:a.test_unseen]
    train_order = balanced_order(pools["train"], rng)          # the WHOLE training pool, in balanced order

    # 4. checks
    sets = {"train": train_order, "val": val, "test_seen": test_seen, "test_unseen": test_unseen}
    subj = {k: {p["subject"] for p in v} for k, v in sets.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = subj[names[i]] & subj[names[j]]
            assert not inter, f"subject overlap between {names[i]} and {names[j]}: {len(inter)}"
    train_rels = {p["relation_id"] for p in train_order} | {p["relation_id"] for p in val}
    assert not (train_rels & {p["relation_id"] for p in test_unseen}), "a held-out relation leaked into training"
    print("\nChecks passed: no subject appears in two splits; held-out relations are absent from train and validation.\n")

    print("Split sizes and balance (train shows the first 250 / 500 / 1000 prompts of the balanced order):")
    for n_ in (250, 500, 1000):
        describe(f"train[:{n_}]", train_order[:n_])
    describe("train (pool)", train_order)
    describe("val", val)
    describe("test_seen", test_seen)
    describe("test_unseen", test_unseen)

    if a.dry_run:
        print("\nDry run: nothing written.")
        return
    out = {"seed": a.seed, "source": "data/counterfact_hard_set.json", "heldout_relations": heldout,
           "dropped_small_relations": small,
           "note": "train is an ordered list: use the first N prompts (every prefix is balanced across relations and rank bands)",
           "train": [p["case_id"] for p in train_order], "val": [p["case_id"] for p in val],
           "test_seen": [p["case_id"] for p in test_seen], "test_unseen": [p["case_id"] for p in test_unseen]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    summary = {k: {"n": len(v), "relations": len({p["relation_id"] for p in v}),
                   "bands": dict(Counter(band_of(p["baseline_rank"]) for p in v))} for k, v in sets.items()}
    summary["train_first_1000"] = {"n": 1000, "relations": len({p["relation_id"] for p in train_order[:1000]}),
                                   "bands": dict(Counter(band_of(p["baseline_rank"]) for p in train_order[:1000]))}
    summary.update({"seed": a.seed, "heldout_relations": heldout})
    with open(a.out.replace(".json", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"\nWrote {a.out}\nWrote {a.out.replace('.json', '_summary.json')}")


if __name__ == "__main__":
    main()
