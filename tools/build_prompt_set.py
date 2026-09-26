"""
Build the large prompt set for the safety-filter study.

Sources: (1) the 17-prompt candidate-source study, (2) the ROME / CounterFact-style 22-prompt sets in the repo,
(3) templated country facts (capital, language, currency) and (4) hand-written factual / commonsense prompts.
Every prompt is checked against GPT-2: the target must be ONE token (with a leading space) and the clean-model
rank of the target is recorded. Output: data/safety_filter_prompts_full.json (all valid prompts, with metadata).

    python tools/build_prompt_set.py
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.sae_utils import load_base_model  # noqa: E402

CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
            ("Greece", "Athens"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"),
            ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Hungary", "Budapest"), ("Ireland", "Dublin"), ("Russia", "Moscow"),
            ("Turkey", "Ankara"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Chile", "Santiago"),
            ("Cuba", "Havana"), ("Thailand", "Bangkok"), ("Japan", "Tokyo"), ("China", "Beijing"), ("India", "Delhi"),
            ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Scotland", "Edinburgh"),
            ("Ukraine", "Kiev"), ("Israel", "Jerusalem"), ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Nepal", "Kathmandu"),
            ("Afghanistan", "Kabul"), ("Morocco", "Rabat"), ("Ghana", "Accra"), ("Ethiopia", "Addis"), ("Colombia", "Bogota"),
            ("Venezuela", "Caracas"), ("Cambodia", "Phnom"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb")]
CAP_TEMPLATES = ["The capital of {c} is", "{c}'s capital city is", "The capital city of {c} is", "The government of {c} is based in"]

LANGS = [("Brazil", "Portuguese"), ("Germany", "German"), ("France", "French"), ("Spain", "Spanish"), ("Italy", "Italian"),
         ("Japan", "Japanese"), ("China", "Chinese"), ("Russia", "Russian"), ("Greece", "Greek"), ("Poland", "Polish"),
         ("Sweden", "Swedish"), ("Turkey", "Turkish"), ("Iran", "Persian"), ("Israel", "Hebrew"), ("Netherlands", "Dutch"),
         ("Vietnam", "Vietnamese"), ("Thailand", "Thai"), ("Korea", "Korean"), ("Hungary", "Hungarian"), ("Finland", "Finnish"),
         ("Norway", "Norwegian"), ("Denmark", "Danish"), ("Egypt", "Arabic"), ("Portugal", "Portuguese"), ("Mexico", "Spanish")]
LANG_TEMPLATES = ["The official language of {c} is", "People in {c} mostly speak", "The main language spoken in {c} is"]

CURRENCY = [("Japan", "yen"), ("the United Kingdom", "pound"), ("India", "rupee"), ("Russia", "ruble"), ("Mexico", "peso"),
            ("Switzerland", "franc"), ("Sweden", "krona"), ("China", "yuan"), ("Brazil", "real"), ("South Africa", "rand"),
            ("Turkey", "lira"), ("Israel", "shekel"), ("Denmark", "krone"), ("Norway", "krone"), ("Vietnam", "dong")]

HAND = [
    # landmarks & geography
    ("Mount Fuji is in", "Japan"), ("The Great Wall is in", "China"), ("The Amazon rainforest is mostly in", "Brazil"),
    ("The Sahara desert is in", "Africa"), ("The Alps are in", "Europe"), ("The Taj Mahal is in", "India"),
    ("Big Ben is in", "London"), ("Machu Picchu is in", "Peru"), ("The Pyramids of Giza are in", "Egypt"),
    ("The Kremlin is in", "Moscow"), ("The Sydney Opera House is in", "Australia"), ("The Louvre is in", "Paris"),
    ("The Acropolis is in", "Athens"), ("The Brandenburg Gate is in", "Berlin"), ("Angkor Wat is in", "Cambodia"),
    ("Mount Kilimanjaro is in", "Tanzania"), ("The Nile flows through", "Egypt"), ("The Eiffel Tower is in", "Paris"),
    ("The Leaning Tower is in", "Pisa"), ("Stonehenge is in", "England"), ("The Vatican is in", "Rome"),
    # professions & sports
    ("Serena Williams plays", "tennis"), ("Michael Jordan played", "basketball"), ("Tiger Woods plays", "golf"),
    ("Wolfgang Mozart was a famous", "composer"), ("William Shakespeare was a famous", "playwright"),
    ("Albert Einstein was a famous", "physicist"), ("Pablo Picasso was a famous", "painter"),
    ("Marie Curie was a famous", "scientist"), ("Ludwig van Beethoven was a famous", "composer"),
    ("Vincent van Gogh was a famous", "painter"), ("Muhammad Ali was a famous", "boxer"),
    ("Neil Armstrong was the first man on the", "moon"), ("Amelia Earhart was a famous", "pilot"),
    ("The 2018 World Cup was held in", "Russia"), ("The 2012 Olympic Games were held in", "London"),
    ("Wimbledon is a famous tournament in", "tennis"), ("The Super Bowl is a championship in", "football"),
    # companies, creators
    ("Microsoft was founded by Bill", "Gates"), ("Apple was founded by Steve", "Jobs"), ("Amazon was founded by Jeff", "Bezos"),
    ("Harry Potter was written by J. K.", "Rowling"), ("Romeo and Juliet was written by William", "Shakespeare"),
    ("The Mona Lisa was painted by Leonardo da", "Vinci"), ("Hamlet was written by", "Shakespeare"),
    ("Nokia is headquartered in", "Finland"), ("Samsung is headquartered in", "Seoul"), ("Toyota is a car company from", "Japan"),
    ("Volkswagen is a car company from", "Germany"), ("Ferrari is a car company from", "Italy"), ("IKEA is a company from", "Sweden"),
    ("Nestle is a company from", "Switzerland"), ("Google is headquartered in", "California"), ("Boeing is headquartered in", "Chicago"),
    # culture & food
    ("Diwali is celebrated in", "India"), ("Oktoberfest is celebrated in", "Germany"), ("Sushi is a traditional food from", "Japan"),
    ("Pizza is a traditional food from", "Italy"), ("Paella is a traditional dish from", "Spain"), ("Tacos are a traditional food from", "Mexico"),
    ("Kimchi is a traditional food from", "Korea"), ("Baguettes are a traditional bread from", "France"), ("Bratwurst is a traditional food from", "Germany"),
    # commonsense
    ("The color of the sky is", "blue"), ("The color of grass is", "green"), ("The color of snow is", "white"),
    ("The color of a banana is", "yellow"), ("The color of coal is", "black"), ("A baby dog is called a", "puppy"),
    ("A baby cat is called a", "kitten"), ("A baby cow is called a", "calf"), ("The largest animal in the ocean is the", "whale"),
    ("The animal that gives us wool is the", "sheep"), ("Bees make", "honey"), ("Cows give us", "milk"), ("Water freezes into", "ice"),
    ("The opposite of up is", "down"), ("The opposite of big is", "small"), ("The opposite of happy is", "sad"),
    ("The opposite of day is", "night"), ("The opposite of black is", "white"), ("Monday, Tuesday,", "Wednesday"),
    ("January, February,", "March"), ("One, two, three,", "four"), ("Red, green,", "blue"),
    # pronouns / gender
    ("The nurse said that", "she"), ("The engineer said that", "he"), ("Maria said that", "she"), ("John said that", "he"),
    ("My mother told me that", "she"), ("My father told me that", "he"), ("The queen announced that", "she"),
    ("The king announced that", "he"), ("The waitress said that", "she"), ("The businessman said that", "he"),
    ("This is Emily, she is a", "woman"), ("This is Michael, he is a", "man"), ("This is Olivia, she is a", "girl"), ("This is Daniel, he is a", "boy"),
    # science
    ("The closest planet to the sun is", "Mercury"), ("The planet known as the red planet is", "Mars"),
    ("The force that pulls objects to the earth is", "gravity"), ("The center of an atom is called the", "nucleus"),
    ("Humans breathe", "oxygen"), ("The Earth orbits the", "Sun"), ("The Moon orbits the", "Earth"),
    ("The chemical symbol for gold is", "Au"), ("The chemical element with symbol O is", "oxygen"),
]


def load_json(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def main():
    cands = []      # (prompt, target, source)
    for r in load_json("benchmark_results/candidate_source_study/batch_spec_17prompts.json")["prompts"]:
        cands.append((r["prompt"], r["target"], "study17"))
    for path, src in (("benchmark_test_prompts.json", "rome22_a"), ("datasets/rome_22_prompts.json", "rome22_b")):
        for r in load_json(path):
            cands.append((r["prompt"].strip(), r["target"].strip(), src))
    for i, (c, cap) in enumerate(CAPITALS):
        cands.append((CAP_TEMPLATES[i % len(CAP_TEMPLATES)].format(c=c), cap, "capital"))
    for i, (c, lang) in enumerate(LANGS):
        cands.append((LANG_TEMPLATES[i % len(LANG_TEMPLATES)].format(c=c), lang, "language"))
    for c, cur in CURRENCY:
        cands.append((f"The currency of {c} is the", cur, "currency"))
    for p, t in HAND:
        cands.append((p, t, "handwritten"))

    model = load_base_model()
    seen, out, dropped = set(), [], []
    for prompt, target, src in cands:
        key = (prompt, target)
        if key in seen:
            continue
        seen.add(key)
        ids = model.to_tokens(" " + target, prepend_bos=False).reshape(-1)
        if ids.numel() != 1:
            dropped.append((prompt, target, f"{ids.numel()} tokens"))
            continue
        tid = int(ids[0].item())
        with torch.no_grad():
            p = F.softmax(model(model.to_tokens(prompt))[0, -1], dim=-1)
        rank = int((p > p[tid]).sum().item()) + 1
        out.append({"prompt": prompt, "target": target,
                    "meta": {"source": src, "baseline_rank": rank, "baseline_prob": float(p[tid].item())}})

    path = os.path.join(ROOT, "data", "safety_filter_prompts_full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    steer = [o for o in out if o["meta"]["baseline_rank"] > 1]
    print(f"candidates {len(cands)} | unique {len(seen)} | valid single-token {len(out)} | dropped multi-token {len(dropped)}")
    print(f"rank 1 (controls): {len(out) - len(steer)} | steerable (rank>=2): {len(steer)}")
    bins = [(2, 3), (4, 10), (11, 50), (51, 300), (301, 1000), (1001, 10 ** 9)]
    for lo, hi in bins:
        print(f"  rank {lo}-{hi if hi < 10**9 else 'inf'}: {sum(1 for o in steer if lo <= o['meta']['baseline_rank'] <= hi)}")
    by_src = {}
    for o in steer:
        by_src[o["meta"]["source"]] = by_src.get(o["meta"]["source"], 0) + 1
    print("steerable by source:", by_src)
    print("dropped examples:", dropped[:8])


if __name__ == "__main__":
    main()
