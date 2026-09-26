"""
Analyse a safety-filter batch result from the command line (the Batch tab shows the same thing on screen).

    python tools/analyze_safety_batch.py outputs/safety_batches/pilot.json [--out docs/Research_Journal/images] [--tag e21]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.batch_analysis import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "Research_Journal", "images"))
    ap.add_argument("--tag", default="e21")
    a = ap.parse_args()
    with open(a.result, encoding="utf-8") as f:
        res = json.load(f)
    out = analyze(res, tag=a.tag)
    if not out["multi_arm"]:
        print("Only one arm in this result - nothing to compare.")
        return
    os.makedirs(a.out, exist_ok=True)
    for name, svg in out["figures"].items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            f.write(svg)
    with open(os.path.join(a.out, f"{a.tag}_summary.md"), "w", encoding="utf-8") as f:
        f.write(out["markdown"])
    print(out["markdown"])
    print(f"\nwrote {len(out['figures'])} figures and {a.tag}_summary.md to {a.out}")


if __name__ == "__main__":
    main()
