"""
Build a labeled paper pack (figures + tables + data + provenance + journal draft) from a finished batch result.

    python tools/make_paper_pack.py outputs/safety_batches/safety_filter_spec_pilot.json --entry 21 [--label pilot]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.batch_analysis import analyze  # noqa: E402
from src.paper_pack import build_pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--entry", default="21", help="journal entry number this pack belongs to")
    ap.add_argument("--label", default=None, help="short name for the pack folder (default: result file name)")
    a = ap.parse_args()
    with open(a.result, encoding="utf-8") as f:
        res = json.load(f)
    an = analyze(res)
    if not an["multi_arm"]:
        print("This result has a single arm; the paper pack needs a multi-arm study.")
        return
    info = build_pack(a.result, res, an, entry=a.entry, label=a.label)
    print(f"pack: {info['pack_dir']}\nstatus: {info['status']}\nfigures: {len(info['manifest']['figures'])}, tables: {len(info['manifest']['tables'])}")


if __name__ == "__main__":
    main()
