"""Command-line entry point for the candidate-source batch (same engine as the Streamlit batch tab).

    python run_batch.py data/candidate_source_batch_spec.json [output.json]
"""
import json
import sys
from datetime import datetime

from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.batch_runner import run_batch, parse_spec


def main():
    spec_path = sys.argv[1] if len(sys.argv) > 1 else "data/candidate_source_batch_spec.json"
    out = sys.argv[2] if len(sys.argv) > 2 else f"outputs/batch_runs/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with open(spec_path, encoding="utf-8") as f:
        spec = parse_spec(json.load(f))
    device = get_default_device()
    model = load_base_model()
    sae = load_sae_for_layer(layer=layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{layer}.hook_resid_pre")

    def cb(done, total, rec):
        s = rec.get("error") or f"rank {rec.get('baseline_rank')} -> {(rec.get('best_result') or {}).get('rank')} steps {len(rec.get('hybrid_details') or [])}"
        print(f"[{done}/{total}] {rec['candidate_source']:<21} {rec['prompt'][:48]!r:<52} {s}", flush=True)

    res = run_batch(model, sae, hook_name, layer, device, spec, progress_cb=cb, save_path=out)
    print("\nAGGREGATE:", json.dumps(res["aggregate"], indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
