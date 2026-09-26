"""Command-line entry point for prompt batches (same engine as the Streamlit batch tab).

    python run_batch.py SPEC.json [OUTPUT.json] [--layer 8] [--resume]

--resume continues an interrupted run: finished runs already in OUTPUT.json are kept and skipped.
"""
import argparse
import json
from datetime import datetime

from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.batch_runner import run_batch, parse_spec, write_progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default="data/candidate_source_batch_spec.json")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: this worker runs every n-th job starting at i (parallel workers)")
    a = ap.parse_args()
    out = a.out or f"outputs/batch_runs/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(a.spec, encoding="utf-8") as f:
        spec = parse_spec(json.load(f))
    device = get_default_device()
    model = load_base_model()
    sae = load_sae_for_layer(layer=a.layer)
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{a.layer}.hook_resid_pre")
    shard = None
    if a.shard:
        i_, n_ = a.shard.split("/")
        shard = (int(i_), int(n_))
    total = len(spec["prompts"]) * len(spec["arms"]) * len(spec["modes"]) * spec["repeats"]
    if shard:
        total = len(range(shard[0], total, shard[1]))          # approximate for the banner; the runner reports the exact count
    print(f"{spec['name']}: {total} runs -> {out}", flush=True)

    def cb(done, total, rec):
        best = (rec.get("best_result") or {}).get("rank")
        s = rec.get("error") or f"rank {rec.get('baseline_rank')} -> {best} ok={rec.get('success')} steps {len(rec.get('hybrid_details') or [])} {rec.get('total_time_s')}s"
        print(f"[{done}/{total}] {rec.get('arm', '-'):<14} {rec['candidate_source'][:5]:<6} {rec['prompt'][:46]!r:<50} {s}", flush=True)

    progress_path = out + ".progress.json"
    try:
        res = run_batch(model, sae, hook_name, a.layer, device, spec, progress_cb=cb, save_path=out, resume=a.resume,
                        progress_path=progress_path, shard=shard)
    except BaseException as e:                     # make a crash visible to the Batch tab, then re-raise
        write_progress(progress_path, {"name": spec["name"], "total": total, "done": None, "finished": False,
                                       "error": f"{type(e).__name__}: {e}", "out": out,
                                       "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        raise
    print("\nDONE. errors:", res["meta"].get("errors"), "| elapsed s:", res["meta"].get("elapsed_s"))
    ac = res.get("arm_comparison")
    if ac:
        print("\nPER ARM:")
        for k, v in ac["per_arm_mode"].items():
            print(f"  {k:<22} rank1 {v['reached_rank1']}/{v['n_prompts']}  mean rank gain {v['mean_rank_gain']}  features {v['mean_features_at_best']}  KL {v['mean_kl_nats']}  time {v['mean_time_s']}s")
    print("Saved:", out)


if __name__ == "__main__":
    main()
