"""
Step D13 of docs/cross_model_transfer/PLAN.md: many more, and more varied, edits made in GPT-2 small, with random target words (synthetic targets). The design is in the
plan (written before this tool was built).

    .venv\\Scripts\\python.exe tools/transfer/21_varied_edits.py --smoke      # 6 sentences, one process, about a minute (numbers mean nothing)
    .venv\\Scripts\\python.exe tools/transfer/21_varied_edits.py              # the real run: 3 worker processes in parallel, about an hour, resumable

For every sentence of the D7 pool (minus the sentences whose subject is the subject of a dev or test record) 2 random words are drawn (single-token alphabetic words, rank-matched to
the sentence's own counterfactual target; never a true or counterfactual word of any dev or test record) and the project's gradient-descent edit is tuned in small toward each
(200 features, 100 steps, all positions, as in D2). Each worker saves its shard (outputs/transfer/d13_edits_shard<k>.pt) and its own log; the parent prints the progress of all
workers every minute and merges the shards into outputs/transfer/d13_edits.pt. Stopping and running the same command again resumes.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.editing import build_clean_context
from src.gradient_editing import run_gradient_descent_edit
from src.sae_utils import load_sae_for_layer
from src.transfer.models import load_model, pick_device
from src.transfer.runlog import tee_to

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
HOOK = "blocks.8.hook_resid_pre"
PER_SENTENCE = 2


def load_jobs(limit=None):
    """Sentences (pool minus the subject hold-out) and the banned word ids (true and counterfactual words of every dev and test record)."""
    pool = json.load(open(os.path.join(OUT_DIR, "d7_pool.json"), encoding="utf8"))
    bench = json.load(open(os.path.join(OUT_DIR, "d_benchmark.json"), encoding="utf8"))
    held_subjects = {r["subject"] for r in bench}
    banned = {r["tid_new"] for r in bench} | {r["tid_true"] for r in bench}
    recs = [r for r in pool if r["subject"] not in held_subjects]
    if limit:
        recs = recs[:limit]
    return recs, banned


def pick_random_word(probs, ok_mask, tid_new, tid_true, start_rank, rng, banned):
    """A random alphabetic single-token word whose rank on this prompt is within 0.7 to 1.4 (then 0.5 to 2, then 0.3 to 4) times start_rank."""
    order = torch.argsort(probs, descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(1, len(order) + 1, device=order.device)
    ban = torch.zeros_like(ok_mask)
    ban[list(banned | {tid_new, tid_true})] = True
    for lo, hi in ((0.7, 1.4), (0.5, 2.0), (0.3, 4.0)):
        cand = ok_mask & ~ban & (rank_of >= max(2, lo * start_rank)) & (rank_of <= hi * start_rank)
        ids = cand.nonzero().flatten().tolist()
        if ids:
            w = rng.choice(ids)
            return w, int(rank_of[w])
    return None, None


def shard_path(k, smoke):
    return os.path.join(OUT_DIR, f"d13_edits_shard{k}{'_smoke' if smoke else ''}.pt")


def run_worker(k, n, smoke, limit):
    print(f"log: {os.path.relpath(tee_to(f'21_varied_edits_w{k}'), ROOT)}", flush=True)
    dev = pick_device(None)
    recs, banned = load_jobs(limit)
    jobs = [(i, m) for i in range(len(recs)) for m in range(PER_SENTENCE)]
    mine = [jb for jb in range(len(jobs)) if jb % n == k]
    small, sae = load_model("gpt2", dev), load_sae_for_layer(8, dev)
    names = small.tokenizer.convert_ids_to_tokens(list(range(small.cfg.d_vocab)))
    ok_mask = torch.tensor([bool(t.startswith("Ġ") and t[1:].isalpha() and len(t) > 3) for t in names], device=dev)
    path = shard_path(k, smoke)
    res = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    prog = os.path.join(OUT_DIR, f"d13_progress_{k}{'_smoke' if smoke else ''}.txt")
    t0 = time.time()
    done0 = len(res)
    print(f"worker {k} of {n}: {len(mine)} jobs, {done0} already done", flush=True)
    for c, jb in enumerate(mine):
        i, m = jobs[jb]
        r = recs[i]
        key = f"{r['case_id']}_{m}"
        if key in res:
            continue
        ctx0 = build_clean_context(small, sae, r["prompt"], r["tid_new"])
        rng = random.Random(7000 + r["index"] * 10 + m)
        w, _ = pick_random_word(ctx0.clean_probs, ok_mask, r["tid_new"], r["tid_true"], ctx0.clean_rank, rng, banned)
        if w is None:
            res[key] = {"skipped": True}
        else:
            ctx = build_clean_context(small, sae, r["prompt"], w)
            out = run_gradient_descent_edit(small, sae, ctx, w, r["prompt"], 8, HOOK, top_n=200, steps=100)
            res[key] = {"case_id": r["case_id"], "m": m, "tid": w, "word": small.tokenizer.decode([w]), "fids": out["fids"], "a": out["a"], "start_rank": ctx.clean_rank,
                        "rank": out["real_path"]["rank"], "kl": out["real_path"]["kl"]}
        if len(res) % 5 == 0:
            torch.save(res, path)
            ok = sum(1 for v in res.values() if v.get("rank") == 1)
            open(prog, "w").write(f"{len(res)} {len(mine)} {ok} {time.time() - t0:.0f} {done0}")
            print(f"   {len(res)}/{len(mine)} done, rank 1 in small on {ok}; {time.time() - t0:.0f} s", flush=True)
    torch.save(res, path)
    ok = sum(1 for v in res.values() if v.get("rank") == 1)
    open(prog, "w").write(f"{len(res)} {len(mine)} {ok} {time.time() - t0:.0f} {done0}")
    print(f"worker {k} finished: {len(res)} edits, rank 1 in small on {ok}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.worker is not None:
        run_worker(a.worker, a.of, a.smoke, a.limit)
        return
    log_path = tee_to("21_varied_edits")
    print(f"log: {os.path.relpath(log_path, ROOT)}", flush=True)
    if a.smoke:
        run_worker(0, 1, True, 6)
        res = torch.load(shard_path(0, True), weights_only=False)
        ok = [v for v in res.values() if v.get("rank") == 1]
        print(f"smoke: {len(res)} edits, rank 1 in small on {len(ok)}; examples: " + ", ".join(f"{v['word']!r} (start rank {v['start_rank']})" for v in list(res.values())[:6] if "word" in v))
        return
    recs, banned = load_jobs()
    n_jobs = len(recs) * PER_SENTENCE
    print(f"{len(recs)} sentences x {PER_SENTENCE} random words = {n_jobs} edits to tune (about 3 s each), {a.workers} workers; {len(banned)} words are banned as targets", flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(k), "--of", str(a.workers)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for k in range(a.workers)]
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        tot = ok = done0 = 0
        el = 0
        for k in range(a.workers):
            f = os.path.join(OUT_DIR, f"d13_progress_{k}.txt")
            if os.path.exists(f):
                d, mine, o, e, d0 = (float(x) for x in open(f).read().split())
                tot, ok, done0, el = tot + d, ok + o, done0 + d0, max(el, e)
        new = tot - done0
        rate = new / max(time.time() - t0, 1)
        left = (n_jobs - tot) / rate / 60 if rate > 0 else float("nan")
        print(f"   [{time.strftime('%H:%M:%S')}] {int(tot)}/{n_jobs} edits done, rank 1 in small on {int(ok)} ({100 * ok / max(tot, 1):.0f}%); about {left:.0f} min left", flush=True)
    merged = {}
    for k in range(a.workers):
        p = shard_path(k, False)
        if os.path.exists(p):
            merged.update(torch.load(p, weights_only=False))
    torch.save(merged, os.path.join(OUT_DIR, "d13_edits.pt"))
    ok = [v for v in merged.values() if v.get("rank") == 1]
    words = {v["tid"] for v in ok}
    print(f"\nmerged {len(merged)} edits into outputs/transfer/d13_edits.pt; {len(ok)} reached rank 1 in small ({100 * len(ok) / max(len(merged), 1):.0f}%), {len(words)} distinct target words, {len({v['case_id'] for v in ok})} sentences")
    print(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
