"""
Step A0 of docs/cross_model_transfer/PLAN.md: is everything in place for the translator study?

    .venv\\Scripts\\python.exe tools/transfer/00_setup_check.py            # full check
    .venv\\Scripts\\python.exe tools/transfer/00_setup_check.py --quick    # 20 sentences, validation text only

The first run DOWNLOADS (into your Hugging Face cache): GPT-2 medium (model.safetensors 1520 MB plus about 3 MB of tokenizer files)
and the wikitext-2 text (Salesforce/wikitext, config wikitext-2-raw-v1: train 6.4 MB, validation 0.7 MB, test 0.7 MB).

Checks, in order:
 1. DEVICE      which device is used and how much GPU memory is free
 2. MODELS      GPT-2 small and GPT-2 medium load; depth, width, vocabulary, size are printed
 3. TEXT        wikitext-2 loads
 4. TOKENIZERS  both models cut text into IDENTICAL tokens (same vocabulary, same special tokens, same ids on real sentences)
 5. FIDELITY    TransformerLens output matches the plain Hugging Face model (log-probabilities), for both models
 6. HOOKS+GPU   the hook points step A1 will read exist, a batch passes through both models on the GPU together without running out
                of memory, and peak GPU memory is reported
 7. TOKENS      wikitext-2 is cut into sequences of 128 tokens (first token = start-of-text) and saved for step A1

Writes (all under outputs/transfer/, which is kept out of git by a .gitignore created here):
    setup_check.json (or setup_check_quick.json), wikitext2_tokens.pt (or wikitext2_tokens_quick.pt)
Exit code 0 = every check passed, 1 = at least one failed. Stop and report if the tokenizers differ.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.device_utils import describe_device, device_memory_gb
from src.transfer.models import load_model, model_facts, pick_device, tokenizers_match

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
SMALL, MEDIUM = "gpt2", "gpt2-medium"
SEQ = 128
FIDELITY_TOL = 1e-2          # largest allowed difference in log-probability between TransformerLens and Hugging Face
FALLBACK_TEXTS = [
    "The Eiffel Tower is located in the city of Paris.",
    "In 1998 the company announced that it would open a new office in Melbourne.",
    "She opened the door and walked slowly into the dark room.",
    "The capital of Germany is Berlin, and the capital of France is Paris.",
    "He was born in a small village near the river.",
]

results = []   # (check name, ok, detail)


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  -> {'OK  ' if ok else 'FAIL'} {name}: {detail}", flush=True)


def finish(quick):
    print("\n" + "=" * 78)
    print("RESULT  step A0 (setup check)" + ("  [quick]" if quick else ""))
    for name, ok, detail in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<12} {detail}")
    all_ok = all(ok for _, ok, _ in results)
    print("  " + ("ALL CHECKS PASSED" if all_ok else "AT LEAST ONE CHECK FAILED: paste this block back"))
    print("=" * 78)
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "setup_check_quick.json" if quick else "setup_check.json"), "w", encoding="utf8") as f:
            json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "all_ok": all_ok,
                       "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in results]}, f, indent=1)
    except Exception as e:
        print("  (could not write the result file:", e, ")")
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 sentences, validation text only")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    a = ap.parse_args()
    t_start = time.time()

    # 1. DEVICE
    stage("1/7  device")
    device = pick_device(a.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, ".gitignore"), "w", encoding="utf8") as f:
        f.write("*\n!.gitignore\n")
    mem = device_memory_gb(device)
    record("device", True, describe_device(device) + (f", {mem[0]:.1f} of {mem[1]:.1f} GB free" if mem else "")
           + ("" if device != "cpu" else "  (CPU only: everything will be slow)"))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # 2. MODELS
    stage("2/7  loading GPT-2 small, then GPT-2 medium (the first run downloads medium, 1.5 GB: a progress bar will show)")
    try:
        small = load_model(SMALL, device)
        medium = load_model(MEDIUM, device)
    except Exception as e:
        record("models", False, f"could not load: {e!r}"[:300])
        return finish(a.quick)
    fs, fm = model_facts(small), model_facts(medium)
    record("models", True, f"{SMALL}: {fs['n_layers']} layers, width {fs['d_model']}, {fs['params_millions']} M params;  "
                           f"{MEDIUM}: {fm['n_layers']} layers, width {fm['d_model']}, {fm['params_millions']} M params")
    record("vocab_size", fs["d_vocab"] == fm["d_vocab"], f"{fs['d_vocab']} vs {fm['d_vocab']}")

    # 3. TEXT
    stage("3/7  wikitext-2 (the first run downloads about 8 MB)")
    texts, split_lines = FALLBACK_TEXTS, {}
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        for split in ("train", "validation"):
            split_lines[split] = ds[split]["text"]
        sentences = [l.strip() for l in split_lines["validation"] if 40 < len(l.strip()) < 400]
        n_want = 20 if a.quick else 200
        texts = sentences[:n_want]
        record("wikitext-2", True, f"train {len(split_lines['train'])} lines, validation {len(split_lines['validation'])} lines; "
                                   f"{len(texts)} sentences picked for the tokenizer check")
    except Exception as e:
        record("wikitext-2", False, f"could not load: {e!r}"[:300] + "  (continuing with 5 built-in sentences)")

    # 4. TOKENIZERS
    stage("4/7  tokenizers")
    ok, detail = tokenizers_match(small.tokenizer, medium.tokenizer, texts)
    if ok:
        same_bos = all(torch.equal(small.to_tokens(t).cpu(), medium.to_tokens(t).cpu()) for t in texts[:20])
        ok, detail = same_bos, detail + ("; to_tokens (with start token) identical" if same_bos else "; BUT to_tokens differs")
    record("tokenizers", ok, detail)

    # 5. FIDELITY
    stage("5/7  TransformerLens versus the plain Hugging Face model (log-probabilities)")
    try:
        from transformers import AutoModelForCausalLM
        probe = "The Eiffel Tower is located in the city of"
        for name, tl in ((SMALL, small), (MEDIUM, medium)):
            try:
                hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
            except TypeError:
                hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
            toks = tl.to_tokens(probe)
            with torch.no_grad():
                lp_tl = F.log_softmax(tl(toks).float(), dim=-1).cpu()
                lp_hf = F.log_softmax(hf(toks.cpu()).logits.float(), dim=-1)
            diff = (lp_tl - lp_hf).abs().max().item()
            top_tl, top_hf = lp_tl[0, -1].argmax().item(), lp_hf[0, -1].argmax().item()
            record(f"fidelity_{name}", diff < FIDELITY_TOL and top_tl == top_hf,
                   f"max log-prob difference {diff:.2e} (limit {FIDELITY_TOL:g}); next word: {tl.tokenizer.decode(top_tl)!r} vs {tl.tokenizer.decode(top_hf)!r}")
            del hf
    except Exception as e:
        record("fidelity", False, f"{e!r}"[:300])

    # 6. HOOKS + JOINT GPU RUN
    stage("6/7  hook points and a joint batch through both models")
    try:
        for name, m in ((SMALL, small), (MEDIUM, medium)):
            n = m.cfg.n_layers
            need = [f"blocks.0.hook_resid_pre", f"blocks.{n // 2}.hook_resid_pre", f"blocks.{n - 1}.hook_resid_post"]
            missing = [h for h in need if h not in m.hook_dict]
            record(f"hooks_{name}", not missing, "all present" if not missing else f"missing {missing}")
        g = torch.Generator().manual_seed(0)
        batch = torch.randint(0, fs["d_vocab"], (32, SEQ), generator=g).to(device)
        shapes = {}
        for name, m, layer in ((SMALL, small, 8), (MEDIUM, medium, 16)):
            hook = f"blocks.{layer}.hook_resid_pre"
            with torch.no_grad():
                _, cache = m.run_with_cache(batch, names_filter=lambda n_, h=hook: n_ == h)
            x = cache[hook]
            shapes[name] = tuple(x.shape)
            if not torch.isfinite(x).all():
                raise RuntimeError(f"{name}: non-finite values in {hook}")
            del cache, x
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None
        total = mem[1] if mem else None
        fits = True if peak is None or total is None else peak < 0.9 * total
        record("joint_run", fits, f"batch 32 x {SEQ} through both models, resid shapes {shapes}"
               + (f", peak GPU memory {peak:.2f} GB of {total:.1f} GB" if peak is not None else ""))
    except Exception as e:
        record("joint_run", False, f"{e!r}"[:300])

    # 7. TOKENS FOR STEP A1
    stage("7/7  cutting wikitext-2 into sequences of %d tokens for step A1" % SEQ)
    try:
        if not split_lines:
            raise RuntimeError("wikitext-2 did not load (see check 3)")
        tok = small.tokenizer
        bos = tok.bos_token_id
        packed, info = {}, []
        for split in (("validation",) if a.quick else ("train", "validation")):
            text = "".join(split_lines[split])
            ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"], dtype=torch.long)
            n_seq = len(ids) // (SEQ - 1)
            body = ids[: n_seq * (SEQ - 1)].view(n_seq, SEQ - 1)
            packed[split] = torch.cat([torch.full((n_seq, 1), bos, dtype=torch.long), body], dim=1)
            info.append(f"{split}: {len(ids):,} tokens -> {n_seq:,} sequences")
        packed["seq_len"], packed["bos_id"] = SEQ, bos
        path = os.path.join(OUT_DIR, "wikitext2_tokens_quick.pt" if a.quick else "wikitext2_tokens.pt")
        torch.save(packed, path)
        record("tokens", True, "; ".join(info) + f"; saved {os.path.relpath(path, ROOT)}")
    except Exception as e:
        record("tokens", False, f"{e!r}"[:300])

    print(f"\n(total time {time.time() - t_start:.0f} s)")
    return finish(a.quick)


if __name__ == "__main__":
    sys.exit(main())
