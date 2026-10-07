# Local patches to vendored ROME

Upstream: https://github.com/kmeng01/rome
Pinned commit: 0874014 (MIT license)

Every change made to files under third_party/rome is listed here.

| File | Change | Why |
|------|--------|-----|
| util/globals.py | Added DEVICE (env ROME_DEVICE, default cpu) | No CUDA on Mac |
| rome/compute_u.py, compute_v.py, layer_stats.py, util/perplexity.py, experiments/py/eval_utils_*.py, experiments/causal_trace.py, experiments/evaluate.py | Replaced hardcoded "cuda" and .cuda() with DEVICE | Run on CPU or MPS |
| experiments/evaluate.py | Removed baseline imports and ALG_DICT entries (FT, KN, MEND, KE); added gpt2 to model choices | Baselines need extra packages; only ROME is used |
| rome/layer_stats.py | Added gpt2 to model choices; pin_memory=False, num_workers=0 in tally | CPU/macOS; no CUDA pinning, avoid worker hangs |
| rome/layer_stats.py | load_dataset uses Salesforce/wikitext when ds_name is wikitext | Short dataset alias no longer resolves in current huggingface_hub |
