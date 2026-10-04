import sys, time, json
sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from src.sae_utils import load_base_model, load_sae_for_layer, get_default_device
from src.device_utils import sync_device
from src.editing import get_top_active_features, build_clean_context, get_target_token_id
from src.batched_eval import batched_ablation_probs

dev = get_default_device()
m = load_base_model(dev)
sae = load_sae_for_layer(8, dev)
HOOK = "blocks.8.hook_resid_pre"
for p in m.parameters():
    p.requires_grad_(False)

d = json.load(open('datasets/counterfact.json', encoding='utf8'))
recs = d[:6]
def sync():
    sync_device(dev)

print("device", dev)
rows = []
for r in recs:
    rw = r['requested_rewrite']
    prompt = rw['prompt'].format(rw['subject'])
    tgt = " " + rw['target_true']['str'].strip()
    ids = m.to_tokens(tgt, prepend_bos=False).reshape(-1)
    if ids.numel() != 1:
        continue
    tid = int(ids[0])
    toks = m.to_tokens(prompt)
    # 1) candidate cache build: clean ctx + top-200 active (all positions) + batched removal scoring
    sync(); t = time.time()
    ctx = build_clean_context(m, sae, prompt, tid)
    cands = get_top_active_features(m, sae, prompt, top_n=200, clean_ctx=ctx, positions="all")
    fids = [int(f) for f, _ in cands]
    probs = batched_ablation_probs(m, sae, toks, fids, 1.0, tid, HOOK)
    sync(); t_cache = time.time() - t
    # 2) one differentiable training step (full forward) and from-layer-8 forward
    with torch.no_grad():
        _, cache = m.run_with_cache(toks, names_filter=HOOK)
    resid = cache[HOOK]
    acts = sae.encode(resid)
    idx = torch.as_tensor(fids, device=dev)
    mult = torch.ones(len(fids), device=dev, requires_grad=True)
    def step(use_start):
        a = acts[..., idx]
        delta = ((mult - 1.0) * a) @ sae.W_dec[idx]
        if use_start:
            out = m(resid + delta, start_at_layer=8, tokens=toks)
        else:
            out = m.run_with_hooks(toks, fwd_hooks=[(HOOK, lambda r, hook: r + delta)])
        lp = F.log_softmax(out[0, -1], -1)
        loss = -lp[tid]
        g, = torch.autograd.grad(loss, mult)
        return g
    for use_start in (False, True):
        try:
            step(use_start)
            sync(); t = time.time()
            for _ in range(5):
                step(use_start)
            sync(); dt = (time.time() - t) / 5
        except Exception as e:
            dt = None
            print("start_at_layer failed:", repr(e)[:200])
        rows.append((prompt, len(fids), round(t_cache, 2), use_start, None if dt is None else round(dt * 1000)))
for r in rows:
    print(r)
# batch scaling of the training step (same-length prompts)
toks = m.to_tokens("The mother tongue of Danielle Darrieux is")
with torch.no_grad():
    _, cache = m.run_with_cache(toks, names_filter=HOOK)
resid = cache[HOOK]
for B in (1, 8, 32):
    rb = resid.repeat(B, 1, 1).clone()
    mult = torch.ones(B, 200, device=dev, requires_grad=True)
    idx = torch.arange(200, device=dev)
    acts = sae.encode(rb)
    def stepb():
        delta = torch.einsum('bpk,bk,kd->bpd', acts[..., idx], mult - 1.0, sae.W_dec[idx])
        out = m(rb + delta, start_at_layer=8, tokens=toks.repeat(B, 1))
        loss = -F.log_softmax(out[:, -1], -1)[:, 0].sum()
        return torch.autograd.grad(loss, mult)[0]
    stepb(); sync(); t = time.time()
    for _ in range(5):
        stepb()
    sync(); print("batch", B, "ms per step", round((time.time() - t) / 5 * 1000))
# correctness check: start_at_layer forward equals full forward with the same edit
delta = torch.zeros_like(resid); delta[:, -1, :5] = 0.5
with torch.no_grad():
    a = m.run_with_hooks(toks, fwd_hooks=[(HOOK, lambda r, hook: r + delta)])
    b = m(resid + delta, start_at_layer=8, tokens=toks)
print("max abs logit diff full vs start_at_layer:", float((a - b).abs().max()))
