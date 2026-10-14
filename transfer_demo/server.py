"""
Backend for the cross-model transfer demo (the React page in transfer_demo/web talks to this).

WHAT THIS FILE IS
It is the Streamlit tab (src/transfer/transfer_tab.py) with the Streamlit parts taken out and a small web server put around it.
The science is NOT rewritten: run_transfer_edit() below is the same code as in the tab, line for line, apart from three things:
  1. the cache that lived in st.session_state is now a plain dict (GD_CACHE),
  2. the progress callback also sends a structured "step" event (step number and current rank) so the page can draw a live ladder,
  3. the result also carries the clean (before) numbers of both models, which the tab kept in session state.
Every number that goes into the edit (100 steps, 200 candidate features, dose, the ridge / neural / trained maps, the layer 8 -> 16 pair)
is chosen by the page and passed straight through. Nothing is added or tuned here.

WHY STARLETTE AND NOT FASTAPI
Starlette (the web toolkit FastAPI is built on) and uvicorn are already in the project's .venv because Streamlit needs them, so nothing extra
has to be installed. We only need four routes, so FastAPI's extras would not be used.

WHY ONE PROCESS HOLDS THE MODELS
GPT-2 Small, its SAE and GPT-2 Medium take about 3.5 GB of the 6.1 GB card. They are loaded once when the server starts and kept, exactly as
st.cache_resource did in the app. Only one edit runs at a time (RUN_LOCK) because two at once would fight for the same GPU memory.

RUN IT (from the repository root, with the project's venv)
    $env:HF_HUB_OFFLINE=1; .venv\\Scripts\\python.exe transfer_demo\\server.py
Close the Streamlit app first: it holds its own copy of the same models and the two together do not fit on a 6 GB card.
"""
import contextlib
import csv
import json
import os
import queue
import sys
import threading
import time

# Models and SAEs come from the local Hugging Face cache; nothing is downloaded when the server starts (same advice as the project's CLAUDE.md).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Let "python transfer_demo/server.py" find the project's src/ folder, whichever directory it is started from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # the map files are found by relative paths such as outputs/transfer/..., as in the app

import torch                                                    # noqa: E402
from starlette.applications import Starlette                    # noqa: E402
from starlette.responses import FileResponse, JSONResponse, StreamingResponse   # noqa: E402
from starlette.routing import Mount, Route                      # noqa: E402
from starlette.staticfiles import StaticFiles                   # noqa: E402

# The two modules whose gradient-descent loops compute the dial values (see DialRecorder below).
import src.gradient_editing as _ge                              # noqa: E402
import src.transfer.b_aware_edit as _bae                        # noqa: E402

# Same imports as the Streamlit tab: the project's own functions, unchanged.
from src.editing import build_clean_context, get_target_token_id, get_top_active_features, tokens_without_bos   # noqa: E402
from src.gradient_editing import run_gradient_descent_edit      # noqa: E402
from src.sae_utils import get_default_device, load_base_model, load_sae_for_layer   # noqa: E402
from src.transfer.b_aware_edit import tune_through_translator   # noqa: E402
from src.transfer.mlp_maps import load_mlp_translator           # noqa: E402
from src.transfer.models import load_model, model_facts         # noqa: E402
from src.transfer.stitch import states                          # noqa: E402
from src.transfer.swap import run_injected                      # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------------------------
# Constants copied from the tab (same values, same meaning)
# ---------------------------------------------------------------------------------------------------------------------------------------
SMALL_LAYER = 8      # the SAE layer used throughout the project
MEDIUM_LAYER = 16    # the layer paired with it in the study; 8 -> 16 was the primary pair

MAPS_SUMMARY = os.path.join("docs", "Research_Journal", "packs", "cross_model_transfer", "maps_summary.csv")   # quality numbers of the fitted maps (in git)
MAPS_FILE = os.path.join("outputs", "transfer", "maps_gpt2_to_gpt2-medium.pt")     # the 25 fitted ridge maps (local only, not in git)
TRAINED_MAP_FILE = os.path.join("outputs", "transfer", "d7_map.pt")                # ridge map nudged on 289 edits (D7)
MLP_MAP_FILE = os.path.join("outputs", "transfer", "mlp_maps.pt")                  # neural maps
STUDY_PAIRS = {(8, 16): "primary pair of the study", (8, 12): "second pair of the study"}

# The two edit mechanisms. The exact strings matter: run_transfer_edit decides "additive or not" by looking at how the string starts.
EDIT_METHODS = {
    "Gradient descent (default)": "One multiplier (0 to 3) for each of 200 candidate SAE features of Small, tuned on this prompt by 100 Adam steps "
                                  "(rank term, KL term, size term). The main method of the study.",
    "Gradient descent + additive (silent features)": "As above, and in addition an amount of up to 200 silent features is added at the last position. "
                                                     "Strong enough to push almost any word to rank 1, so treat it as a control and not as a fix.",
}
MAP_CHOICES = {
    "ridge": "Ridge map (fitted on ordinary text)",
    "mlp": "Neural map (ridge plus a small MLP correction)",
    "trained": "Ridge map trained on edits (D7)",
    "direct": "Direct: tune the dials against Medium's output (D6)",
}
MAP_HELP = {
    "ridge": "One linear map fitted once on Wikitext. The edit is made in Small, its change is translated and added to Medium. Medium's output is never looked at.",
    "mlp": "Same idea with a nonlinear map: translated change = f(h + change) - f(h). Fits text better, exported no better in the study.",
    "trained": "The ridge matrix nudged on 289 edits so that Medium's answers move as Small's do (8 to 16 only, dose 2.0). Medium's output is not looked at when you run it.",
    "direct": "Not a map: the 200 dials are tuned through the ridge map against Medium's own output for this sentence (the D6 method). Needs Small layer 8.",
}

# ---------------------------------------------------------------------------------------------------------------------------------------
# Models, loaded once and kept (this replaces st.cache_resource)
# ---------------------------------------------------------------------------------------------------------------------------------------
class State:
    device = None
    small = None
    sae = None            # the SAE for layer 8
    sae_layer = SMALL_LAYER
    medium = None
    loading = True        # True until the models are in memory
    error = None          # text of the failure if loading failed, so the page can show it


S = State()
RUN_LOCK = threading.Lock()   # one edit at a time (GPU memory)
GD_CACHE = {}                 # remembers Small's gradient-descent edit per (prompt, target, settings), see run_transfer_edit


def load_everything():
    """Load Small, its SAE and Medium. Runs in a background thread so the server answers /api/status while it works."""
    try:
        S.device = os.environ.get("DEMO_DEVICE") or get_default_device()
        S.small = load_base_model(device=S.device)                       # GPT-2 Small, the same loader as the app
        S.sae = load_sae_for_layer(SMALL_LAYER, device=S.device)         # the layer-8 SAE, the same loader as the app
        S.medium = load_model("gpt2-medium", device=S.device)            # about 1.4 GB on the GPU
    except Exception as e:   # shown on the page instead of a silent hang
        S.error = f"{type(e).__name__}: {e}"
    finally:
        S.loading = False


# ---------------------------------------------------------------------------------------------------------------------------------------
# Map files. Failures are NOT cached: the Streamlit tab cached "no neural maps" once when the folder was missing and kept showing it until a restart.
# ---------------------------------------------------------------------------------------------------------------------------------------
_RIDGE = {}
_MLP_KEYS = {}


def ridge_maps():
    if "maps" not in _RIDGE:
        _RIDGE["maps"] = torch.load(MAPS_FILE)["maps"]
    return _RIDGE["maps"]


def mlp_keys():
    if "keys" not in _MLP_KEYS:
        try:
            _MLP_KEYS["keys"] = set(torch.load(MLP_MAP_FILE, weights_only=False).keys())
        except OSError:
            return set()          # not stored, so it is looked for again next time
    return _MLP_KEYS["keys"]


def load_translation_pairs() -> dict:
    """Fitted Small -> Medium translators and how good they are: {(small_layer, medium_layer): {r2, cos}}."""
    pairs = {}
    try:
        with open(MAPS_SUMMARY, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["dir"] == "s2m":
                    pairs[(int(r["src_layer"]), int(r["dst_layer"]))] = {"r2": float(r["perdim_r2"]), "cos": float(r["cosine"])}
    except OSError:
        pass
    return pairs


def available_maps(pair):
    """Which translation options exist for a layer pair (the neural map needs its entry in the file; D7 exists for 8 -> 16 only; Direct needs Small layer 8)."""
    sl, ml = pair
    av = ["ridge"]
    if f"s2m_L{sl}_L{ml}" in mlp_keys():
        av.append("mlp")
    if pair == (8, 16) and os.path.exists(TRAINED_MAP_FILE):
        av.append("trained")
    if sl == 8:
        av.append("direct")
    return av


# ---------------------------------------------------------------------------------------------------------------------------------------
# Watching the dials move (new; only used by the dial visualiser on the page)
#
# In both gradient-descent loops (src/gradient_editing.py for Small's edit, src/transfer/b_aware_edit.py for Direct) the optimiser holds one
# raw number z per candidate feature, and once per step turns it into the multiplier change `a` by calling _to_a(z). The multiplier the
# edit applies to feature k is m_k = 1 + a_k, between 0 and 3, and it starts at exactly 1 (no change).
# Those loops never hand `a` out while they run. To show it live WITHOUT editing the project's files, we swap _to_a for a copy that
# does the same thing and also writes down what it returned. The returned tensor is the original one, untouched, so the edit is
# numerically identical. The swap is undone as soon as the run ends, and it only happens inside RUN_LOCK (one run at a time).
# ---------------------------------------------------------------------------------------------------------------------------------------
class DialRecorder:
    def __init__(self, emit):
        self.frames = []     # one list of 200 multipliers per gradient step (step 0 is the start: all 1.0)
        self.emit = emit

    def __enter__(self):
        self._orig = (_ge._to_a, _bae._to_a)

        def recording(z, _orig=_ge._to_a):
            a = _orig(z)
            m = [round(float(x), 3) for x in (1.0 + a.detach()).cpu().tolist()]
            self.frames.append(m)
            self.emit({"type": "dials", "step": len(self.frames) - 1, "m": m})   # goes to the page straight away
            return a

        _ge._to_a = recording
        _bae._to_a = recording
        return self

    def __exit__(self, *exc):
        _ge._to_a, _bae._to_a = self._orig


TRAJ_CACHE = {}   # the recorded dial movements of each cached gradient-descent edit, so a re-run that reuses the edit can still show them


# ---------------------------------------------------------------------------------------------------------------------------------------
# Reading a model's next-word ranking (same two helpers as the tab)
# ---------------------------------------------------------------------------------------------------------------------------------------
def clean_readout(model, prompt: str, target: str, top_k: int = 5) -> dict:
    """Unedited next-word ranking of `model` for `prompt`: top-k words and the target's rank and probability."""
    ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.cfg.device)
    with torch.no_grad():
        logits = model(ids)[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, top_k)
    out = {
        "top": [(model.tokenizer.decode([int(i)]), float(p)) for p, i in zip(top.values, top.indices)],
        "n_tokens": int(ids.shape[1]),
    }
    if target.strip():
        target = " " + target.strip()   # next words carry a leading space: " France", not "France"
        out["target_str"] = target
        tid = get_target_token_id(model, target)
        out["target_id"] = tid
        out["target_prob"] = float(probs[tid])
        out["target_rank"] = int((probs > probs[tid]).sum().item()) + 1
        out["target_multi_token"] = int(tokens_without_bos(model, target).numel()) > 1
    return out


def _readout_after(logits, tokenizer, tid, top_k=5):
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, top_k)
    return {"prob": float(probs[tid]), "rank": int((probs > probs[tid]).sum().item()) + 1,
            "top": [(tokenizer.decode([int(i)]), float(pv)) for pv, i in zip(top.values, top.indices)]}


# ---------------------------------------------------------------------------------------------------------------------------------------
# The edit itself. This is run_transfer_edit from the tab; see the note at the top of the file for the three small differences.
# `progress(fraction, text)` reports how far along it is; `step_cb(step, total, rank)` reports each gradient step of Small's edit.
# ---------------------------------------------------------------------------------------------------------------------------------------
def run_transfer_edit(small, sae, medium, prompt, target, pair, mechanism, map_kind, dose, steps, top_n, add_cap_factor, progress, step_cb, emit):
    """Make the edit, translate it, inject it into Medium; return before/after for both models."""
    sl, ml = pair
    dev = small.cfg.device
    tgt = " " + target.strip()
    tid = get_target_token_id(small, tgt)
    additive = mechanism.startswith("Gradient descent + additive")
    hook_name = getattr(sae.cfg, "hook_name", f"blocks.{sl}.hook_resid_pre")
    small.reset_hooks()
    medium.reset_hooks()
    clean = build_clean_context(small, sae, prompt, tid)
    if clean.clean_rank == 1:
        return {"note": "The target is already rank 1 in Small; nothing to edit."}
    tokens = small.to_tokens(prompt)
    t0 = time.perf_counter()
    out = {"prompt": prompt, "target": tgt, "pair": list(pair), "mechanism": mechanism, "map": map_kind, "dose": dose}
    ridge = ridge_maps()[f"s2m_L{sl}_L{ml}"]
    W = ridge["W"].to(dev)

    if map_kind == "direct":
        # D6: the dials are tuned by gradient descent through the translator, looking at MEDIUM's output. Small is only used to say which 200 features are active.
        progress(0.0, "Choosing candidate features in Small")
        cands = get_top_active_features(small, sae, prompt, top_n=int(top_n), clean_ctx=clean, positions="all")
        fids = [f for f, _ in cands]
        # tell the page which dials there are (feature ids and how strongly each fires on this prompt) before they start moving
        emit({"type": "dial_setup", "fids": fids, "strength": [round(float(s_), 3) for _, s_ in cands]})
        progress(0.05, "Tuning the dials against Medium (gradient steps through Medium)")
        with DialRecorder(emit) as rec:
            tuned = tune_through_translator(small, medium, sae, W, fids, tokens, tid, ml, dose, steps=int(steps))
        out["dials"] = {"fids": fids, "strength": [round(float(s_), 3) for _, s_ in cands], "frames": rec.frames, "ranks": []}   # Direct has no per-step rank in Small
        fids, a, added = tuned["fids"], tuned["a"], {}
        if additive:
            out["mechanism"] = "Gradient descent (the additive option does not apply to the Direct method)"
        out["small_edit"] = True            # the tuned dials are also a change of Small's state; what they do in Small is shown too
        out["small_note"] = "These dials were tuned against Medium. This is what the same dials do inside Small."
        out["medium_train_rank"] = tuned["train_rank"]
    else:
        # Every other map: the edit is made by looking at SMALL only (stops when the target is rank 1 in Small); Medium just receives the translated change.
        ranks = []   # target's rank in Small after each step (the page plots this next to the dials)

        def _cb(step, total, rank):
            ranks.append(int(rank))
            progress(min(step / total, 1.0) * 0.8, f"Gradient descent in Small: step {step}/{total}, target rank {rank}")
            step_cb(step, total, rank)

        extra = {"additive": True, "add_cap_factor": float(add_cap_factor)} if additive else {}
        # Small's edit does not depend on the map or the dose, so it is kept: changing only the map or the dose does not repeat the gradient steps.
        ckey = (prompt, tgt, sl, additive, int(top_n), int(steps), float(add_cap_factor))
        if ckey in GD_CACHE:
            gd = GD_CACHE[ckey]
            progress(0.8, "Reusing the edit already made in Small for this prompt and settings")
            out["dials"] = TRAJ_CACHE.get(ckey)
        else:
            # Same candidate list the edit function will pick for itself (same call, same arguments), asked for here only so the page can draw the dials
            cands = get_top_active_features(small, sae, prompt, top_n=int(top_n), clean_ctx=clean, positions="all")
            setup = {"fids": [f for f, _ in cands], "strength": [round(float(s_), 3) for _, s_ in cands]}
            emit({"type": "dial_setup", **setup})
            with DialRecorder(emit) as rec:
                gd = run_gradient_descent_edit(small, sae, clean, tid, prompt, sl, hook_name, top_n=int(top_n), positions="all", steps=int(steps), on_step=_cb, **extra)
            GD_CACHE[ckey] = gd
            out["dials"] = TRAJ_CACHE[ckey] = {**setup, "frames": rec.frames, "ranks": ranks}
        fids, a, added = gd["fids"], gd["a"], gd.get("added") or {}
        out["small_edit"] = True
        out["small_gd_rank"] = gd["real_path"]["rank"]

    # the change the recipe makes to Small's state at layer sl, at every position (start-of-text position left alone)
    progress(0.85, "Translating the change and injecting it into Medium")
    with torch.no_grad():
        h = states(small, tokens, sl)[0]
        fid_t = torch.as_tensor(fids, device=dev)
        a_t = torch.as_tensor(a, device=dev, dtype=torch.float32)
        acts = sae.encode(h)[:, fid_t]
        delta = torch.einsum("pk,k,kd->pd", acts, a_t, sae.W_dec[fid_t].detach())
        if added:
            ids = torch.as_tensor([int(k) for k in added.keys()], device=dev)
            amt = torch.as_tensor([float(v) for v in added.values()], device=dev, dtype=torch.float32)
            delta[-1] = delta[-1] + amt @ sae.W_dec[ids].detach()
        delta[0] = 0
        # Small after its own edit
        small_logits = run_injected(small, tokens, delta[None], sl)[0]
        out["small_after"] = _readout_after(small_logits, small.tokenizer, tid)
        # translated change for Medium
        if map_kind == "mlp":
            f = load_mlp_translator(f"s2m_L{sl}_L{ml}", dev)
            tr = f(h + delta) - f(h)
        elif map_kind == "trained":
            Wt = torch.load(TRAINED_MAP_FILE, weights_only=False)["W"].to(dev)
            tr = delta @ Wt
        else:
            tr = delta @ W
        tr = dose * tr
        tr[0] = 0
        med_before = medium(tokens)[0, -1]
        out["medium_before"] = _readout_after(med_before, medium.tokenizer, tid)
        med_logits = run_injected(medium, tokens, tr[None], ml)[0]
        out["medium_after"] = _readout_after(med_logits, medium.tokenizer, tid)
        out["injected_size_frac"] = float(tr[-1].norm().item() / states(medium, tokens, ml)[0, -1].norm().item())
    out["seconds"] = time.perf_counter() - t0
    progress(1.0, "Done")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Web routes
# ---------------------------------------------------------------------------------------------------------------------------------------
def _json(data, status=200):
    return JSONResponse(data, status_code=status)


async def api_status(request):
    """What the page needs on load: are the models in memory, which device, which files exist, facts of both models."""
    info = {"loading": S.loading, "error": S.error, "device": S.device,
            "maps_file": MAPS_FILE, "maps_file_found": os.path.exists(MAPS_FILE)}
    if S.small is not None and S.medium is not None:
        info["small"] = {**model_facts(S.small), "sae": "gpt2-small-res-jb", "sae_features": int(getattr(S.sae.cfg, "d_sae", 24576)),
                         "edit_layer": SMALL_LAYER, "hook": getattr(S.sae.cfg, "hook_name", f"blocks.{SMALL_LAYER}.hook_resid_pre")}
        info["medium"] = {**model_facts(S.medium), "sae": "none", "injection_layer": MEDIUM_LAYER}
    return _json(info)


async def api_config(request):
    """Texts and layer pairs the page shows. They live here, in one place, so the page and the study cannot disagree."""
    pairs = load_translation_pairs()
    return _json({
        "edit_methods": EDIT_METHODS, "map_choices": MAP_CHOICES, "map_help": MAP_HELP,
        "default_pair": [SMALL_LAYER, MEDIUM_LAYER], "sae_layer": S.sae_layer,
        "pairs": [{"small": a, "medium": b, **q, "note": STUDY_PAIRS.get((a, b), "not used in the study"),
                   "maps": available_maps((a, b))} for (a, b), q in sorted(pairs.items())],
    })


async def api_readout(request):
    """Unedited ranking of the prompt in both models (the two boxes shown before any edit)."""
    body = await request.json()
    prompt, target = body.get("prompt", ""), body.get("target", "")
    if S.loading or S.error or not prompt.strip():
        return _json({"error": "models not ready or empty prompt"}, 400)
    with RUN_LOCK:
        return _json({"small": clean_readout(S.small, prompt, target), "medium": clean_readout(S.medium, prompt, target)})


async def api_run(request):
    """
    Run one edit and stream what happens as lines of JSON, one per line:
      {"type": "progress", "frac": 0.4, "text": "..."}        for the progress bar
      {"type": "step", "step": 12, "total": 100, "rank": 7}   one per gradient step in Small (the page draws a live ladder from these)
      {"type": "result", "result": {...}}                    the finished run, or {"note": "..."} if the target was already rank 1
      {"type": "error", "text": "..."}                       if anything failed
    Streaming (and not one answer at the end) because the gradient steps take a while and the page should show them as they happen.
    """
    b = await request.json()
    if S.loading or S.error:
        return _json({"error": "models not ready"}, 400)
    prompt, target = b.get("prompt", ""), b.get("target", "")
    if not prompt.strip() or not target.strip():
        return _json({"error": "enter a prompt and a target"}, 400)
    mechanism = b.get("mechanism", "Gradient descent (default)")
    map_kind = b.get("map", "ridge")
    pair = tuple(b.get("pair", [SMALL_LAYER, MEDIUM_LAYER]))
    if mechanism not in EDIT_METHODS or map_kind not in MAP_CHOICES or map_kind not in available_maps(pair):
        return _json({"error": "that mechanism, map or layer pair is not available"}, 400)
    if pair[0] != S.sae_layer:   # only the layer-8 SAE is held in memory, as in the app (the tab asked for the sidebar layer to match)
        return _json({"error": f"Small layer {pair[0]} needs its own SAE; this server holds the SAE for layer {S.sae_layer}."}, 400)
    if not os.path.exists(MAPS_FILE):
        return _json({"error": f"Translator file missing: {MAPS_FILE}"}, 400)
    # the same ranges the tab's inputs allowed
    dose = min(max(float(b.get("dose", 2.0)), 0.5), 6.0)
    steps = min(max(int(b.get("steps", 100)), 10), 300)
    top_n = min(max(int(b.get("top_n", 200)), 20), 400)
    cap = min(max(float(b.get("add_cap", 1.0)), 0.05), 3.0)

    q = queue.Queue()

    def work():
        # runs in its own thread; pushes events into the queue that the response below reads from
        try:
            with RUN_LOCK:
                before_small = clean_readout(S.small, prompt, target)
                res = run_transfer_edit(S.small, S.sae, S.medium, prompt, target, pair, mechanism, map_kind, dose, steps, top_n, cap,
                                        progress=lambda f, t: q.put({"type": "progress", "frac": min(max(f, 0.0), 1.0), "text": t}),
                                        step_cb=lambda s, n, r: q.put({"type": "step", "step": int(s), "total": int(n), "rank": int(r)}),
                                        emit=q.put)
                if "note" not in res:
                    res["small_before"] = {"rank": before_small["target_rank"], "prob": before_small["target_prob"]}
            q.put({"type": "result", "result": res})
        except Exception as e:
            q.put({"type": "error", "text": f"{type(e).__name__}: {e}"})
        q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            ev = q.get()
            if ev is None:
                return
            yield json.dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


async def index(request):
    return FileResponse(os.path.join(DIST, "index.html"))


# The built page (npm run build in transfer_demo/web). While developing, use "npm run dev" instead, which proxies /api to this server.
DIST = os.path.join(ROOT, "transfer_demo", "web", "dist")
routes = [
    Route("/api/status", api_status),
    Route("/api/config", api_config),
    Route("/api/readout", api_readout, methods=["POST"]),
    Route("/api/run", api_run, methods=["POST"]),
]
if os.path.isdir(DIST):
    routes += [Route("/", index), Mount("/", StaticFiles(directory=DIST), name="web")]


@contextlib.asynccontextmanager
async def lifespan(app):
    # start loading the models as the server starts; the page polls /api/status until they are in memory
    threading.Thread(target=load_everything, daemon=True).start()
    yield


app = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DEMO_PORT", "8765")))
