"""
Streamlit tab for the cross-model transfer study (new code; imported by experiment_app.py).

Layout: one shared prompt and target at the top, then a split in the middle. The LEFT pane handles GPT-2 Small (the source model, with the
layer-8 SAE); the RIGHT pane handles GPT-2 Medium (the receiving model, no SAE). Each pane shows its model's facts and the clean (unedited)
next-word ranking for the shared prompt. Later steps (edit in small, translate, inject into medium, import) are added to the panes one by one.

The small model is passed in by the app (it already holds the cached copy); medium is loaded on request and cached for the whole server.
"""
import csv
import html
import os
import time

import streamlit as st
import torch

from src.editing import build_clean_context, get_target_token_id, get_top_active_features, tokens_without_bos
from src.gradient_editing import run_gradient_descent_edit
from src.transfer.b_aware_edit import tune_through_translator
from src.transfer.mlp_maps import load_mlp_translator
from src.transfer.models import load_model, model_facts
from src.transfer.stitch import states
from src.transfer.swap import run_injected

SMALL_LAYER = 8      # the SAE layer used throughout the project
MEDIUM_LAYER = 16    # the layer paired with it in the study (same relative depth is not required; 8 -> 16 was the primary pair)


@st.cache_resource(show_spinner=False)
def get_cached_medium(device: str):
    """GPT-2 Medium through TransformerLens, loaded once per server (about 1.4 GB on the GPU)."""
    return load_model("gpt2-medium", device=device)


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


def _facts_table(facts: dict, extra: dict) -> None:
    rows = {"Layers": facts["n_layers"], "Width (d_model)": facts["d_model"], "Parameters (millions)": facts["params_millions"],
            "Vocabulary": facts["d_vocab"], "Context length": facts["n_ctx"], **extra}
    st.table({"Property": list(rows.keys()), "Value": [str(v) for v in rows.values()]})


def _top_table(top, target: str, col_name: str) -> None:
    """Top words as a table; the row of the target token is highlighted."""
    tgt = " " + target.strip() if target.strip() else None
    rows = []
    for w, pr in top:
        hit = tgt is not None and w == tgt
        style = "background:rgba(40,160,120,0.30);font-weight:700;" if hit else ""
        mark = " &#9664; target" if hit else ""
        rows.append(f"<tr style='{style}'><td style='padding:6px 10px;'>{html.escape(repr(w))}{mark}</td><td style='padding:6px 10px;'>{100 * pr:.2f}%</td></tr>")
    st.markdown("<table style='width:100%;border-collapse:collapse;'><thead><tr style='opacity:0.7;text-align:left;'><th style='padding:6px 10px;'>Word</th>"
                f"<th style='padding:6px 10px;'>{col_name}</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>", unsafe_allow_html=True)


def _readout_block(res: dict, target: str) -> None:
    """The result first and large: the target's probability and rank, then the top 5 words."""
    if "target_rank" in res:
        st.markdown(
            f"<div style='padding:14px 16px;border-radius:10px;background:rgba(40,160,120,0.15);border:1px solid rgba(40,160,120,0.5);'>"
            f"<div style='font-size:0.85rem;opacity:0.8;'>Probability of the token {res['target_str']!r}</div>"
            f"<div style='font-size:2.6rem;font-weight:700;line-height:1.1;'>{100 * res['target_prob']:.3f}%</div>"
            f"<div style='font-size:1.1rem;margin-top:4px;'>Rank <b>{res['target_rank']:,}</b></div></div>",
            unsafe_allow_html=True,
        )
        if res["target_multi_token"]:
            st.warning("The target is several tokens long; only its last token is scored (the same rule as the rest of the app).")
    st.markdown("**Top 5 next words**")
    _top_table(res["top"], target, "Probability")



MAPS_SUMMARY = os.path.join("docs", "Research_Journal", "packs", "cross_model_transfer", "maps_summary.csv")
MAPS_FILE = os.path.join("outputs", "transfer", "maps_gpt2_to_gpt2-medium.pt")
STUDY_PAIRS = {(8, 16): "primary pair of the study", (8, 12): "second pair of the study"}

EDIT_METHODS = {
    "Gradient descent (default)": "One multiplier (0 to 3) for each of 200 candidate SAE features of Small, tuned on this prompt by 100 Adam steps "
                                  "(rank term, KL term, size term). The main method of the study.",
    "Gradient descent + additive (silent features)": "As above, and in addition an amount of up to 200 silent features is added at the last position. "
                                                     "Strong enough to push almost any word to rank 1, so treat it as a control and not as a fix.",
}


def load_translation_pairs() -> dict:
    """Fitted Small -> Medium translators (ridge maps on ordinary text): {(small_layer, medium_layer): quality numbers}."""
    pairs = {}
    try:
        with open(MAPS_SUMMARY, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["dir"] == "s2m":
                    pairs[(int(r["src_layer"]), int(r["dst_layer"]))] = {"r2": float(r["perdim_r2"]), "cos": float(r["cosine"])}
    except OSError:
        pass
    return pairs


TRAINED_MAP_FILE = os.path.join("outputs", "transfer", "d7_map.pt")
MLP_MAP_FILE = os.path.join("outputs", "transfer", "mlp_maps.pt")
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


@st.cache_resource(show_spinner=False)
def _ridge_maps():
    import torch as _t
    return _t.load(MAPS_FILE)["maps"]


@st.cache_resource(show_spinner=False)
def _mlp_keys():
    try:
        return set(torch.load(MLP_MAP_FILE, weights_only=False).keys())
    except OSError:
        return set()


def available_maps(pair):
    sl, ml = pair
    av = ["ridge"]
    if f"s2m_L{sl}_L{ml}" in _mlp_keys():
        av.append("mlp")
    if pair == (8, 16) and os.path.exists(TRAINED_MAP_FILE):
        av.append("trained")
    if sl == 8:
        av.append("direct")
    return av


def _sae_layer(sae) -> int:
    hn = getattr(sae.cfg, "hook_name", "")
    try:
        return int(hn.split(".")[1])
    except (IndexError, ValueError):
        return -1


def _readout_after(logits, tokenizer, tid, top_k=5):
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, top_k)
    return {"prob": float(probs[tid]), "rank": int((probs > probs[tid]).sum().item()) + 1,
            "top": [(tokenizer.decode([int(i)]), float(pv)) for pv, i in zip(top.values, top.indices)]}


def run_transfer_edit(small, sae, medium, prompt, target, pair, mechanism, map_kind, dose, steps, top_n, add_cap_factor, progress):
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
    out = {"prompt": prompt, "target": tgt, "pair": pair, "mechanism": mechanism, "map": map_kind, "dose": dose}
    ridge = _ridge_maps()[f"s2m_L{sl}_L{ml}"]
    W = ridge["W"].to(dev)

    if map_kind == "direct":
        progress(0.0, "Choosing candidate features in Small")
        cands = get_top_active_features(small, sae, prompt, top_n=int(top_n), clean_ctx=clean, positions="all")
        fids = [f for f, _ in cands]
        progress(0.05, "Tuning the dials against Medium (gradient steps through Medium)")
        tuned = tune_through_translator(small, medium, sae, W, fids, tokens, tid, ml, dose, steps=int(steps))
        fids, a, added = tuned["fids"], tuned["a"], {}
        if additive:
            out["mechanism"] = "Gradient descent (the additive option does not apply to the Direct method)"
        out["small_edit"] = True            # the tuned dials are also a change of Small's state; what they do in Small is shown too
        out["small_note"] = "These dials were tuned against Medium. This is what the same dials do inside Small."
        out["medium_train_rank"] = tuned["train_rank"]
    else:
        def _cb(step, total, rank):
            progress(min(step / total, 1.0) * 0.8, f"Gradient descent in Small: step {step}/{total}, target rank {rank}")

        extra = {"additive": True, "add_cap_factor": float(add_cap_factor)} if additive else {}
        cache = st.session_state.setdefault("xf_gd_cache", {})
        ckey = (prompt, tgt, sl, additive, int(top_n), int(steps), float(add_cap_factor))
        if ckey in cache:
            gd = cache[ckey]
            progress(0.8, "Reusing the edit already made in Small for this prompt and settings")
        else:
            gd = run_gradient_descent_edit(small, sae, clean, tid, prompt, sl, hook_name, top_n=int(top_n), positions="all", steps=int(steps), on_step=_cb, **extra)
            cache[ckey] = gd
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


def _summary_box(prompt: str, target: str, placeholder, layers, mechanism: str) -> None:
    """The sentence about to be edited and the change we are going to make, filled in after the choices below are read."""
    tgt = " " + target.strip() if target.strip() else ""
    sm = st.session_state.get("xf_small_readout")
    md = st.session_state.get("xf_medium_readout") if st.session_state.get("xf_medium_loaded") else None

    def now(r):
        return f"rank {r['target_rank']:,}, {100 * r['target_prob']:.3f}%" if r and "target_rank" in r else "not loaded"

    with placeholder, st.container(border=True):
        st.markdown("**Sentence to edit**")
        st.markdown(f"<div style='font-size:1.25rem;padding:6px 0;'>{html.escape(prompt.strip()) or '(empty prompt)'}"
                    f"<span style='background:rgba(40,160,120,0.25);border-radius:6px;padding:1px 8px;margin-left:4px;font-weight:600;'>{html.escape(tgt) or '?'}</span></div>",
                    unsafe_allow_html=True)
        st.markdown("**Change**")
        st.markdown(f"Make the token `{tgt}` the **first** next word.")
        c1, c2 = st.columns(2)
        c1.markdown(f"GPT-2 Small now: **{now(sm)}**")
        c2.markdown(f"GPT-2 Medium now: **{now(md)}**")
        st.caption(f"Plan: {mechanism} in Small at layer {layers[0]}; translate the change and add it to Medium at layer {layers[1]}.")


def render_edit_section(prompt: str, target: str) -> None:
    summary_slot = st.container()
    st.markdown("---")
    st.subheader("Edit")

    # ---------------------------------------------------------------- translation layers
    st.markdown("**Translation layers**")
    st.caption("The change an edit makes in Small's residual stream at one layer is translated by a linear map and added to Medium's residual stream at "
               "another layer. Pick the pair. All 25 fitted pairs are offered; 8 to 16 was the primary pair of the study.")
    pairs = load_translation_pairs()
    if not pairs:
        st.error(f"No fitted translators found ({MAPS_SUMMARY}).")
        return
    small_layers = sorted({a for a, _ in pairs})
    medium_layers = sorted({b for _, b in pairs})
    c1, c2 = st.columns(2)
    sl = c1.selectbox("Small layer (where the edit is made)", small_layers, index=small_layers.index(SMALL_LAYER) if SMALL_LAYER in small_layers else 0,
                      key="xf_small_layer")
    ml = c2.selectbox("Medium layer (where the change is added)", medium_layers, index=medium_layers.index(MEDIUM_LAYER) if MEDIUM_LAYER in medium_layers else 0,
                      key="xf_medium_layer")
    q = pairs[(sl, ml)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Translator fit (per-dimension R2)", f"{q['r2']:.2f}")
    m2.metric("Cosine of translated states", f"{q['cos']:.2f}")
    m3.metric("Map file", "found" if os.path.exists(MAPS_FILE) else "missing",
              help=f"{MAPS_FILE}. Not stored in git (about 1 GB outputs folder); restore it from the backup if it is missing.")
    st.caption(STUDY_PAIRS.get((sl, ml), "not used in the study") + f"  |  Small layer {sl} hook: blocks.{sl}.hook_resid_pre,  Medium layer {ml} hook: blocks.{ml}.hook_resid_pre")
    st.session_state["xf_layers"] = (sl, ml)

    # ---------------------------------------------------------------- edit mechanism
    st.markdown("**Edit mechanism (in Small)**")
    choice = st.radio("Edit mechanism", list(EDIT_METHODS.keys()), key="xf_edit_method", label_visibility="collapsed")
    st.caption(EDIT_METHODS[choice])
    st.session_state["xf_edit_choice"] = choice
    _summary_box(prompt, target, summary_slot, (sl, ml), choice)

    # ---------------------------------------------------------------- translation map
    st.markdown("**Translation map**")
    avail = available_maps((sl, ml))
    map_kind = st.radio("Translation map", avail, format_func=lambda k: MAP_CHOICES[k], key="xf_map_kind", label_visibility="collapsed")
    st.caption(MAP_HELP[map_kind] + ("" if len(avail) == 4 else "  Maps offered for this layer pair: " + ", ".join(MAP_CHOICES[k].split(" (")[0] for k in avail) + "."))
    if map_kind == "direct":
        st.caption("With the direct method the edit is made against Medium, so the edit mechanism above only decides whether silent features are added in Small's view; "
                   "the dials are tuned on Medium's output.")

    d1, d2 = st.columns(2)
    dose = d1.slider("Dose (strength of the translated change added to Medium)", 0.5, 6.0, 2.0, 0.5, key="xf_dose",
                     help="2.0 was chosen on development records for the 8 to 16 pair. Other pairs were not tuned.")
    with d2.expander("Advanced settings", expanded=False):
        steps = st.number_input("Gradient steps", 10, 300, 100, 10, key="xf_steps")
        top_n = st.number_input("Candidate features (top N)", 20, 400, 200, 20, key="xf_topn")
        add_cap = st.number_input("Additive cap factor (additive only)", 0.05, 3.0, 1.0, 0.05, key="xf_capf")

    ready, why = True, ""
    if not prompt.strip() or not target.strip():
        ready, why = False, "Enter a prompt and a target."
    elif sl != _SAE_LAYER_NOW[0]:
        ready, why = False, f"Small layer {sl} needs its SAE; the app has the SAE for layer {_SAE_LAYER_NOW[0]}. Set 'Select Model Layer' in the sidebar to {sl}."
    elif not st.session_state.get("xf_medium_loaded"):
        ready, why = False, "Load GPT-2 Medium first (right pane)."
    elif not os.path.exists(MAPS_FILE):
        ready, why = False, f"Translator file missing: {MAPS_FILE}"
    if why:
        st.warning(why)
    if st.button("Run edit and inject into Medium", type="primary", disabled=not ready, key="xf_run"):
        bar = st.progress(0.0, text="Starting")
        res = run_transfer_edit(_SMALL[0], _SAE[0], get_cached_medium(_DEVICE[0]), prompt, target, (sl, ml), choice, map_kind, float(dose), steps, top_n, add_cap,
                                lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
        bar.empty()
        if "note" in res:
            st.session_state["xf_note"] = res["note"]
        else:
            st.session_state["xf_note"] = None
            st.session_state.setdefault("xf_runs", []).append(res)
        st.rerun()

    if st.session_state.get("xf_note"):
        st.info(st.session_state["xf_note"])
    _runs_block(prompt, target)


def _before_after(title: str, before: dict, after: dict, target: str) -> None:
    """One model's result: probability and rank before and after as two boxes, then the top 5 words after the edit."""
    st.markdown(f"**{title}**")
    c1, c2 = st.columns(2)
    for col, label, prob, rank, tint in ((c1, "Before", before["prob"], before["rank"], "40,160,120"), (c2, "After", after["prob"], after["rank"], "80,120,220")):
        col.markdown(
            f"<div style='padding:12px 14px;border-radius:10px;background:rgba({tint},0.15);border:1px solid rgba({tint},0.55);'>"
            f"<div style='font-size:0.85rem;opacity:0.8;'>{label}</div>"
            f"<div style='font-size:2.2rem;font-weight:700;line-height:1.1;'>{100 * prob:.3f}%</div>"
            f"<div style='font-size:1.05rem;margin-top:2px;'>Rank <b>{rank:,}</b></div></div>", unsafe_allow_html=True)
    moved = before["rank"] - after["rank"]
    st.caption(f"Rank {before['rank']:,} to {after['rank']:,} ({'+' if moved >= 0 else ''}{moved:,} places); probability {100 * before['prob']:.3f}% to {100 * after['prob']:.3f}%.")
    _top_table(after["top"], target, "Probability after the edit")


def _run_name(res: dict) -> str:
    """Short name that tells runs apart: the edit mechanism and the translation map."""
    mech = "GD + additive" if res["mechanism"].startswith("Gradient descent + additive") else "GD"
    return f"{mech} | {MAP_CHOICES[res['map']].split(' (')[0]}"


def _diff_text(model: str, now: dict, other: dict, other_no: int) -> str:
    dr = now["rank"] - other["rank"]
    dp = 100 * (now["prob"] - other["prob"])
    unit = "place" if abs(dr) == 1 else "places"
    rank_txt = "same rank" if dr == 0 else (f"{abs(dr):,} {unit} better" if dr < 0 else f"{dr:,} {unit} worse")
    return f"{model}: {rank_txt}, probability {'+' if dp >= 0 else ''}{dp:.3f} points (rank {other['rank']:,}, {100 * other['prob']:.3f}% in run {other_no})"


def _runs_block(prompt: str, target: str) -> None:
    """Every run for the current prompt and target, one result block each, going down the page. Later runs say how they differ from the earlier ones."""
    runs = [r for r in st.session_state.get("xf_runs", []) if r["prompt"] == prompt and r["target"] == " " + target.strip()]
    st.markdown("---")
    h1, h2 = st.columns([4, 1])
    h1.subheader("Result: before and after")
    if runs and h2.button("Clear runs", key="xf_clear"):
        st.session_state["xf_runs"] = []
        st.rerun()
    if not runs:
        st.caption("No runs yet for this prompt and target. Press Run, then change the map or the dose and run again; each run adds a block below.")
        return
    sb = st.session_state.get("xf_small_readout") or {}
    small_before = {"rank": sb["target_rank"], "prob": sb["target_prob"]} if "target_rank" in sb else None
    collapse_old = st.checkbox("Collapse older runs (keep only the newest open)", value=False, key="xf_collapse_old")
    for i, res in enumerate(runs):
        is_last = i == len(runs) - 1
        box = st.expander(f"Run {i + 1}: {_run_name(res)} | dose {res['dose']}", expanded=(is_last or not collapse_old))
        with box:
            _render_run(i, res, runs, small_before, target)


def _render_run(i: int, res: dict, runs: list, small_before, target: str) -> None:
    if True:
        additive = res["mechanism"].startswith("Gradient descent + additive")
        st.markdown(f"<span style='background:rgba({'230,140,40' if additive else '80,120,220'},0.25);border-radius:6px;padding:3px 10px;font-weight:600;'>"
                    f"Edit: {res['mechanism']}</span>", unsafe_allow_html=True)
        st.caption(f"Mechanism: {res['mechanism']}  |  map: {MAP_CHOICES[res['map']]}  |  layers {res['pair'][0]} to {res['pair'][1]}  |  dose {res['dose']}  |  "
                   f"{res['seconds']:.1f} s  |  injected change {100 * res['injected_size_frac']:.1f}% of Medium's state norm at the last position")
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                if res["small_edit"] and small_before:
                    _before_after("GPT-2 Small (edited)", small_before, res["small_after"], target)
                    if res.get("small_note"):
                        st.caption(res["small_note"])
                else:
                    st.markdown("**GPT-2 Small**")
                    st.caption("Small's clean numbers are not available for this prompt.")
        with right:
            with st.container(border=True):
                mb = res["medium_before"]
                _before_after("GPT-2 Medium (receives the translated change)", {"prob": mb["prob"], "rank": mb["rank"]}, res["medium_after"], target)
        if i > 0:
            lines = []
            for j in range(i):
                prev = runs[j]
                parts = [_diff_text("Medium", res["medium_after"], prev["medium_after"], j + 1)]
                if res["small_edit"] and prev["small_edit"]:
                    parts.append(_diff_text("Small", res["small_after"], prev["small_after"], j + 1))
                lines.append(f"**Compared with run {j + 1}** ({_run_name(prev)}): " + "; ".join(parts))
            st.info("  \n".join(lines))


_SMALL, _SAE, _DEVICE, _SAE_LAYER_NOW = [None], [None], [None], [None]


def render(small_model, sae, device: str, layer: int) -> None:
    _SMALL[0], _SAE[0], _DEVICE[0], _SAE_LAYER_NOW[0] = small_model, sae, device, int(layer)
    st.header("Cross-model transfer: GPT-2 Small and GPT-2 Medium")
    st.markdown(
        "Left: **GPT-2 Small**, which has the SAE and is where an edit is made. Right: **GPT-2 Medium**, which has no SAE and is the model that "
        "receives the translated change. Both read the same prompt. This tab will be built step by step; for now each side shows its model and "
        "its clean (unedited) ranking."
    )

    c1, c2 = st.columns(2)
    prompt = c1.text_input("Prompt", "The mother tongue of Danielle Darrieux is", key="xf_prompt")
    target = c2.text_input("Target completion", " English", key="xf_target",
                           help="One word; the leading space is added for you. The same prompt and target are read by both models.")

    st.markdown("---")
    left, right = st.columns(2, gap="large")

    # ------------------------------------------------------------------ LEFT: GPT-2 Small
    with left:
        with st.container(border=True):
            st.subheader("GPT-2 Small  |  source model")
            st.caption("Has the SAE. The edit (one multiplier per SAE feature) is made here.")
            if prompt.strip():
                st.session_state["xf_small_readout"] = clean_readout(small_model, prompt, target)
                _readout_block(st.session_state["xf_small_readout"], target)
            hook = getattr(sae.cfg, "hook_name", f"blocks.{SMALL_LAYER}.hook_resid_pre")
            with st.expander("Model details", expanded=False):
                _facts_table(model_facts(small_model), {"SAE": "gpt2-small-res-jb", "SAE features": int(getattr(sae.cfg, "d_sae", 24576)),
                                                        "Edit layer / hook": f"{SMALL_LAYER} / {hook}", "Device": device})

    # ------------------------------------------------------------------ RIGHT: GPT-2 Medium
    with right:
        with st.container(border=True):
            st.subheader("GPT-2 Medium  |  receiving model")
            st.caption("No SAE. It receives the change an edit makes in Small, translated by a linear map into its own coordinates.")
            with st.spinner("Loading GPT-2 Medium (first run only)..."):
                medium = get_cached_medium(device)
            st.session_state["xf_medium_loaded"] = True
            if prompt.strip():
                st.session_state["xf_medium_readout"] = clean_readout(medium, prompt, target)
                _readout_block(st.session_state["xf_medium_readout"], target)
            with st.expander("Model details", expanded=False):
                _facts_table(model_facts(medium), {"SAE": "none", "Injection layer (paired with Small layer 8)": MEDIUM_LAYER, "Device": device})

    render_edit_section(prompt, target)
