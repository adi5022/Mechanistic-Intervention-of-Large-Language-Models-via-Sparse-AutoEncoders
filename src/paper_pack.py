"""
Paper pack: everything needed to cite a study later, in one labeled, self-describing folder (+ zip).

    from src.paper_pack import build_pack
    info = build_pack("outputs/safety_batches/x.json", result, analysis, entry="21")

Contents of docs/Research_Journal/packs/<pack_id>/:
    README.md                what the pack is, its status (PRELIMINARY / FINAL), key numbers, how to regenerate
    MANIFEST.json            machine-readable provenance: git commit, settings, versions, checksums, figure/table index
    FIGURES.md               numbered figures with captions and the data file each one comes from
    journal_entry_draft.md   factual draft of the journal entry (figures + tables embedded; interpretation left to write)
    figures/                 FigNN_*.svg (vector; provenance embedded in each file's <desc>)
    tables/                  CSV + markdown versions of every table
    data/                    the exact spec, a small summary JSON, and (if not huge) the full result JSON
"""
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import zipfile
from datetime import datetime

MIN_STEERABLE_FOR_FINAL = 30       # below this many comparable prompts the pack is marked PRELIMINARY


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def _versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        from importlib import metadata
        for pkg in ("torch", "transformer-lens", "sae-lens", "streamlit", "pandas", "numpy"):
            try:
                out[pkg] = metadata.version(pkg)
            except Exception:
                pass
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _md_table(rows):
    if not rows:
        return "_(no data)_\n"
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if r[k] is None else str(r[k]) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def build_pack(result_path, res, an, root=None, entry="21", label=None, full_json_max_mb=25):
    """Write the pack folder and return {"pack_dir", "zip_bytes", "manifest", "status"}."""
    root = root or os.path.join("docs", "Research_Journal", "packs")
    stem = _slug(label or os.path.splitext(os.path.basename(result_path))[0])
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    pack_id = f"entry{entry}_{stem}_{stamp}"
    pack_dir = os.path.join(root, pack_id)
    for sub in ("figures", "tables", "data"):
        os.makedirs(os.path.join(pack_dir, sub), exist_ok=True)

    meta, spec = res["meta"], res.get("spec") or {}
    arms, modes, ref = an["arms"], an["modes"], an["reference"]
    n_pairs = min((v["n_prompts"] for v in res["arm_comparison"]["per_arm_mode"].values()), default=0)
    status = "FINAL" if (n_pairs >= MIN_STEERABLE_FOR_FINAL and meta.get("errors", 0) == 0) else "PRELIMINARY"
    commit, dirty = _git("rev-parse", "--short", "HEAD"), bool(_git("status", "--porcelain", "--untracked-files=no"))
    res_sha = _sha256(result_path) if os.path.exists(result_path) else None
    res_mb = os.path.getsize(result_path) / 1e6 if os.path.exists(result_path) else 0

    # ------------------------------------------------------------------ figures
    fig_index = []
    names = list(an["figures"].keys())

    def _order(n):
        kind = ("success", "features", "collateral", "time", "difficulty", "scatter", "reasons")
        k = next((i for i, w in enumerate(kind) if w in n), 99)
        return (0 if "_all_" in n else 1 if "_last_" in n else 2, k, n)

    for i, name in enumerate(sorted(names, key=_order), start=1):
        fid = f"Fig{i:02d}"
        meta_f = an.get("figure_meta", {}).get(name, {"title": name, "caption": name, "table": ""})
        clean_name = re.sub(r"^e\d+_", "", name)
        fname = f"{fid}_{clean_name}"
        desc = (f"{meta_f['title']}. Study: {meta.get('name')}. Pack: {pack_id}. Status: {status}. "
                f"Source result: {os.path.basename(result_path)} (sha256 {res_sha[:12] if res_sha else 'n/a'}). Git commit {commit}{'+uncommitted' if dirty else ''}.")
        svg = an["figures"][name]
        svg = svg.replace("</title>", f"</title><desc>{desc}</desc>", 1)
        with open(os.path.join(pack_dir, "figures", fname), "w", encoding="utf-8") as f:
            f.write(svg)
        extra = []
        spec_ = getattr(an["figures"][name], "spec", None)
        if spec_ is not None:
            try:
                from src.paper_figs_mpl import render
                base = os.path.join(pack_dir, "figures", os.path.splitext(fname)[0])
                for written in render(spec_, base, footer=f"{pack_id} | {status} | commit {commit}"):
                    extra.append("figures/" + os.path.basename(written))
            except ImportError:
                pass                                   # matplotlib not installed: SVG only
        fig_index.append({"id": fid, "file": f"figures/{fname}", "other_formats": extra, "title": meta_f["title"],
                          "caption": meta_f["caption"], "data": meta_f.get("table", "")})

    # ------------------------------------------------------------------ tables
    tables = {}
    for mode in modes:
        rows = [{k: v for k, v in r.items() if k != "Mode"} for r in an["arm_rows"] if r["Mode"] == mode]
        tables[f"arms_{mode}"] = (f"Per-arm results, {mode} mode", rows)
        vrows = [{k: v for k, v in r.items() if k != "Mode"} for r in an["vs_rows"] if r["Mode"] == mode]
        tables[f"vs_reference_{mode}"] = (f"Each arm versus the reference '{ref}', {mode} mode", vrows)
    tables["difficulty_bins"] = ("Success rate by starting difficulty (baseline rank bins)", an["bin_rows"])
    rs = an["reasons"]
    tables["strict_filter_reasons"] = ("Why the strict filter rejects candidates (round 0, reference arm)", [
        {"Verdict": "accepted", "Count": rs["ok"]}, {"Verdict": "rejected: harm rule only", "Count": rs["harm"]},
        {"Verdict": "rejected: rank rule only", "Count": rs["rank"]}, {"Verdict": "rejected: both rules", "Count": rs["both"]},
        {"Verdict": "features unusable on both sides", "Count": an["discard"]["unusable_on_both_sides"]},
        {"Verdict": "candidate features in total", "Count": an["discard"]["features_total"]}])
    wide = []
    for c in an["cells"]:
        row = {"Mode": c["mode"], "Prompt": c["prompt"], "Target": c["target"], "Baseline rank": c["baseline_rank"]}
        for a in arms:
            row[f"{a}: final rank"] = c[a]["final_rank"]
            row[f"{a}: reached #1"] = c[a]["success"]
            row[f"{a}: features at best"] = c[a]["features_at_best"]
            row[f"{a}: collateral KL"] = c[a]["kl_nats"]
        wide.append(row)
    tables["per_prompt"] = ("Per-prompt results for every arm", wide)
    table_index = []
    for tname, (ttitle, rows) in tables.items():
        _write_csv(os.path.join(pack_dir, "tables", f"{tname}.csv"), rows)
        with open(os.path.join(pack_dir, "tables", f"{tname}.md"), "w", encoding="utf-8") as f:
            f.write(f"**{ttitle}**\n\n" + _md_table(rows))
        table_index.append({"file": f"tables/{tname}.csv", "title": ttitle})

    # ------------------------------------------------------------------ data
    with open(os.path.join(pack_dir, "data", "spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    slim = {"meta": meta, "spec": spec, "arm_comparison": res.get("arm_comparison"), "per_arm": res.get("per_arm")}
    with open(os.path.join(pack_dir, "data", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=1, default=str)
    full_copied = False
    if os.path.exists(result_path) and res_mb <= full_json_max_mb:
        shutil.copyfile(result_path, os.path.join(pack_dir, "data", "result_full.json"))
        full_copied = True

    # ------------------------------------------------------------------ manifest
    manifest = {
        "pack_id": pack_id, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": status,
        "journal_entry": f"docs/Research_Journal/{entry}.md (draft in journal_entry_draft.md)",
        "study": {"name": meta.get("name"), "layer": meta.get("layer"), "hook": meta.get("hook_name"), "device": meta.get("device"),
                  "runs": len(res["runs"]), "errors": meta.get("errors", 0), "elapsed_s": meta.get("elapsed_s"),
                  "arms": [{"label": a["label"], "settings": a["settings"]} for a in spec.get("arms", [])], "modes": spec.get("modes"),
                  "default_settings": spec.get("settings"), "prompts": len(spec.get("prompts", [])), "comparable_prompts_per_arm_mode": n_pairs},
        "source_result": {"path": result_path, "sha256": res_sha, "size_mb": round(res_mb, 1), "copied_into_pack": full_copied},
        "code": {"git_commit": commit, "uncommitted_changes_to_tracked_files": dirty},
        "environment": _versions(),
        "figures": fig_index, "tables": table_index,
        "regenerate": {"tab": "Batch: Last vs All Tokens -> Results -> Build paper pack",
                       "cli": f"python tools/make_paper_pack.py {result_path} --entry {entry}"},
    }
    with open(os.path.join(pack_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=str)

    # ------------------------------------------------------------------ FIGURES.md
    banner = ("> **PRELIMINARY - do not cite.** Too few comparable prompts (or errors) for publication-grade claims.\n\n" if status != "FINAL" else "")
    lines = [f"# Figures - {pack_id}\n", banner, f"Study: **{meta.get('name')}** - layer {meta.get('layer')} - result file `{os.path.basename(result_path)}` "
             f"(sha256 `{(res_sha or 'n/a')[:16]}...`) - code commit `{commit}`{' + uncommitted changes' if dirty else ''}.\n"]
    for fg in fig_index:
        fmts = ", ".join(f"[{os.path.splitext(x)[1][1:].upper()}]({x})" for x in fg.get("other_formats", []))
        lines.append(f"## {fg['id']}. {fg['title']}\n\n![{fg['id']}]({fg['file']})\n\n**Caption.** {fg['caption']}  \n"
                     f"*Data:* `tables/{fg['data']}`" + (f"  \n*Also as:* {fmts}" if fmts else "") + "\n")
    with open(os.path.join(pack_dir, "FIGURES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ------------------------------------------------------------------ journal draft (facts only)
    arm_desc = "\n".join(f"- **{a['label']}**: `{json.dumps(a['settings'])}`" for a in spec.get("arms", []))
    jl = [f"# Research Journal Entry {entry} (DRAFT): Dynamic safety filtering - {meta.get('name')}\n",
          f"**Status:** {status}  \n**Generated:** {manifest['created']} from `{os.path.basename(result_path)}` (pack `{pack_id}`)  \n"
          f"**Code commit:** `{commit}`{' (+ uncommitted changes)' if dirty else ''}  \n**Model / layer:** GPT-2 small, layer {meta.get('layer')} (`{meta.get('hook_name')}`), {meta.get('device')}\n\n---\n",
          banner,
          "## 1. Question\n\nDoes relaxing the strict safety filter (tolerance or graded per-feature strength) raise the share of prompts whose target token reaches rank #1, "
          "without extra collateral damage? _(edit if the framing changed)_\n",
          f"## 2. Method\n\n- Prompts: {len(spec.get('prompts', []))} in the spec; {n_pairs} comparable (target not already rank #1) per arm and mode.\n- Modes: {', '.join(modes)}.\n"
          f"- Default settings: `{json.dumps(spec.get('settings'))}`\n- Arms compared on identical prompts:\n{arm_desc}\n- Reference arm: `{ref}`. Collateral damage = mean KL(clean || edited) on 20 unrelated prompts.\n"]
    for mode in modes:
        jl.append(f"## 3.{modes.index(mode) + 1} Results, {mode} mode\n")
        jl.append(_md_table([{k: v for k, v in r.items() if k != "Mode"} for r in an["arm_rows"] if r["Mode"] == mode]))
        jl.append(_md_table([{k: v for k, v in r.items() if k != "Mode"} for r in an["vs_rows"] if r["Mode"] == mode]))
        for fg in fig_index:
            if f"_{mode}_" in fg["file"]:
                jl.append(f"![{fg['id']}]({fg['file']})\n\n*{fg['id']}. {fg['caption']}*\n")
    for fg in fig_index:
        if "reasons" in fg["file"]:
            jl.append(f"## 4. Why the strict filter rejects features\n\n![{fg['id']}]({fg['file']})\n\n*{fg['id']}. {fg['caption']}*\n")
    jl.append("## 5. Interpretation\n\n_TO BE WRITTEN after the results are reviewed._ State which arms beat the reference (sign / McNemar p-values above), "
              "the cost in features/steps/time, and the collateral-damage comparison.\n")
    jl.append("## 6. Limitations\n\n- GPT-2 small, one layer, prompts not drawn at random.\n- Success = rank #1 on the next token only.\n- _(add any new caveats)_\n")
    jl.append(f"## 7. Reproduction\n\n```\n{manifest['regenerate']['cli']}\n```\nOr: Batch tab -> select the result -> Build paper pack. Full provenance in `MANIFEST.json`.\n")
    with open(os.path.join(pack_dir, "journal_entry_draft.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(jl))

    # ------------------------------------------------------------------ README
    rd = [f"# Paper pack `{pack_id}`\n", banner,
          f"Everything needed to cite this study in a paper, with provenance.\n\n- **Study:** {meta.get('name')}\n- **Status:** {status} ({n_pairs} comparable prompts per arm and mode; {meta.get('errors', 0)} run errors)\n"
          f"- **Result file:** `{result_path}` sha256 `{res_sha}`{' (copy in data/result_full.json)' if full_copied else ' (too large to copy; keep the original file)'}\n"
          f"- **Code:** git commit `{commit}`{' + uncommitted changes to tracked files - commit before citing' if dirty else ''}\n\n"
          "## Files\n\n- `FIGURES.md` - numbered figures with captions and their data source\n- `figures/` - each figure as SVG (vector, provenance embedded), PNG (300 dpi) and PDF (vector, for LaTeX)\n"
          "- `tables/` - every table as CSV and markdown\n- `data/` - exact spec, small summary JSON, full result JSON (when small enough)\n"
          "- `journal_entry_draft.md` - factual draft of the journal entry (interpretation still to write)\n- `MANIFEST.json` - machine-readable provenance (settings, versions, checksums)\n\n"
          "## Before you cite\n\n1. Status must be **FINAL**.\n2. Commit the code (`git status` clean for tracked files) so the commit hash in the manifest is exact.\n"
          "3. Copy the finished journal entry to `docs/Research_Journal/" + entry + ".md` and keep this folder next to it.\n"]
    with open(os.path.join(pack_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rd))

    # ------------------------------------------------------------------ zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(pack_dir):
            for fn in files:
                full = os.path.join(base, fn)
                z.write(full, os.path.join(pack_id, os.path.relpath(full, pack_dir)))
    return {"pack_dir": pack_dir, "zip_bytes": buf.getvalue(), "manifest": manifest, "status": status, "pack_id": pack_id}
