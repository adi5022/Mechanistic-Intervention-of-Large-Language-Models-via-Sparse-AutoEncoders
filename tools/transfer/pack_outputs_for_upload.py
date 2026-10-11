"""
Package outputs/transfer (local only, not in git) into archives that are ready to upload to a cloud service. Nothing is uploaded; nothing in outputs/transfer is changed or deleted.

    .venv\\Scripts\\python.exe tools/transfer/pack_outputs_for_upload.py

Writes outputs/transfer_upload/ (that folder has its own .gitignore, so it never reaches git):
  transfer_essentials.zip   everything except the caches below: trained maps, ridge and neural translators, the edits, per-record numbers, result files, logs
  transfer_caches.zip       rebuildable caches and rehearsal files: d8_items, d10_judge_items, d13_items, b1_cache_*, wikitext2_tokens, *_smoke*, *_quick*
  MANIFEST.json             every file with size and sha256, which archive holds it, the git commit of the code that produced it
  RESTORE.md                how to restore and what each group of files is for
A copy of MANIFEST.json (without the archives' own hashes) is also written next to the data pack so a clean clone can verify a restored folder.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "outputs", "transfer")
DST = os.path.join(ROOT, "outputs", "transfer_upload")
PACK_MANIFEST = os.path.join(ROOT, "docs", "Research_Journal", "packs", "cross_model_transfer", "outputs_transfer_manifest.json")

CACHE_NAMES = ("d8_items.pt", "d10_judge_items.pt", "d13_items.pt", "wikitext2_tokens.pt")
CACHE_PREFIXES = ("b1_cache_",)
CACHE_MARKS = ("_smoke", "_quick")


def which_archive(rel):
    base = os.path.basename(rel)
    if base in CACHE_NAMES or base.startswith(CACHE_PREFIXES) or any(m in base for m in CACHE_MARKS):
        return "transfer_caches.zip"
    return "transfer_essentials.zip"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, ".gitignore"), "w") as f:
        f.write("*\n")
    files = []
    for dirpath, _, names in os.walk(SRC):
        for n in sorted(names):
            full = os.path.join(dirpath, n)
            rel = os.path.relpath(full, SRC).replace("\\", "/")
            files.append((rel, full))
    files.sort()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f"{len(files)} files in {os.path.relpath(SRC, ROOT)}; code commit {commit[:8]}", flush=True)
    manifest = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "code_commit": commit, "source": "outputs/transfer", "files": []}
    for rel, full in files:
        manifest["files"].append({"path": rel, "bytes": os.path.getsize(full), "sha256": sha256(full), "archive": which_archive(rel)})
    print(f"hashed in {time.time() - t0:.0f} s", flush=True)
    for arch in ("transfer_essentials.zip", "transfer_caches.zip"):
        path = os.path.join(DST, arch)
        mine = [m for m in manifest["files"] if m["archive"] == arch]
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for m in mine:
                z.write(os.path.join(SRC, m["path"]), arcname="transfer/" + m["path"])
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            n = len(z.namelist())
        assert bad is None and n == len(mine), (arch, bad, n, len(mine))
        size = os.path.getsize(path)
        manifest[arch] = {"files": n, "bytes": size, "sha256": sha256(path)}
        print(f"{arch}: {n} files, {size / 1e6:.0f} MB, verified ({time.time() - t0:.0f} s)", flush=True)
    # small git-ignored items that exist only on this computer (never .env, which holds API tokens)
    extras = [("Plans/" + n, os.path.join(ROOT, "Plans", n)) for n in sorted(os.listdir(os.path.join(ROOT, "Plans")))] + [("data/counterfact_hard_set.json", os.path.join(ROOT, "data", "counterfact_hard_set.json"))]
    path = os.path.join(DST, "project_extras.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arcname, full in extras:
            z.write(full, arcname=arcname)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist()) == len(extras)
    manifest["project_extras.zip"] = {"files": [{"path": a, "bytes": os.path.getsize(f), "sha256": sha256(f)} for a, f in extras], "bytes": os.path.getsize(path), "sha256": sha256(path)}
    print(f"project_extras.zip: {len(extras)} files, {os.path.getsize(path) / 1e6:.0f} MB, verified", flush=True)
    json.dump(manifest, open(os.path.join(DST, "MANIFEST.json"), "w"), indent=1)
    slim = {k: v for k, v in manifest.items() if k not in ("transfer_essentials.zip", "transfer_caches.zip", "project_extras.zip")}
    json.dump(slim, open(PACK_MANIFEST, "w"), indent=0)
    ess, cac = manifest["transfer_essentials.zip"], manifest["transfer_caches.zip"]
    open(os.path.join(DST, "RESTORE.md"), "w", encoding="utf8").write(f"""# Restoring outputs/transfer (cross-model transfer study)

Made {manifest['made']} from code commit `{commit}` (branch `cross-model-transfer`). Nothing here is needed to read the paper or redraw a figure (those need only `docs/Research_Journal/packs/cross_model_transfer/`, which is in git).
It is needed to re-run, extend or re-check experiments without recomputing about a day of GPU work.

## Restore
1. Clone the repo and check out `cross-model-transfer`.
2. Unzip both archives into `outputs/` of the clone (each holds a `transfer/` folder): `outputs/transfer/...`. `transfer_caches.zip` is optional.
3. Optional check: compare each file's sha256 with `outputs_transfer_manifest.json` in `docs/Research_Journal/packs/cross_model_transfer/` (same list as `MANIFEST.json` here).

## What is in the archives
- `transfer_essentials.zip` ({ess['files']} files, {ess['bytes'] / 1e6:.0f} MB): the translators (`maps_gpt2_to_gpt2-medium.pt` refits in about 2.5 minutes, `mlp_maps.pt` in 6), the edits made in small (`d_edits.pt`, `d7_train_edits.pt`, `d13_edits.pt`, `d6_recipes.pt`; 20 minutes to 1 hour to remake), every trained map (`d7_map`, `d9b_map_*`, `d10_map_*`, `d13_map_*`; 20 minutes to 6 hours), per-sentence results (`d_eval_*`, `d5_*`, `d6_eval`, `d7_eval`; about 50 minutes each) and the small result files and logs.
- `transfer_caches.zip` ({cac['files']} files, {cac['bytes'] / 1e6:.0f} MB): rebuildable caches (`d8_items`, `d10_judge_items`, `d13_items`, `b1_cache_*`, `wikitext2_tokens`; 10 to 40 minutes each) and rehearsal files (`*_smoke*`, `*_quick*`).

- `project_extras.zip`: `Plans/` (four planning notes, git-ignored) and `data/counterfact_hard_set.json` (git-ignored). Unzip into the repo root (not into `outputs/`). `.env` (API tokens) is deliberately not included.

Archive hashes: essentials `{ess['sha256']}`, caches `{cac['sha256']}`.
""")
    print(f"done in {time.time() - t0:.0f} s; wrote {os.path.relpath(DST, ROOT)}: transfer_essentials.zip, transfer_caches.zip, MANIFEST.json, RESTORE.md; manifest copy {os.path.relpath(PACK_MANIFEST, ROOT)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
