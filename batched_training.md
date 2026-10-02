# batched_training.md: instructions for an AI agent running part of the data preparation on SYSTEM TWO

You are an AI coding agent (Claude Code or similar) working on a second Windows computer. This file tells you exactly what to do, what to check, when to stop, and what to report. Follow it literally. If anything does not match what is written here, **stop and report; do not improvise or edit the project's code.**

## 1. The project in plain words

FeatureScalpel edits GPT-2 small during a single forward pass, using features from a sparse autoencoder (SAE) at layer 8, to make a suppressed true answer (for example "French" for "The mother tongue of Danielle Darrieux is") become the model's top prediction. The editor mutes some features and boosts others with strengths that are currently fixed (mute 0.6, boost 0.5) for every prompt.

**The study being prepared:** a small neural network that picks the mute and boost strengths per prompt instead of using fixed values. Full plan: `docs/Research_Journal/23.md`. Nothing is being trained yet.

## 2. What is happening right now, and what your job is

The study needs a per-prompt "cache": for each prompt, one run of GPT-2 and the SAE that records the baseline, the 200 candidate features, and what happens when each candidate is switched off. This is slow-ish data preparation (about 0.6 s per prompt on the main machine), not training.

There are **1,450 prompts** (1,000 training + 150 validation + 150 test-seen + 150 test-unseen). Two machines share the work:
- **System one (the user's main machine)** runs `--shard 0/2` (725 prompts).
- **System two (you)** run `--shard 1/2` (the other 725 prompts).

Each prompt produces one file, `outputs/strength_cache/<case_id>.pt`. File names never clash between machines. Your job: produce your 725 files correctly, verify them, package them, and report. The user will merge them.

**Do not run `--shard 0/2`. Do not run anything else heavy at the same time (the GPU has 6 GB).**

## 3. What has been done already (on system one, committed to git)

- Downloaded CounterFact (`datasets/counterfact.json`, 45 MB, 21,919 records) and built the hard set: prompts whose true answer is a single GPT-2 token and is NOT GPT-2's top prediction, with the true answer ranked 2 to 1000. Result: **16,360 prompts** (`data/counterfact_hard_set_summary.json`).
- Split them with `tools/split_counterfact_set.py`: held-out relations (P140, P641, P1303, P176, P190) form the unseen test set; subjects never appear in two splits. Result is `data/counterfact_split.json` (case IDs only).
- Wrote and tested the cache builder `tools/build_strength_cache.py` (`src/strength_cache.py`): 25-prompt test, 0.60 s per prompt, saved candidate rankings identical to the sweep's own functions on 6 of 6 prompts, a built-in sanity check that stops the run if the batched computation disagrees with a one-feature-at-a-time computation.
- A bug was already caught and fixed in an earlier draft (wrong `scale` argument). The current code is correct; do not "fix" it.

## 4. Rules for you

1. Work only on the branch `pool-refill-implementation`. Do not create other branches. Do not commit anything under `outputs/`. Do not modify any file in `src/` or `tools/`.
2. You may download exactly two things: `counterfact.json` from `https://rome.baulab.info/data/dsets/counterfact.json` (only if the user has not copied it to you), and the model and SAE weights that the Python libraries fetch themselves (GPT-2 small from Hugging Face and the `gpt2-small-res-jb` SAE). The user has authorised these for this task. Download nothing else.
3. Do not enter or look for passwords, tokens or keys. If Hugging Face asks for a token and it is not already configured, stop and report.
4. If a check below fails, stop and report. Do not continue "to see what happens".
5. Keep the machine awake while the job runs (disable sleep, plug in power). Windows sleep pauses the job; it can be resumed, but avoid it.

## 5. Step by step

All commands are PowerShell, run from the project root (the folder containing `tools/` and `src/`).

### Step 1: get the code
```powershell
git clone https://github.com/adi5022/Mechanistic-Intervention-of-Large-Language-Models-via-Sparse-AutoEncoders.git FeatureScalpel
cd FeatureScalpel
git checkout pool-refill-implementation
git pull
git log --oneline -1
```
(If the folder already exists, `cd` into it, `git checkout pool-refill-implementation`, `git pull`.) Record the commit hash you are on; the user wants it in your report. It must contain `tools/build_strength_cache.py`, `tools/split_counterfact_set.py` and `data/counterfact_split.json`. If `data/counterfact_split.json` is missing, stop and report.

### Step 2: Python environment
The main machine uses Python 3.13.7, torch 2.6.0+cu124, transformer-lens 3.5.1, sae-lens 6.45.3, transformers 5.13.0, numpy 2.4.4, pandas 3.0.3. Note that `requirements.txt` pins `torch==2.13.0`, which does NOT match the working environment; do not install torch from it.
```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install transformer-lens==3.5.1 sae-lens==6.45.3 transformers==5.13.0 numpy==2.4.4 pandas==3.0.3 scipy altair python-dotenv tabulate
```
If Python 3.13 is not available, use the closest available version and **say so in your report**. Then verify:
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `2.6.0+cu124 True <an NVIDIA RTX 4050 name>`. If CUDA is not available, stop and report (a CPU run would take far too long).

### Step 3: data files
You need `data/counterfact_hard_set.json` (15.5 MB, not in git). Preferred: the user copies it from system one into `data/`. Check its fingerprint:
```powershell
Get-FileHash data\counterfact_hard_set.json -Algorithm SHA256
```
Expected SHA-256 (PowerShell prints it in upper case; compare ignoring case): `657b475e3cc37c7fb7c24a2c5c9d376791c70370cc894dca5ac3db31a3526ec8`.

If you do not have the file, and the user has not provided it, rebuild it (takes about a minute):
1. Put `counterfact.json` in `datasets\` (copy from the user, or download it from the URL in rule 2). Check its SHA-256 equals `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`.
2. Run `.venv\Scripts\python.exe tools/build_counterfact_set.py`.
3. Compare the printed yield with this: usable 21,913; already rank 1: 1,817; above rank 1000: 3,736; **kept 16,360** (15,594 distinct subjects, 34 relations).
4. Compare the SHA-256 of the rebuilt `data\counterfact_hard_set.json` with the expected value above. **If the numbers differ even slightly, or the hash differs, STOP and report:** GPU rounding on a different card can change which prompts land at the rank-1000 border, and both machines must use exactly the same prompt list. The user will then copy the file from system one.

Also check the split file:
```powershell
Get-FileHash data\counterfact_split.json -Algorithm SHA256
```
Expected: `68638fd7193c599accb3803bbc9c21bdcf870856a92caf6902c2bc899f5f10b0`. If it differs, stop and report.

### Step 4: smoke test (about one minute)
```powershell
.venv\Scripts\python.exe tools/build_strength_cache.py --shard 1/2 --limit 5 --out-dir outputs\strength_cache_smoke
```
Required: the line `Sanity check (batched vs one-at-a-time removal, first prompt): OK`, a final line with "Rank mismatches vs the hard-set file: 0", and a per-prompt time (it was 0.6 s on a GTX 1660 Ti; anything from 0.3 to 3 s is plausible). If the sanity check says MISMATCH, stop and report; the script will also stop by itself. Then delete the smoke folder:
```powershell
Remove-Item -Recurse -Force outputs\strength_cache_smoke
```

### Step 5: the real run
Run it in the background with a log, so a closed terminal does not matter:
```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "tools/build_strength_cache.py --shard 1/2" -RedirectStandardOutput outputs\cache_run.log -RedirectStandardError outputs\cache_run.err -NoNewWindow
```
Check progress with `Get-Content outputs\cache_run.log -Tail 5`. Lines look like `[225/725] 1.60 prompts/s | elapsed 2.3 min | ETA 5.2 min`. Expected total: 725 prompts, probably 5 to 25 minutes. `outputs\cache_run.err` normally only has library warnings.

If the job is interrupted (crash, reboot, sleep), run the **same** command again. It skips every prompt whose file exists and continues. Never delete the files to "start fresh".

### Step 6: verify
The run is finished when the log contains `Done: 725 prompts` (or, after a resume, the count of the remaining ones, and a final "Nothing to do" on a re-run). Then:
```powershell
.venv\Scripts\python.exe tools/verify_strength_cache.py --expected 725 --split-file data/counterfact_split.json
```
It loads every file and prints the file count, the number with a problem, the run info (GPU, torch version, seconds, rank mismatches) and finally `RESULT: OK` or `RESULT: NOT OK`. Expected: **725 files, 0 with a problem, RESULT: OK**. (It will also say that about 725 of the 1,450 planned prompts are "missing"; that is correct, they belong to system one.) Re-running the exact command from Step 5 must print "to compute: 0" and "Nothing to do.", which confirms completeness.
The run info line shows `baseline_rank_mismatches_vs_hard_set`. It should be 0; one or two caused by exact probability ties would be acceptable but must be reported. A prompt with no active features at all counts as a problem; report any case IDs.

### Step 7: package
```powershell
Compress-Archive -Path outputs\strength_cache\* -DestinationPath outputs\strength_cache_system2.zip -Force
Get-Item outputs\strength_cache_system2.zip | Select-Object Length
```
Expected size: roughly 30 to 40 MB. Tell the user where the zip is. The user will move it to system one (USB, cloud drive) and unzip it into system one's `outputs\strength_cache\`. Do not commit it and do not upload it anywhere yourself.

## 6. If something goes wrong

| Symptom | What to do |
|---|---|
| CUDA not available | stop, report the driver / `nvidia-smi` output |
| Hugging Face asks for a login | stop, report (do not look for or create a token) |
| Hash of the hard set or the split file differs | stop, report both hashes |
| Sanity check prints MISMATCH | stop, report the full log line |
| Out of memory | stop and report the GPU memory shown by `nvidia-smi`; do not edit `MAX_EVAL_BATCH` |
| Rank mismatches greater than a handful | stop and report the number |
| A file fails to load in Step 6 | list the case IDs, delete only those files, re-run the Step 5 command (it recomputes just those) |
| `git pull` conflicts | stop and report; do not force anything |

## 7. Report back (copy this template and fill it in)

```
SYSTEM TWO REPORT
Commit: <hash>   Branch: pool-refill-implementation
Python: <version>   torch: <version>   GPU: <name>   driver/CUDA: <info>
Hard-set hash matches expected: yes/no (rebuilt locally or copied?)
Split hash matches expected: yes/no
Smoke test: sanity check OK? yes/no, seconds per prompt: <x>
Real run: prompts computed: <n> (expected 725), total seconds: <x>, seconds per prompt: <x>
Rank mismatches vs hard-set file: <n>
Files in outputs\strength_cache: <n> (expected 725), files with problems: <n>
Re-run says "Nothing to do": yes/no
Zip: outputs\strength_cache_system2.zip, size <MB>
Anything unexpected: <free text>
```

## 8. What comes next (not your job)

After both halves are merged on system one: a headroom measurement and then the training code for the strength network (`docs/Research_Journal/23.md`, Section 13). Do not start any of that.

---

## 9. JOB 2 (strength tables) for system two

Same rules as sections 4 and 6 above (do not edit code, do not commit `outputs/`, stop and report on any mismatch). This job needs the **full merged cache** of 1,450 files in `outputs\strength_cache` on system two (the user copies it over; check it with `tools/verify_strength_cache.py --expected 1450 --split-file data/counterfact_split.json`, expected `RESULT: OK`).

1. `git pull` on the branch `pool-refill-implementation`. The file `tools/build_strength_tables.py` must be present.
2. Smoke test (about 1 minute): 
```powershell
.venv\Scripts\python.exe tools/build_strength_tables.py --limit 3 --out-dir outputs\strength_tables_smoke
```
Required: the line `Sanity check: ... OK` (both differences about 1e-9 or smaller). If it says MISMATCH, stop and report. Then delete the smoke folder.
3. Real run, **this machine takes slices 0 to 4 of 7** (it is the faster machine; the user's other machine takes slices 5 and 6):
```powershell
Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "tools/build_strength_tables.py --shard 0,1,2,3,4/7" -RedirectStandardOutput outputs\tables_run.log -RedirectStandardError outputs\tables_run.err -NoNewWindow
```
Expected: 1,036 prompts (5/7 of 1,450). Roughly 30 to 60 minutes on this machine (the user's slower machine needs about 4.6 s per prompt; this machine should be about 2 times faster). Progress lines look like `[200/1036] 0.50 prompts/s | ...`. If interrupted, run the same command again (it skips finished prompts).
4. Verify (the number of files must be 1,036 for this shard):
```powershell
.venv\Scripts\python.exe tools/build_strength_tables.py --verify --expected 1036
```
Expected: `Problems: 0` and `RESULT: OK`; it also prints the average number of candidates passing the strict filter at each strength (report these two lines).
5. Package and report:
```powershell
Compress-Archive -Path outputs\strength_tables\* -DestinationPath outputs\strength_tables_system2.zip -Force
```
Report using the template in section 7, adding: files in `outputs\strength_tables` (expected 1,036), problems (expected 0), total seconds, seconds per prompt, and the zip size. Do not upload or commit the zip.

