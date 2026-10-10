"""
Overnight queue (2026-10-11): waits until the D13 training run (tools/transfer/22_varied_training.py) is no longer running, then runs two descriptive diagnostics, each first as a --smoke
rehearsal and only if that succeeds for real:  E5 (23_softer_import.py) and S1 (24_sae_basis_control.py). Their plans and reading rules are in docs/cross_model_transfer/PLAN.md.

    $env:HF_HUB_OFFLINE=1; .venv\\Scripts\\python.exe tools/transfer/queue_overnight.py

Progress lines go to this window and to outputs/transfer/queue_status.txt. To stop the queue before its next step, create an empty file outputs/transfer/STOP_QUEUE.
Each tool writes its own log to outputs/transfer/logs/. Nothing here changes any earlier result.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "transfer")
STATUS = os.path.join(OUT, "queue_status.txt")
STOP = os.path.join(OUT, "STOP_QUEUE")
PY = sys.executable
STEPS = [("E5 smoke", ["tools/transfer/23_softer_import.py", "--smoke"]), ("E5 real", ["tools/transfer/23_softer_import.py"]),
         ("S1 smoke", ["tools/transfer/24_sae_basis_control.py", "--smoke"]), ("S1 real", ["tools/transfer/24_sae_basis_control.py"])]


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf8") as f:
        f.write(line + "\n")


def python_command_lines():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0


def main():
    os.makedirs(OUT, exist_ok=True)
    say("queue started: waiting for 22_varied_training.py to finish")
    quiet = 0
    while quiet < 2:
        if os.path.exists(STOP):
            say("STOP_QUEUE found: stopping before any step")
            return
        running = [l for l in python_command_lines() if "22_varied_training.py" in l or "21_varied_edits.py" in l]
        quiet = quiet + 1 if not running else 0
        time.sleep(60)
    while gpu_used_mib() > 1500:
        say(f"GPU still holds {gpu_used_mib()} MiB; waiting")
        time.sleep(60)
    done = os.path.exists(os.path.join(OUT, "d13_training.json"))
    say(f"the D13 training run has ended ({'result file d13_training.json exists' if done else 'NO result file: it may have failed; the diagnostics do not depend on it'})")
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    skip_real = set()
    for name, cmd in STEPS:
        if os.path.exists(STOP):
            say("STOP_QUEUE found: stopping")
            return
        tool = name.split()[0]
        if name.endswith("real") and tool in skip_real:
            say(f"{name}: skipped because its smoke rehearsal failed")
            continue
        say(f"{name}: starting ({' '.join(cmd)})")
        t0 = time.time()
        r = subprocess.run([PY] + cmd, cwd=ROOT, env=env, stdout=open(os.path.join(OUT, "logs", f"queue_{name.replace(' ', '_')}.txt"), "w", encoding="utf8"), stderr=subprocess.STDOUT)
        say(f"{name}: finished with exit code {r.returncode} after {time.time() - t0:.0f} s")
        if name.endswith("smoke") and r.returncode != 0:
            skip_real.add(tool)
    say("queue finished")


if __name__ == "__main__":
    main()
