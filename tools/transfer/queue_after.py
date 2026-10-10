"""
Second overnight queue (2026-10-11): waits until the first queue (tools/transfer/queue_overnight.py) has written "queue finished" to outputs/transfer/queue_status.txt, then runs step D14
(25_regularised_map.py) first as a --smoke rehearsal and, only if that succeeds, for real; before them it runs the one-minute significance test S2 (26_d6_rewordings_test.py). The plans and
readings are in docs/cross_model_transfer/PLAN.md (steps D14 and S2).

    $env:HF_HUB_OFFLINE=1; .venv\\Scripts\\python.exe tools/transfer/queue_after.py

Progress goes to this window and to outputs/transfer/queue_status.txt (same file as the first queue). To stop it before its next step, create an empty file outputs/transfer/STOP_QUEUE.
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
STEPS = [("S2 real", ["tools/transfer/26_d6_rewordings_test.py"]), ("D14 smoke", ["tools/transfer/25_regularised_map.py", "--smoke"]), ("D14 real", ["tools/transfer/25_regularised_map.py"])]


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] (queue B) {msg}"
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf8") as f:
        f.write(line + "\n")


def first_queue_finished():
    try:
        text = open(STATUS, encoding="utf8").read()
    except OSError:
        return False
    return any("queue finished" in l and "(queue B)" not in l for l in text.splitlines()) or any("STOP_QUEUE found" in l and "(queue B)" not in l for l in text.splitlines())


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0


def main():
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    say("queue B started: waiting for the first queue (E5 and S1) to finish")
    while not first_queue_finished():
        if os.path.exists(STOP):
            say("STOP_QUEUE found: stopping before any step")
            return
        time.sleep(60)
    if os.path.exists(STOP):
        say("STOP_QUEUE found: stopping before any step")
        return
    while gpu_used_mib() > 1500:
        say(f"GPU still holds {gpu_used_mib()} MiB; waiting")
        time.sleep(60)
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    smoke_failed = False
    for name, cmd in STEPS:
        if os.path.exists(STOP):
            say("STOP_QUEUE found: stopping")
            return
        if name.endswith("real") and smoke_failed:
            say(f"{name}: skipped because its smoke rehearsal failed")
            continue
        say(f"{name}: starting ({' '.join(cmd)})")
        t0 = time.time()
        r = subprocess.run([PY] + cmd, cwd=ROOT, env=env, stdout=open(os.path.join(OUT, "logs", f"queueB_{name.replace(' ', '_')}.txt"), "w", encoding="utf8"), stderr=subprocess.STDOUT)
        say(f"{name}: finished with exit code {r.returncode} after {time.time() - t0:.0f} s")
        if name.endswith("smoke") and r.returncode != 0:
            smoke_failed = True
    say("queue B finished")


if __name__ == "__main__":
    main()
