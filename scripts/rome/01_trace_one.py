import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rome_baseline.tracing import plot_trace, run_trace, summarize

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
p.add_argument("--subject", default="Eiffel Tower")
p.add_argument("--name", default="eiffel")
p.add_argument("--window", type=int, default=3)
p.add_argument("--noise-mult", type=float, default=3.0)
a = p.parse_args()

results, noise = run_trace(a.prompt, a.subject, window=a.window, noise_mult=a.noise_mult)
print("noise level:", round(noise, 4))
summarize(results)
out = REPO / "docs" / "rome_baseline" / "tracing" / f"{a.name}.png"
plot_trace(results, a.prompt, out)
print("saved", out)
