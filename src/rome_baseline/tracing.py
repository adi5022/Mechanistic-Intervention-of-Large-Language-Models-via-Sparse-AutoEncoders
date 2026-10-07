import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import rome_env

KINDS = (None, "mlp", "attn")
CMAPS = {None: "Purples", "mlp": "Greens", "attn": "Reds"}
TITLES = {
    None: "Residual stream (one layer restored)",
    "mlp": "MLP output (window restored)",
    "attn": "Attention output (window restored)",
}


def run_trace(prompt, subject, model_name="gpt2", noise_mult=3.0, window=3, samples=10, mt=None):
    rome_env.setup()
    from experiments.causal_trace import (
        ModelAndTokenizer,
        calculate_hidden_flow,
        collect_embedding_std,
    )

    mt = mt or ModelAndTokenizer(model_name)
    noise = noise_mult * collect_embedding_std(mt, [subject])
    results = {
        kind: calculate_hidden_flow(
            mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
        )
        for kind in KINDS
    }
    return results, noise


def summarize(results):
    for kind, r in results.items():
        s = r["scores"].numpy()
        row = r["subject_range"][1] - 1
        layer = int(s[row].argmax())
        print(
            f"{str(kind):>5}: answer={r['answer'].strip()!r} clean p={r['high_score']:.3f} "
            f"corrupted p={r['low_score']:.3f} | last subject token best layer={layer} "
            f"p={s[row, layer]:.3f}"
        )


def plot_trace(results, prompt, path):
    ntoks = len(results[None]["input_tokens"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.45 * ntoks + 2.5))
    for ax, kind in zip(axes, KINDS):
        r = results[kind]
        s = r["scores"].numpy()
        labels = [t.strip() or "_" for t in r["input_tokens"]]
        for i in range(*r["subject_range"]):
            labels[i] += "*"
        im = ax.imshow(s, cmap=CMAPS[kind], vmin=r["low_score"], aspect="auto")
        ax.set_xticks(range(s.shape[1]))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("layer")
        ax.set_title(TITLES[kind], fontsize=10)
        fig.colorbar(im, ax=ax, label=f"p({r['answer'].strip()})")
    r0 = results[None]
    fig.suptitle(f"{prompt}   clean p={r0['high_score']:.3f}  corrupted p={r0['low_score']:.3f}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
