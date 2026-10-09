"""
Step B1 of docs/cross_model_transfer/PLAN.md: the country-swap positive control (Gate 2).

    .venv\\Scripts\\python.exe tools/transfer/03_swap_control.py --quick    # two primary pairs, fewer cases and doses (about 1 minute)
    .venv\\Scripts\\python.exe tools/transfer/03_swap_control.py            # all 8 layer pairs from A2 (a few minutes)

A test case is an ordered pair of countries (a, b) that the RECEIVER answers correctly on both ("The capital of France is the city of" -> Paris).
In the SOURCE model we take  (state for b) - (state for a)  at every position, translate it with the A1 map, multiply by a dose alpha and add it
to the receiver's internal state while the receiver reads the sentence for a. Success = the receiver now answers with b's capital.

Arms:  translated  the real thing
       native      the receiver's OWN real difference (a ceiling; at alpha 1 on all positions it is an exact state replacement)
       random      a random vector with the same size at each position
       randmap     the source difference pushed through a random linear map, same size at each position
       wrong       the translated difference of a DIFFERENT country pair
Variants: all positions / only the word's position / only the last position.
alpha is chosen on 30% of the country pairs (dev) and reported on the other 70% (test). Controls are also shown at their own best alpha on test,
which favours the controls. Gate 2 (fixed in the plan before this was run) is evaluated for the two primary pairs, all-positions variant.

Writes outputs/transfer/b1_swap.json (b1_swap_quick.json in quick mode).
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from src.transfer.models import load_model, pick_device
from src.transfer.stitch import states
from src.transfer.swap import NEUTRAL, TEMPLATES, case_metrics, last_logits, run_injected, single_token_pairs, token_id, wilson

OUT_DIR = os.path.join(ROOT, "outputs", "transfer")
SMALL, MEDIUM = "gpt2", "gpt2-medium"
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ARMS = ["translated", "native", "random", "randmap", "wrong"]
CONTROLS = ["random", "randmap", "wrong"]
VARIANTS = ["all", "word", "last"]
PRIMARY = {"s2m_L8_L16": "Export (small edit -> medium)", "m2s_L16_L8": "Import (medium state -> small)"}
DEFAULT_PAIRS = "s2m_L8_L16,m2s_L16_L8,s2m_L8_L12,s2m_L6_L8,s2m_L2_L4,m2s_L12_L8,m2s_L8_L6,m2s_L4_L2"
GATE_RATE, GATE_MARGIN, GATE_P = 0.20, 0.15, 0.01


def stage(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_dev(a, b):
    return zlib.crc32(f"{a}->{b}".encode()) % 10 < 3                  # about 30% of the country pairs, fixed forever


def norm_to(X, ref):
    """Rescale X so that, at every position, it has the same length as ref (positions where ref is 0 stay 0)."""
    return X / X.norm(dim=-1, keepdim=True).clamp(min=1e-9) * ref.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------------------------------------------------------------------
class Context:
    """Everything that depends only on the models and the prompts: countries, tokens, the receiver's own answers."""

    def __init__(self, small, medium, dev):
        self.dev = dev
        self.models = {"small": small, "medium": medium}
        tok = small.tokenizer
        self.pairs = single_token_pairs(tok)
        self.countries = list(self.pairs)
        self.cap_id = torch.tensor([token_id(tok, self.pairs[c]) for c in self.countries], device=dev)
        self.tokens, self.word_pos = [], []
        for T in TEMPLATES:
            t = small.to_tokens([T.replace("{X}", c) for c in self.countries]).to(dev)
            cols = [c for c in range(t.shape[1]) if len(set(t[:, c].tolist())) > 1]
            assert len(cols) == 1, "countries must differ at exactly one position"
            self.tokens.append(t); self.word_pos.append(cols[0])
        self.logits, self.known = {}, {}
        for name, m in self.models.items():
            self.logits[name] = [last_logits(m, t) for t in self.tokens]
            self.known[name] = [[i for i in range(len(self.countries)) if lg[i].argmax().item() == self.cap_id[i].item()] for lg in self.logits[name]]


# ----------------------------------------------------------------------------------------------------------------------------------
def evaluate_pair(ctx, key, maps, alphas, max_cases, seed=0):
    d, p, q = key.split("_")[0], int(key.split("_L")[1]), int(key.split("_L")[2])
    S_name, R_name = ("small", "medium") if d == "s2m" else ("medium", "small")
    S, R = ctx.models[S_name], ctx.models[R_name]
    dev = ctx.dev
    W = maps[key]["W"].to(dev)
    dS_dim, dR_dim = W.shape
    g = torch.Generator().manual_seed(seed)
    Wr = (torch.randn(dS_dim, dR_dim, generator=g) / math.sqrt(dS_dim)).to(dev)

    res = {}            # (arm, variant, alpha) -> {metric: tensor over all cases}
    meta = {"a": [], "b": [], "template": [], "dev": [], "wrong_valid": []}
    clean_ld, clean_hit, clean_rank = [], [], []
    last_vecs = []      # translated last-position vectors, for the leak measurement
    sanity = {}

    for ti in range(len(TEMPLATES)):
        known = ctx.known[R_name][ti]
        cases = [(i, j) for i in known for j in known if i != j]
        if max_cases and len(cases) > max_cases:
            g2 = torch.Generator().manual_seed(seed + ti)
            keep = torch.randperm(len(cases), generator=g2)[:max_cases].tolist()
            cases = [cases[k] for k in keep]
        if not cases:
            continue
        A = torch.tensor([c[0] for c in cases], device=dev)
        B = torch.tensor([c[1] for c in cases], device=dev)
        tok = ctx.tokens[ti]
        base = tok[A]
        L = tok.shape[1]
        HS, HR = states(S, tok, p), states(R, tok, q)
        dS = HS[B] - HS[A]
        Dt = dS @ W
        arms = {"translated": Dt, "native": HR[B] - HR[A]}
        rg = torch.Generator().manual_seed(seed + 100 + ti)
        arms["random"] = norm_to(torch.randn(Dt.shape, generator=rg).to(dev), Dt)
        arms["randmap"] = norm_to(dS @ Wr, Dt)
        # wrong pair: the translated difference of another case, chosen so that it heads for a different capital
        best = None
        for _ in range(20):
            perm = torch.randperm(len(cases), generator=rg).to(dev)
            valid = B[perm] != B
            if best is None or valid.sum() > best[1].sum():
                best = (perm, valid)
            if valid.all():
                break
        perm, valid = best
        arms["wrong"] = Dt[perm]
        cap_a, cap_b = ctx.cap_id[A], ctx.cap_id[B]
        cap_other = ctx.cap_id[B[perm]]
        p_native_b = F.softmax(ctx.logits[R_name][ti][B], dim=-1)
        clean = ctx.logits[R_name][ti][A]
        cm = case_metrics(clean, cap_a, cap_b, p_native_b)
        clean_ld.append(cm["ld"]); clean_hit.append(cm["hit"]); clean_rank.append(cm["rank_b"])
        meta["a"] += [ctx.countries[c[0]] for c in cases]; meta["b"] += [ctx.countries[c[1]] for c in cases]
        meta["template"] += [ti] * len(cases); meta["dev"] += [is_dev(ctx.countries[c[0]], ctx.countries[c[1]]) for c in cases]
        meta["wrong_valid"] += valid.cpu().tolist()
        last_vecs.append(Dt[:, -1].cpu())

        positions = {"all": None, "word": ctx.word_pos[ti], "last": L - 1}
        for variant in VARIANTS:
            mask = torch.zeros(1, L, 1, device=dev)
            if positions[variant] is None:
                mask[:] = 1
            else:
                mask[0, positions[variant]] = 1
            for arm in ARMS:
                for alpha in alphas:
                    delta = alpha * arms[arm] * mask
                    lg = run_injected(R, base, delta, q)
                    m = case_metrics(lg, cap_a, cap_b, p_native_b, cap_other if arm == "wrong" else None, ctx.cap_id)
                    slot = res.setdefault((arm, variant, alpha), {})
                    for k, v in m.items():
                        slot.setdefault(k, []).append(v)
        # sanity: zero dose reproduces the unchanged receiver
        z = run_injected(R, base[:16], torch.zeros(min(16, len(base)), L, dR_dim, device=dev), q)
        sanity["zero_dose_max_logit_difference"] = max(sanity.get("zero_dose_max_logit_difference", 0.0), (z - clean[:16]).abs().max().item())
        del HS, HR, dS, arms
    res = {k: {m: torch.cat(v) for m, v in d_.items()} for k, d_ in res.items()}
    meta = {k: (torch.tensor(v) if k in ("dev", "wrong_valid", "template") else v) for k, v in meta.items()}
    return {"res": res, "meta": meta, "clean_ld": torch.cat(clean_ld), "clean_hit": torch.cat(clean_hit), "clean_rank": torch.cat(clean_rank), "last_vecs": torch.cat(last_vecs),
            "sanity": sanity, "S": S_name, "R": R_name, "layer_R": q}


# ----------------------------------------------------------------------------------------------------------------------------------
def rates(out, arm, variant, alpha, subset):
    """Hit rate, mean logit-difference gain and mean KL over the cases in `subset` (bool tensor)."""
    r = out["res"][(arm, variant, alpha)]
    sel = subset.clone()
    if arm == "wrong":
        sel &= out["meta"]["wrong_valid"]
    n = int(sel.sum())
    if n == 0:
        nan = float("nan")
        return {"n": 0, "hit": nan, "ld_gain": nan, "kl": nan, "rank_median": nan, "top_original": nan, "top_other_capital": nan, "top_non_capital": nan}
    return {"n": n, "hit": r["hit"][sel].mean().item(), "ld_gain": (r["ld"][sel] - out["clean_ld"][sel]).mean().item(), "kl": r["kl"][sel].mean().item(),
            "rank_median": r["rank_b"][sel].median().item(), "top_original": r["top_original"][sel].mean().item(),
            "top_other_capital": r["top_other_capital"][sel].mean().item(), "top_non_capital": r["top_non_capital"][sel].mean().item()}


def summarise(out, alphas):
    devm, testm = out["meta"]["dev"], ~out["meta"]["dev"]
    table = {}
    for arm in ARMS:
        for variant in VARIANTS:
            for a_ in alphas:
                table[(arm, variant, a_)] = {"dev": rates(out, arm, variant, a_, devm), "test": rates(out, arm, variant, a_, testm)}
    # alpha for the translated arm, chosen on dev only (ties: larger logit gain, then smaller alpha)
    alpha_dev = {}
    for variant in VARIANTS:
        alpha_dev[variant] = max(alphas, key=lambda a_: (round(table[("translated", variant, a_)]["dev"]["hit"], 6),
                                                         table[("translated", variant, a_)]["dev"]["ld_gain"], -a_))
    best_test = {}
    for arm in ARMS:
        for variant in VARIANTS:
            best_test[(arm, variant)] = max(alphas, key=lambda a_: (round(table[(arm, variant, a_)]["test"]["hit"], 6),
                                                                    table[(arm, variant, a_)]["test"]["ld_gain"], -a_))
    return table, alpha_dev, best_test


def wilcoxon_greater(x, y):
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(x.numpy(), y.numpy(), alternative="greater").pvalue)
    except ValueError:
        return 1.0


def gate(out, table, alpha_dev, best_test, alphas):
    variant = "all"
    a_t = alpha_dev[variant]
    tr = table[("translated", variant, a_t)]["test"]["hit"]
    ctrl = {c: table[(c, variant, best_test[(c, variant)])]["test"]["hit"] for c in CONTROLS}
    strongest = max(CONTROLS, key=lambda c: (round(ctrl[c], 6), table[(c, variant, best_test[(c, variant)])]["test"]["ld_gain"]))
    testm = ~out["meta"]["dev"]
    sel = testm.clone()
    if strongest == "wrong":
        sel &= out["meta"]["wrong_valid"]
    x = out["res"][("translated", variant, a_t)]["ld"][sel] - out["clean_ld"][sel]
    y = out["res"][(strongest, variant, best_test[(strongest, variant)])]["ld"][sel] - out["clean_ld"][sel]
    p = wilcoxon_greater(x, y)
    native = table[("native", "all", 1.0)]["test"]["hit"] if 1.0 in alphas else float("nan")
    sanity_ok = out["sanity"]["zero_dose_max_logit_difference"] < 1e-3 and (native >= 0.97 if not math.isnan(native) else True)
    a_ok, b_ok, c_ok = tr >= GATE_RATE, (tr - ctrl[strongest]) >= GATE_MARGIN, p < GATE_P
    return {"alpha": a_t, "translated_hit": tr, "controls_best_alpha_hit": ctrl, "strongest_control": strongest, "wilcoxon_p": p,
            "native_alpha1_hit": native, "zero_dose_max_logit_difference": out["sanity"]["zero_dose_max_logit_difference"],
            "sanity_ok": bool(sanity_ok), "a_rate": bool(a_ok), "b_margin": bool(b_ok), "c_paired": bool(c_ok),
            "passes": bool(sanity_ok and a_ok and b_ok and c_ok)}


# ----------------------------------------------------------------------------------------------------------------------------------
def leak(ctx, out, R, alpha, n_vec=100, seed=0):
    """Add the translated last-position difference (and a random vector of the same size) to the last position of 12 unrelated prompts."""
    dev = ctx.dev
    g = torch.Generator().manual_seed(seed)
    testm = ~out["meta"]["dev"]
    V = out["last_vecs"][testm]
    V = V[torch.randperm(len(V), generator=g)[:n_vec]].to(dev)
    Rn = norm_to(torch.randn(V.shape, generator=g).to(dev), V)
    name = f"blocks.{out['layer_R']}.hook_resid_pre"
    res = {"translated": [], "random": []}
    flips = {"translated": [], "random": []}
    for text in NEUTRAL:
        t = R.to_tokens(text).to(dev)
        with torch.no_grad():
            clean = R(t)[0, -1].float()
        lp = F.log_softmax(clean, -1)
        for nm, vecs in (("translated", V), ("random", Rn)):
            tt = t.expand(len(vecs), -1)

            def hook(resid, hook, vecs=vecs):
                r = resid.clone()
                r[:, -1] += alpha * vecs
                return r
            with torch.no_grad():
                lg = R.run_with_hooks(tt, fwd_hooks=[(name, hook)])[:, -1].float()
            lq = F.log_softmax(lg, -1)
            res[nm].append((lp.exp() * (lp - lq)).sum(-1).mean().item())
            flips[nm].append((lg.argmax(-1) != clean.argmax()).float().mean().item())
    return {nm: {"mean_kl": sum(v) / len(v), "top1_flip_rate": sum(flips[nm]) / len(flips[nm])} for nm, v in res.items()} | {"alpha": alpha, "n_vectors": len(V)}


# ----------------------------------------------------------------------------------------------------------------------------------
def pct(x):
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--max-cases", type=int, default=0, help="cap on cases per template (0 = all)")
    ap.add_argument("--variants", default="all,word,last", help="injection variants to run (all must be included for the gate)")
    ap.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS), help="doses to try; 1 must be included for the sanity check")
    ap.add_argument("--tag", default="", help="added to the result file name, so separate runs do not overwrite each other")
    ap.add_argument("--fresh", action="store_true", help="ignore the per-pair cache and recompute")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    global VARIANTS
    VARIANTS = [v for v in a.variants.split(",") if v]
    alphas = [float(x) for x in a.alphas.split(",") if x]
    suffix = "_quick" if a.quick else ""
    if a.quick:
        a.pairs, a.max_cases, alphas = "s2m_L8_L16,m2s_L16_L8", 40, [1.0, 2.0, 4.0]
    t0 = time.time()
    dev = pick_device(a.device)
    maps_path = os.path.join(OUT_DIR, f"maps_gpt2_to_gpt2-medium{suffix}.pt")
    if not os.path.exists(maps_path):
        print(f"missing {maps_path}: run step A1 first"); return 1
    maps = torch.load(maps_path)["maps"]

    stage("loading models, building prompts, recording what each model knows")
    small, medium = load_model(SMALL, dev), load_model(MEDIUM, dev)
    ctx = Context(small, medium, dev)
    print(f"   {len(ctx.countries)} countries usable; capitals the receiver answers correctly (per template): "
          f"small {[len(k) for k in ctx.known['small']]}, medium {[len(k) for k in ctx.known['medium']]}")

    summary, gates, leaks, outs = {}, {}, {}, {}
    for key in a.pairs.split(","):
        if key not in maps:
            print(f"   no map {key}, skipping"); continue
        stage(f"pair {key}" + (f"  [{PRIMARY[key]}]" if key in PRIMARY else "  [exploratory]"))
        cache_path = os.path.join(OUT_DIR, f"b1_cache_{key}{suffix}.pt")
        sig = {"alphas": alphas, "variants": VARIANTS, "max_cases": a.max_cases}
        blob = None
        if os.path.exists(cache_path) and not a.fresh:
            b_ = torch.load(cache_path, weights_only=False)
            if b_["sig"] == sig:
                blob = b_
        if blob is None:
            out = evaluate_pair(ctx, key, maps, alphas, a.max_cases)
            table, alpha_dev, best_test = summarise(out, alphas)
            lk = leak(ctx, out, ctx.models[out["R"]], alpha_dev["all"]) if key in PRIMARY else None
            torch.save({"sig": sig, "out": out, "leak": lk}, cache_path)          # a finished pair is never lost again
        else:
            out, lk = blob["out"], blob["leak"]
            table, alpha_dev, best_test = summarise(out, alphas)
            print("   (loaded from the saved per-pair result)")
        outs[key] = (out, table, alpha_dev, best_test)
        n = len(out["clean_ld"]); n_dev = int(out["meta"]["dev"].sum())
        print(f"   {n} cases ({n_dev} dev, {n - n_dev} test) from {len(set(out['meta']['a']) | set(out['meta']['b']))} countries; receiver = {out['R']}, layer {out['layer_R']}; "
              f"alpha chosen on dev: " + ", ".join(f"{v} {alpha_dev[v]}" for v in VARIANTS))
        g_ = gate(out, table, alpha_dev, best_test, alphas)
        gates[key] = g_
        if lk is not None:
            leaks[key] = lk
        summary[key] = {f"{arm}|{v}|{a_}": table[(arm, v, a_)] for arm in ARMS for v in VARIANTS for a_ in alphas}
        print(f"   translated, all positions, alpha {g_['alpha']}: {pct(g_['translated_hit'])} of test cases flip to the swapped capital;"
              f" strongest control ({g_['strongest_control']}, best alpha) {pct(g_['controls_best_alpha_hit'][g_['strongest_control']])}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"b1_swap{suffix}{('_' + a.tag) if a.tag else ''}.json"), "w", encoding="utf8") as f:
        json.dump({"gate_rule": {"rate": GATE_RATE, "margin": GATE_MARGIN, "p": GATE_P}, "alphas": alphas, "primary": PRIMARY, "gates": gates,
                   "leak": leaks, "summary": summary,
                   "alpha_dev": {k: v[2] for k, v in outs.items()}, "best_test_alpha": {k: {f"{a_}|{b_}": c for (a_, b_), c in v[3].items()} for k, v in outs.items()}},
                  f, indent=1, default=float)

    # ---- result block -------------------------------------------------------------------------------------------------------------
    print("\n" + "=" * 124)
    print("RESULT  step B1 (country swap)" + ("  [quick: smoke test, do not read the numbers]" if a.quick else ""))
    print("  success = the receiver's top answer becomes the swapped country's capital. Cells: rate at the dose chosen on dev  (best rate over all doses on test).")
    for key in a.pairs.split(","):
        if key not in outs:
            continue
        out, table, alpha_dev, best_test = outs[key]
        n_test = int((~out["meta"]["dev"]).sum())
        print(f"\n  {key}  {PRIMARY.get(key, '[exploratory]')}   receiver {out['R']} layer {out['layer_R']}   {n_test} test cases")
        print(f"  {'arm':<12}" + "".join(f"{v + ' positions' if v == 'all' else v + ' only':>26}" for v in VARIANTS))
        for arm in ARMS:
            cells = []
            for v in VARIANTS:
                a_d = alpha_dev[v]
                at = table[(arm, v, a_d)]["test"]["hit"]
                bt = table[(arm, v, best_test[(arm, v)])]["test"]["hit"]
                cells.append(f"{pct(at)} ({pct(bt)} @{best_test[(arm, v)]:g})".rjust(26))
            print(f"  {arm:<12}" + "".join(cells))
        print("  doses chosen on dev for the translated arm: " + ", ".join(f"{v} {alpha_dev[v]}" for v in VARIANTS) + f";  clean receiver success = {pct(out['clean_hit'].mean().item())}")
        g_ = gates[key]
        print(f"  mean logit gain (swapped minus original capital) for translated, all positions: {table[('translated', 'all', g_['alpha'])]['test']['ld_gain']:+.2f};"
              f"  KL to the receiver really reading the swapped sentence: translated {table[('translated', 'all', g_['alpha'])]['test']['kl']:.2f}, "
              f"random {table[('random', 'all', g_['alpha'])]['test']['kl']:.2f}")
        tt = table[("translated", "all", g_["alpha"])]["test"]
        rr = table[("random", "all", g_["alpha"])]["test"]
        print(f"  where the top answer goes (translated, all positions, dose {g_['alpha']}): swapped capital {pct(tt['hit'])}, still the original capital {pct(tt['top_original'])}, "
              f"a different capital {pct(tt['top_other_capital'])}, not a capital {pct(tt['top_non_capital'])};   random vector: swapped {pct(rr['hit'])}, original {pct(rr['top_original'])}, different capital {pct(rr['top_other_capital'])}, not a capital {pct(rr['top_non_capital'])}")
        from collections import Counter
        ids = out["res"][("translated", "all", g_["alpha"])]["top_id"][~out["meta"]["dev"]].tolist()
        tokR = ctx.models[out["R"]].tokenizer
        print("  most common top answers with the translated difference: " + ", ".join(f"{tokR.decode([i])!r} {100 * c / len(ids):.0f}%" for i, c in Counter(ids).most_common(6)))
        print(f"  median rank of the swapped capital among all words (1 = top): unchanged receiver {out['clean_rank'][~out['meta']['dev']].median().item():.0f}, translated {tt['rank_median']:.0f}, "
              f"random {rr['rank_median']:.0f}, wrong pair {table[('wrong', 'all', g_['alpha'])]['test']['rank_median']:.0f}, receiver's own real difference at dose 1 {table[('native', 'all', 1.0)]['test']['rank_median']:.0f}" if 1.0 in alphas else "")
        if key in PRIMARY:
            lk = leaks[key]
            print(f"  leak onto 12 unrelated prompts (the translated last-position vector at the dose used above, {lk['alpha']}, {lk['n_vectors']} vectors): translated KL {lk['translated']['mean_kl']:.3f}, top-1 flips {pct(lk['translated']['top1_flip_rate'])}"
                  f"   random same size KL {lk['random']['mean_kl']:.3f}, flips {pct(lk['random']['top1_flip_rate'])}")
            print(f"  GATE 2 for {key}: (a) rate >= {GATE_RATE:.0%}: {pct(g_['translated_hit'])} {'ok' if g_['a_rate'] else 'NO'};  "
                  f"(b) beats strongest control [{g_['strongest_control']}: {pct(g_['controls_best_alpha_hit'][g_['strongest_control']])}] by >= {100 * GATE_MARGIN:.0f} points: {'ok' if g_['b_margin'] else 'NO'};  "
                  f"(c) paired p = {g_['wilcoxon_p']:.1e} (< {GATE_P}): {'ok' if g_['c_paired'] else 'NO'};  "
                  f"sanity (zero dose exact, native alpha 1 = {pct(g_['native_alpha1_hit'])}): {'ok' if g_['sanity_ok'] else 'FAILED'}   ->  {'PASS' if g_['passes'] else 'FAIL'}")
    print("\n  exploratory pairs (translated, all positions, dose chosen on dev; best control at its own best dose):")
    for key in a.pairs.split(","):
        if key in outs and key not in PRIMARY:
            g_ = gates[key]
            print(f"     {key:<12} translated {pct(g_['translated_hit'])} @alpha {g_['alpha']:g};  strongest control {g_['strongest_control']} {pct(g_['controls_best_alpha_hit'][g_['strongest_control']])};  native alpha 1 {pct(g_['native_alpha1_hit'])}")
    print(f"\n  total time {time.time() - t0:.0f} s.  Cases share countries, so they are not independent: treat differences of a few points as noise.")
    print("=" * 124)
    return 0


if __name__ == "__main__":
    sys.exit(main())
