"""Human-readable probability formatting.

Next-token probabilities span many orders of magnitude (13% for the top word, 0.000001% for a word at
rank 40,000). Printing them with two decimals turns every small one into "0.00%", which hides real change.
"""


def _human_count(n: float) -> str:
    if n < 1_000:
        return f"{n:.0f}"
    if n < 1_000_000:
        return f"{n:,.0f}"
    if n < 1_000_000_000:
        return f"{n / 1e6:.1f} million"
    if n < 1_000_000_000_000:
        return f"{n / 1e9:.1f} billion"
    return f"{n / 1e12:.1f} trillion"


def fmt_prob(p) -> str:
    """p is a probability in [0, 1]. Large values print as %, small ones also show odds ("1 in N")."""
    p = float(p)
    if p <= 0.0:
        return "0"
    if p >= 0.01:
        return f"{p * 100:.2f}%"
    odds = _human_count(1.0 / p)
    if p >= 1e-4:
        return f"{p * 100:.3f}% (about 1 in {odds})"
    return f"{p:.1e} (about 1 in {odds})"


def fmt_prob_pct(pct) -> str:
    """Same as fmt_prob but the input is a percentage (e.g. 0.0012 means 0.0012%)."""
    return fmt_prob(float(pct) / 100.0)
