"""
Step B1 building blocks: the country-swap positive control.

A test case is an ordered pair of countries (a, b). The difference "state for b minus state for a" is read in a SOURCE model, translated,
and added to a RECEIVER model while it reads the sentence for a. If the channel carries meaning, the receiver should start answering with
b's capital. Everything here is new code; nothing in the older modules is touched.
"""
import torch
import torch.nn.functional as F

COUNTRY_CAPITAL = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo", "China": "Beijing", "Russia": "Moscow",
    "Egypt": "Cairo", "Canada": "Ottawa", "India": "Delhi", "Poland": "Warsaw", "Greece": "Athens", "Turkey": "Ankara", "Ireland": "Dublin",
    "Austria": "Vienna", "Portugal": "Lisbon", "Sweden": "Stockholm", "Norway": "Oslo", "Denmark": "Copenhagen", "Finland": "Helsinki",
    "Hungary": "Budapest", "England": "London", "Scotland": "Edinburgh", "Australia": "Canberra", "Iran": "Tehran", "Iraq": "Baghdad",
    "Israel": "Jerusalem", "Cuba": "Havana", "Peru": "Lima", "Chile": "Santiago", "Syria": "Damascus", "Libya": "Tripoli", "Ukraine": "Kiev",
    "Belgium": "Brussels", "Switzerland": "Bern", "Mexico": "Mexico", "Thailand": "Bangkok", "Korea": "Seoul", "Pakistan": "Islamabad",
    "Afghanistan": "Kabul", "Lebanon": "Beirut",
}
# the plain "The capital of X is" makes both models answer "the" or "a"; these wordings force a city name
TEMPLATES = ["The capital of {X} is the city of", "Everyone knows that the capital of {X} is the city of"]

NEUTRAL = [
    "This boy is", "The weather today is", "She opened the door and", "My favourite food is", "The meeting will start at", "He was born in",
    "The best way to learn is", "I think that the answer is", "The company announced that", "In the morning we", "The old man said",
    "Yesterday I went to the",
]


def single_token_pairs(tokenizer) -> dict:
    """Countries and capitals that are each one token with a leading space (so every sentence has the same length)."""
    one = lambda w: len(tokenizer(" " + w, add_special_tokens=False)["input_ids"]) == 1
    return {c: p for c, p in COUNTRY_CAPITAL.items() if one(c) and one(p)}


def token_id(tokenizer, word):
    return tokenizer(" " + word, add_special_tokens=False)["input_ids"][0]


def last_logits(model, tokens, chunk=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(tokens), chunk):
            out.append(model(tokens[i:i + chunk])[:, -1].float())
    return torch.cat(out)


def run_injected(model, tokens, delta, layer, chunk=64):
    """Last-position logits [P, V] when `delta` [P, L, d] is added to blocks.<layer>.hook_resid_pre while the model reads `tokens` [P, L].
    Only the last position is unembedded (the model's output table is the single most expensive step), which gives the same numbers as
    taking the last position of a full forward pass."""
    name = f"blocks.{layer}.hook_resid_pre"
    out = []
    for i in range(0, len(tokens), chunk):
        d = delta[i:i + chunk]
        inject = lambda resid, hook, d=d: resid + d
        grab = {}

        def keep_last(x, hook):
            grab["h"] = x[:, -1:].clone()

        with torch.no_grad():
            model.run_with_hooks(tokens[i:i + chunk], return_type=None, fwd_hooks=[(name, inject), ("ln_final.hook_normalized", keep_last)])
            out.append(model.unembed(grab["h"])[:, 0].float())
    return torch.cat(out)


def case_metrics(logits, cap_a, cap_b, p_native_b, cap_other=None, cap_set=None):
    """Per-case numbers for injected logits [P, V]:
    hit      the swapped capital is the top answer
    ld       logit(swapped capital) - logit(original capital)
    kl       KL(receiver's own distribution on b's sentence || injected distribution): 0 = identical to the receiver really reading b
    rank_b   rank of the swapped capital among all words (1 = top)
    hit_other (wrong-pair arm only) the top answer is the capital the WRONG pair was heading for
    top_original / top_other_capital / top_non_capital   where the top answer goes when it is not the swapped capital
                                                         (the original capital / some other capital in our list / not a capital at all)"""
    top = logits.argmax(-1)
    ar = torch.arange(len(logits), device=logits.device)
    logp = F.log_softmax(logits, dim=-1)
    kl = (p_native_b * (torch.log(p_native_b.clamp(min=1e-30)) - logp)).sum(-1)
    lb = logits[ar, cap_b]
    out = {"hit": (top == cap_b).float(), "ld": lb - logits[ar, cap_a], "kl": kl, "rank_b": (logits > lb[:, None]).sum(-1).float() + 1, "top_id": top}
    if cap_other is not None:
        out["hit_other"] = (top == cap_other).float()
    if cap_set is not None:
        in_set = torch.isin(top, cap_set)
        out["top_original"] = (top == cap_a).float()
        out["top_other_capital"] = (in_set & (top != cap_b) & (top != cap_a)).float()
        out["top_non_capital"] = (~in_set).float()
    return {k: v.cpu() for k, v in out.items()}


def wilson(k, n, z=1.96):
    """95% Wilson interval for a rate (indicative only here: the cases share countries, so they are not independent)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)
