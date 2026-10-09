"""
Streaming collection of paired residual-stream statistics from two models that read the same text.

Nothing large is stored. For every pair (source layer l, target layer m) we keep running sums that are enough to solve a least-squares
map exactly, however many tokens pass through: sum of x, sum of y, x^T x, y^T y and x^T y. Memory is constant in the number of tokens.

The first token of every sequence (the start-of-text token) is dropped: its state is unusually large and says nothing about language.
"""
import torch


def capture(model, tokens, layers):
    """Residual stream (blocks.L.hook_resid_pre) at the given layers, for every position except the first.
    Returns {layer: float32 tensor [n_positions, d_model]}. The model is stopped after the deepest layer needed (no output logits)."""
    names = {f"blocks.{l}.hook_resid_pre" for l in layers}
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names, stop_at_layer=max(layers) + 1)
    d = model.cfg.d_model
    return {l: cache[f"blocks.{l}.hook_resid_pre"][:, 1:, :].reshape(-1, d).float() for l in layers}


class PairedStats:
    """Running sums for linear maps between a set of source layers and a set of target layers.

    Data are shifted by a reference vector (the first batch's mean) before summing, and sums are kept in float64: both keep the
    arithmetic accurate when a few dimensions are huge. Covariances do not change under a shift, and `centered()` undoes it."""

    def __init__(self, src_dims: dict, dst_dims: dict, device):
        self.n = 0
        self.shift_x = self.shift_y = None
        dd = dict(device=device, dtype=torch.float64)
        self.sx = {l: torch.zeros(d, **dd) for l, d in src_dims.items()}
        self.sy = {m: torch.zeros(d, **dd) for m, d in dst_dims.items()}
        self.gxx = {l: torch.zeros(d, d, **dd) for l, d in src_dims.items()}
        self.gyy = {m: torch.zeros(d, d, **dd) for m, d in dst_dims.items()}
        self.cxy = {(l, m): torch.zeros(dx, dy, **dd) for l, dx in src_dims.items() for m, dy in dst_dims.items()}

    @torch.no_grad()
    def add(self, X: dict, Y: dict):
        """X: {source layer: [n, dx]}, Y: {target layer: [n, dy]}; row i of every tensor is the same token."""
        if self.shift_x is None:
            self.shift_x = {l: x.mean(0) for l, x in X.items()}
            self.shift_y = {m: y.mean(0) for m, y in Y.items()}
        Xs = {l: x - self.shift_x[l] for l, x in X.items()}
        Ys = {m: y - self.shift_y[m] for m, y in Y.items()}
        for l, x in Xs.items():
            self.sx[l] += x.sum(0).double()
            self.gxx[l] += (x.T @ x).double()
        for m, y in Ys.items():
            self.sy[m] += y.sum(0).double()
            self.gyy[m] += (y.T @ y).double()
        for (l, m), c in self.cxy.items():
            c += (Xs[l].T @ Ys[m]).double()
        self.n += next(iter(Xs.values())).shape[0]

    def centered(self) -> dict:
        """Means and centred sums of products (NOT divided by n): Sxx[l], Syy[m], Sxy[(l, m)]."""
        n = self.n
        return {
            "n": n,
            "mu_x": {l: self.shift_x[l].double() + self.sx[l] / n for l in self.sx},
            "mu_y": {m: self.shift_y[m].double() + self.sy[m] / n for m in self.sy},
            "Sxx": {l: self.gxx[l] - torch.outer(self.sx[l], self.sx[l]) / n for l in self.sx},
            "Syy": {m: self.gyy[m] - torch.outer(self.sy[m], self.sy[m]) / n for m in self.sy},
            "Sxy": {k: c - torch.outer(self.sx[k[0]], self.sy[k[1]]) / n for k, c in self.cxy.items()},
        }
