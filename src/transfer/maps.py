"""
Linear maps between two residual streams: exact ridge solution from streamed sums, and held-out scoring.

A map is  prediction = (x - mu_x) @ W + mu_y.  W comes from ridge regression on centred data (least squares with a small stabilising
term). The ridge strength is given relative to the average variance of the source dimensions: lambda = c * trace(Sxx) / dx.

Why not gradient descent: for a linear map under squared error the ridge solution IS the optimum gradient descent would creep towards;
computing it directly is exact and quick. Gradient descent matters once the map is non-linear or the loss is not squared error.
"""
import torch
import torch.nn.functional as F


class RidgeSolver:
    """One eigen-decomposition of a centred source covariance, reused for every target and every ridge strength."""

    def __init__(self, Sxx: torch.Tensor):
        e, V = torch.linalg.eigh(Sxx)
        self.e = e.clamp(min=0)
        self.V = V
        self.unit = self.e.mean()        # trace(Sxx) / dx

    def solve_many(self, Sxy: torch.Tensor, cs) -> dict:
        """{c: W (float64, [dx, dy])} for each relative ridge strength c."""
        Q = self.V.T @ Sxy
        return {c: self.V @ (Q / (self.e + c * self.unit)[:, None]) for c in cs}


class TargetStats:
    """Mean and variance of a target layer's states on held-out text, separately for two halves (0 and 1)."""

    def __init__(self, dim: int, device):
        self.n = [0, 0]
        self.s1 = torch.zeros(2, dim, device=device, dtype=torch.float64)
        self.s2 = torch.zeros(2, dim, device=device, dtype=torch.float64)

    def add(self, half: int, y: torch.Tensor):
        y = y.double()
        self.n[half] += y.shape[0]
        self.s1[half] += y.sum(0)
        self.s2[half] += (y * y).sum(0)

    def sst(self, half: int) -> torch.Tensor:
        """Per-dimension sum of squared deviations from the held-out mean."""
        return self.s2[half] - self.s1[half] ** 2 / self.n[half]


class MapScore:
    """Held-out error of one map, accumulated over batches, separately for halves 0 and 1."""

    def __init__(self, dim: int, device):
        self.sse = torch.zeros(2, dim, device=device, dtype=torch.float64)
        self.cos = torch.zeros(2, device=device, dtype=torch.float64)
        self.n = [0, 0]

    @torch.no_grad()
    def add(self, half: int, x, y, W, mu_x, mu_y):
        pc = (x - mu_x) @ W                  # prediction minus the (training) target mean
        yc = y - mu_y
        self.sse[half] += ((yc - pc) ** 2).sum(0).double()
        self.cos[half] += F.cosine_similarity(pc, yc, dim=1).sum().double()
        self.n[half] += x.shape[0]

    def metrics(self, half: int, sst: torch.Tensor) -> dict:
        """raw_r2: share of total variance explained (dominated by the largest dimensions).
        perdim_r2: the same, averaged over dimensions with equal weight (the honest one).
        cosine: mean cosine between predicted and true state after removing the mean (direction agreement)."""
        sst = sst.clamp(min=1e-12)
        return {
            "raw_r2": float(1 - self.sse[half].sum() / sst.sum()),
            "perdim_r2": float((1 - self.sse[half] / sst).mean()),
            "cosine": float(self.cos[half] / max(self.n[half], 1)),
        }
