"""
A neural translator between two residual streams, started from the linear solution.

    f(x) = (x - mu_x) @ W + mu_y  +  sigma_y * MLP((x - mu_x) / sigma_x)

W is the ridge map from step A1 (so training starts exactly at the linear translator and can only add a non-linear correction); the MLP is
one hidden layer with GELU whose last layer starts at zero. Trained with mean squared error on targets divided by their per-dimension standard
deviation, so every dimension counts equally (the same view as the per-dimension R-squared of step A1).
The translated CHANGE of an edit is f(h + delta) - f(h) (see export_eval.MlpTranslator).
"""
import os

import torch
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "transfer")


class ResidualMlpMap(nn.Module):
    def __init__(self, W, mu_x, mu_y, sigma_x, sigma_y, hidden=2048):
        super().__init__()
        dx, dy = W.shape
        self.W = nn.Parameter(W.clone().float())
        self.register_buffer("mu_x", mu_x.clone().float())
        self.register_buffer("mu_y", mu_y.clone().float())
        self.register_buffer("sigma_x", sigma_x.clone().float().clamp(min=1e-3))
        self.register_buffer("sigma_y", sigma_y.clone().float().clamp(min=1e-3))
        self.up = nn.Linear(dx, hidden)
        self.down = nn.Linear(hidden, dy)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        self.hidden = hidden

    def forward(self, x):
        xc = x - self.mu_x
        return xc @ self.W + self.mu_y + self.sigma_y * self.down(torch.nn.functional.gelu(self.up(xc / self.sigma_x)))


def perdim_r2(pred, y, y_mean=None):
    """Per-dimension R-squared against the mean of y (equal weight for every dimension)."""
    sse = ((y - pred) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0).clamp(min=1e-12)
    return float((1 - sse / sst).mean())


def fit(model, X, Y, Xa, Ya, epochs=12, batch=2048, lr=1e-3, lr_w=1e-4, patience=2, log=print):
    """Train on (X, Y) [CPU tensors]; early-stop on the held-out half A (Xa, Ya) [device tensors]. Returns the best state dict and its score."""
    dev = next(model.parameters()).device
    params = [{"params": [p for n, p in model.named_parameters() if n != "W"], "lr": lr, "weight_decay": 1e-4},
              {"params": [model.W], "lr": lr_w, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params)
    n = len(X)
    steps_per_epoch = n // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr_w], total_steps=epochs * steps_per_epoch, pct_start=0.1)
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            x, y = X[idx].to(dev, non_blocking=True), Y[idx].to(dev, non_blocking=True)
            loss = (((model(x) - y) / model.sigma_y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            score = perdim_r2(torch.cat([model(Xa[i:i + 8192]) for i in range(0, len(Xa), 8192)]), Ya)
        log(f"      epoch {ep + 1:>2}: train loss {tot / steps_per_epoch:.4f}, held-out (half A) per-dim R2 {score:.4f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best_state, best


def load_mlp_translator(key, dev, path=None):
    """The trained neural map for a layer pair key such as 's2m_L8_L16' (written by tools/transfer/07_fit_mlp_maps.py)."""
    blob = torch.load(path or os.path.join(OUT_DIR, "mlp_maps.pt"), weights_only=False)[key]
    z = torch.zeros
    m = ResidualMlpMap(z(blob["dx"], blob["dy"]), z(blob["dx"]), z(blob["dy"]), z(blob["dx"]) + 1, z(blob["dy"]) + 1, hidden=blob["hidden"])
    m.load_state_dict(blob["state"])
    return m.to(dev).eval()
