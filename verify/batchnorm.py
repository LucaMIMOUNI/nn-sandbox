#!/usr/bin/env python3
"""Ground-truth check for the normalisation visualiser.

The HTML page claims that BatchNorm, LayerNorm, InstanceNorm and GroupNorm are
the same arithmetic over different pooling groups:

    y = gamma * (x - mu) / sqrt(var + eps) + beta

    BatchNorm2d    one (mu, var) per channel,          pooled over (N, H, W)
    LayerNorm      one per sample,                     pooled over (C, H, W)
    InstanceNorm2d one per sample and channel,         pooled over (H, W)
    GroupNorm(G)   one per sample and group of C/G,    pooled over (C/G, H, W)

and that the running statistics BatchNorm keeps for eval() follow

    running = (1 - momentum) * running + momentum * batch_statistic

using the *unbiased* variance for the buffer while normalising with the
*biased* one. This script implements all of that in plain Python and checks it
against PyTorch if torch happens to be installed.

    python3 verify_batchnorm.py
"""

from itertools import product

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

KINDS = ("batch", "layer", "instance", "group")


def pools(kind, N, C, G=1):
    """The pooling groups, as lists of (n, c) pairs sharing one (mu, var)."""
    if kind == "batch":
        return [[(n, c) for n in range(N)] for c in range(C)]
    if kind == "layer":
        return [[(n, c) for c in range(C)] for n in range(N)]
    if kind == "instance":
        return [[(n, c)] for n in range(N) for c in range(C)]
    if kind == "group":
        assert C % G == 0, "num_channels must be divisible by num_groups"
        w = C // G
        return [[(n, c) for c in range(g * w, (g + 1) * w)]
                for n in range(N) for g in range(G)]
    raise ValueError(kind)


def stats(x, members, H, W):
    vals = [x[n][c][h][w] for (n, c) in members for h in range(H) for w in range(W)]
    mean = sum(vals) / len(vals)
    ss = sum((v - mean) ** 2 for v in vals)
    biased = ss / len(vals)
    unbiased = ss / (len(vals) - 1) if len(vals) > 1 else 0.0
    return mean, biased, unbiased


def normalise(x, kind, G=1, eps=1e-5, gamma=None, beta=None, frozen=None):
    """frozen: optional (running_mean, running_var) used instead of batch stats."""
    N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
    gamma = gamma or [1.0] * C
    beta = beta or [0.0] * C
    y = [[[[0.0] * W for _ in range(H)] for _ in range(C)] for _ in range(N)]
    for members in pools(kind, N, C, G):
        mean, var, _ = stats(x, members, H, W)
        for (n, c) in members:
            mu, va = (frozen[0][c], frozen[1][c]) if frozen else (mean, var)
            inv = 1.0 / (va + eps) ** 0.5
            for h, w in product(range(H), range(W)):
                y[n][c][h][w] = gamma[c] * (x[n][c][h][w] - mu) * inv + beta[c]
    return y


def normalise_split(x, kind, G=1, eps=1e-5, gamma=None, beta=None, frozen=None):
    """Same operation, but keeping x-hat — the page now draws it as its own stage,
    so the claim "x-hat lands on 0 with width 1" has to actually hold."""
    N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
    gamma = gamma or [1.0] * C
    beta = beta or [0.0] * C
    zero = lambda: [[[[0.0] * W for _ in range(H)] for _ in range(C)] for _ in range(N)]
    xhat, y = zero(), zero()
    for members in pools(kind, N, C, G):
        mean, var, _ = stats(x, members, H, W)
        for (n, c) in members:
            mu, va = (frozen[0][c], frozen[1][c]) if frozen else (mean, var)
            inv = 1.0 / (va + eps) ** 0.5
            for h, w in product(range(H), range(W)):
                xhat[n][c][h][w] = (x[n][c][h][w] - mu) * inv
                y[n][c][h][w] = gamma[c] * xhat[n][c][h][w] + beta[c]
    return xhat, y


def update_running(x, running_mean, running_var, momentum):
    """One BatchNorm training step, PyTorch's rule."""
    N, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
    for c, members in enumerate(pools("batch", N, C)):
        mean, _, unbiased = stats(x, members, H, W)
        running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean
        running_var[c] = (1 - momentum) * running_var[c] + momentum * unbiased
    return running_mean, running_var


def sample(N, C, H, W, seed=0):
    val = seed * 7 + 3
    x = []
    for n in range(N):
        x.append([])
        for c in range(C):
            x[n].append([])
            for h in range(H):
                x[n][c].append([])
                for w in range(W):
                    val = (val * 1103515245 + 12345) % 2147483648
                    x[n][c][h].append(round((val / 2147483648) * 12 - 4 + 3 * c, 1))
    return x


def flat(y):
    return [v for a in y for b in a for c in b for v in c]


def main():
    grid = [(kind, N, C, H, W, G)
            for kind in KINDS
            for N in (1, 2, 3)
            for C in (1, 2, 4)
            for H, W in ((1, 1), (2, 3), (3, 3))
            for G in (1, 2, 4)
            if kind != "group" or C % G == 0]

    checked = skipped = 0
    for i, (kind, N, C, H, W, G) in enumerate(grid):
        x = sample(N, C, H, W, seed=i)
        gamma = [1.0 + 0.1 * c for c in range(C)]
        beta = [-0.5 + 0.25 * c for c in range(C)]
        ref = normalise(x, kind, G, 1e-5, gamma, beta)

        # every pooled group really is zero-mean / unit-variance without affine
        plain = normalise(x, kind, G, 1e-9)
        for members in pools(kind, N, C, G):
            vals = [plain[n][c][h][w] for (n, c) in members
                    for h in range(H) for w in range(W)]
            m = sum(vals) / len(vals)
            v = sum((t - m) ** 2 for t in vals) / len(vals)
            assert abs(m) < 1e-6, (kind, N, C, H, W, G, m)
            assert len(vals) == 1 or abs(v - 1) < 1e-4, (kind, N, C, H, W, G, v)

        # torch refuses degenerate batches that this reference handles fine
        degenerate = ((kind == "batch" and N * H * W == 1)
                      or (kind == "instance" and H * W == 1)
                      or (kind == "group" and (N * C // G) * H * W == 1))
        if torch is None or degenerate:
            skipped += 1
            continue

        t = torch.tensor(x, dtype=torch.float64).view(N, C, H, W)
        if kind == "batch":
            layer = nn.BatchNorm2d(C, eps=1e-5)
        elif kind == "layer":
            layer = nn.LayerNorm([C, H, W], eps=1e-5)
        elif kind == "instance":
            layer = nn.InstanceNorm2d(C, eps=1e-5, affine=True)
        else:
            layer = nn.GroupNorm(G, C, eps=1e-5)
        layer = layer.double()   # float32 noise blows up when a group has zero variance

        with torch.no_grad():
            g64 = torch.tensor(gamma, dtype=torch.float64)
            b64 = torch.tensor(beta, dtype=torch.float64)
            if kind == "layer":                 # per-element affine, so broadcast ours
                layer.weight.copy_(g64.view(C, 1, 1).expand(C, H, W))
                layer.bias.copy_(b64.view(C, 1, 1).expand(C, H, W))
            else:
                layer.weight.copy_(g64)
                layer.bias.copy_(b64)
            got = layer(t)

        exp = torch.tensor(ref, dtype=torch.float64).view(N, C, H, W)
        assert torch.allclose(got, exp, atol=1e-9), (kind, N, C, H, W, G,
                                                     (got - exp).abs().max().item())
        checked += 1

    # ---- the three stages the "This value, step by step" panel draws ----
    split_fails = 0
    for kind, N, C, H, W, G in grid:
        x = sample(N, C, H, W, seed=N + C)
        gamma = [0.5 + 0.3 * c for c in range(C)]
        beta = [-1.0 + 0.4 * c for c in range(C)]
        eps = 1e-5
        xhat, y = normalise_split(x, kind, G, eps, gamma, beta)
        ref = normalise(x, kind, G, eps, gamma, beta)
        if max(abs(a - b) for a, b in zip(flat(y), flat(ref))) > 1e-9:
            print(f"FAIL split {kind} {N}x{C}x{H}x{W}: y != gamma*xhat + beta")
            split_fails += 1
        for members in pools(kind, N, C, G):
            vals = [xhat[n][c][h][w] for (n, c) in members
                    for h, w in product(range(H), range(W))]
            m = sum(vals) / len(vals)
            v = sum((t - m) ** 2 for t in vals) / len(vals)
            _, var, _ = stats(x, members, H, W)
            if abs(m) > 1e-6 or abs(v - var / (var + eps)) > 1e-6:
                print(f"FAIL xhat {kind} {N}x{C}x{H}x{W}: mean={m:.2e} var={v:.6f}")
                split_fails += 1
                break
    print(f"{len(grid)} configurations: x-hat is mean 0 / variance 1, and y = gamma*x-hat + beta"
          + ("" if not split_fails else f" — {split_fails} FAILURES"))

    print(f"{len(grid)} configurations: every pooling group normalises to mean 0, variance 1.")

    # running statistics
    rm, rv = [0.0] * 3, [1.0] * 3
    for step in range(4):
        rm, rv = update_running(sample(2, 3, 2, 2, seed=100 + step), rm, rv, 0.1)
    if torch is not None:
        bn = nn.BatchNorm2d(3, momentum=0.1).double()
        bn.train()
        for step in range(4):
            bn(torch.tensor(sample(2, 3, 2, 2, seed=100 + step),
                            dtype=torch.float64).view(2, 3, 2, 2))
        assert torch.allclose(bn.running_mean, torch.tensor(rm, dtype=torch.float64), atol=1e-9), (bn.running_mean, rm)
        assert torch.allclose(bn.running_var, torch.tensor(rv, dtype=torch.float64), atol=1e-9), (bn.running_var, rv)
        print(f"{checked} value comparisons and the running-stat update match torch {torch.__version__}.")
    else:
        print(f"torch not installed — {skipped} value comparisons skipped "
              f"(pip install torch to run them).")
        print(f"running_mean after 4 batches: {[round(v, 3) for v in rm]}")

    print("\nHow many values each statistic is estimated from, for a (32, 64, 56, 56) tensor:")
    N, C, H, W = 32, 64, 56, 56
    for kind in KINDS:
        G = 8 if kind == "group" else 1
        groups = pools(kind, N, C, G)
        print(f"  {kind + ('(8)' if kind == 'group' else ''):<12} "
              f"{len(groups):>5} statistics x {len(groups[0]) * H * W:>8} values")


if __name__ == "__main__":
    main()
