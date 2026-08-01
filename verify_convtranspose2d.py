#!/usr/bin/env python3
"""Ground-truth check for the ConvTranspose2d visualiser.

The HTML page claims two things:

  1. out = (in - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1
  2. the "stamping" picture: every input pixel adds weight * kernel into a canvas
     of size (in - 1)*stride + dilation*(kernel - 1) + 1, at position i*stride,
     and padding simply crops that canvas.

This script implements (2) with plain Python and checks it against the formula,
and against PyTorch itself if torch happens to be installed.

    python3 verify_convtranspose2d.py
"""

from itertools import product

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def out_size(n, k, s, p, op, d):
    return (n - 1) * s - 2 * p + d * (k - 1) + op + 1


def scatter(x, w, s, p, op, d):
    """Reference ConvTranspose2d for 1 in-channel / 1 out-channel, no bias."""
    n, m = len(x), len(x[0])
    kh, kw = len(w), len(w[0])
    fh = (n - 1) * s + d * (kh - 1) + 1          # canvas height before cropping
    fw = (m - 1) * s + d * (kw - 1) + 1
    canvas = [[0.0] * fw for _ in range(fh)]

    for i, j in product(range(n), range(m)):     # every input pixel stamps the kernel
        for a, b in product(range(kh), range(kw)):
            canvas[i * s + a * d][j * s + b * d] += x[i][j] * w[a][b]

    oh, ow = out_size(n, kh, s, p, op, d), out_size(m, kw, s, p, op, d)
    # padding crops, output_padding un-crops the bottom/right; anything past the
    # canvas edge is a genuine zero row/col.
    return [[canvas[r][c] if 0 <= r < fh and 0 <= c < fw else 0.0
             for c in range(p, p + ow)]
            for r in range(p, p + oh)]


def main():
    grid = [
        dict(n=n, k=k, s=s, p=p, op=op, d=d)
        for n in (1, 2, 3, 5)
        for k in (1, 2, 3, 4, 5)
        for s in (1, 2, 3)
        for p in (0, 1, 2)
        for op in (0, 1, 2)
        for d in (1, 2)
        if op < max(s, d) and out_size(n, k, s, p, op, d) > 0
    ]

    checked = skipped = 0
    for cfg in grid:
        n, k, s, p, op, d = (cfg[key] for key in "n k s p op d".split())
        x = [[1.0 + i * n + j for j in range(n)] for i in range(n)]
        w = [[1.0 + a * k + b for b in range(k)] for a in range(k)]

        ref = scatter(x, w, s, p, op, d)
        assert len(ref) == out_size(n, k, s, p, op, d), cfg

        if torch is None:
            skipped += 1
            continue

        got = F.conv_transpose2d(
            torch.tensor(x).view(1, 1, n, n),
            torch.tensor(w).view(1, 1, k, k),
            stride=s, padding=p, output_padding=op, dilation=d,
        )
        exp = torch.tensor(ref).view(1, 1, len(ref), len(ref[0]))
        assert got.shape == exp.shape, (cfg, got.shape, exp.shape)
        assert torch.allclose(got, exp), cfg
        checked += 1

    print(f"{len(grid)} configurations, shapes match the formula.")
    if torch is None:
        print(f"torch not installed — {skipped} value comparisons skipped "
              f"(pip install torch to run them).")
    else:
        print(f"{checked} value comparisons match torch {torch.__version__}.")

    print("\nA few shapes, 8x8 input:")
    print(f"  {'k':>2} {'s':>2} {'p':>2} {'op':>3} {'d':>2}   out")
    for k, s, p, op, d in [(3, 1, 1, 0, 1), (2, 2, 0, 0, 1), (4, 2, 1, 0, 1),
                           (3, 2, 1, 1, 1), (3, 2, 1, 0, 1), (3, 1, 0, 0, 2)]:
        print(f"  {k:>2} {s:>2} {p:>2} {op:>3} {d:>2}   {out_size(8, k, s, p, op, d)}")


if __name__ == "__main__":
    main()
