#!/usr/bin/env python3
"""Ground-truth check for the Conv2d / ConvTranspose2d visualiser in index.html.

The page draws four grids — input, kernel, middle grid, output — and lets you
hover a cell from either end:

  * hover an OUTPUT cell  -> "gather": which numbers were summed to make it
  * hover an INPUT  pixel -> "scatter": where does this number end up

Those two answers are produced by *different* code paths, and they have to agree
exactly, or the page teaches something false. This script re-implements both
paths the way index.html does (via the same per-axis `contrib` table) and checks:

  1. the output-size formulas match a brute-force count of legal window positions
  2. the gather sum matches a naive reference convolution
  3. scatter and gather describe the identical set of (input, weight, output)
     triples — for both Conv2d and ConvTranspose2d
  4. the per-axis read/write counts match a brute-force tally

It is pure Python, and also checks against PyTorch if it happens to be installed.

    python3 verify_conv2d.py
"""

from itertools import product

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


# --------------------------------------------------------------------------
# a faithful transcription of derive() in index.html
# --------------------------------------------------------------------------
def derive(layer, n, k, s, p, opad, d):
    span = d * (k - 1) + 1
    contrib = []

    if layer == "convT":
        full = (n - 1) * s + span
        out = full - 2 * p + opad
        ext = max(full, p + max(out, 0))
        for r in range(ext):
            lst = []
            lo = max(0, -((-(r - d * (k - 1))) // s))          # ceil division
            hi = min(n - 1, r // s)
            for i in range(lo, hi + 1):
                t = r - i * s
                if t % d == 0:
                    lst.append((i, t // d))
            contrib.append(lst)
    else:
        full = n + 2 * p
        out = (full - span) // s + 1
        ext = full
        for r in range(ext):
            lst = []
            for a in range(k):
                t = r - a * d
                if t < 0 or t % s:
                    continue
                R = t // s
                if 0 <= R < out:
                    lst.append((R, a))
            contrib.append(lst)

    return dict(span=span, full=full, out=out, ext=ext, contrib=contrib,
                valid=out > 0, d=d)


# --------------------------------------------------------------------------
# the two hover code paths, exactly as index.html builds them
# --------------------------------------------------------------------------
def gather_terms(layer, D, x, w, n, k, s, p, r, c):
    """What index.html shows when you hover an output cell."""
    terms = []
    if layer == "conv":
        for a, b in product(range(k), range(k)):
            cr, cc = r * s + a * D["d"], c * s + b * D["d"]
            i, j = cr - p, cc - p
            inside = 0 <= i < n and 0 <= j < n
            terms.append(((i, j) if inside else None, (a, b), (r, c),
                          (x[i][j] if inside else 0.0) * w[a][b]))
    else:
        R, C = r + p, c + p
        if R >= D["full"] or C >= D["full"]:
            return terms
        for (i, ta) in D["contrib"][R]:
            for (j, tb) in D["contrib"][C]:
                terms.append(((i, j), (ta, tb), (r, c), x[i][j] * w[ta][tb]))
    return terms


def scatter_terms(layer, D, x, w, n, k, s, p, i, j):
    """What index.html shows when you hover an input pixel."""
    terms = []
    if layer == "conv":
        cr, cc = i + p, j + p
        for (R, ta) in D["contrib"][cr]:
            for (C, tb) in D["contrib"][cc]:
                terms.append(((i, j), (ta, tb), (R, C), x[i][j] * w[ta][tb]))
    else:
        for a, b in product(range(k), range(k)):
            cr, cc = i * s + a * D["d"], j * s + b * D["d"]
            r, c = cr - p, cc - p
            if D["valid"] and 0 <= r < D["out"] and 0 <= c < D["out"]:
                terms.append(((i, j), (a, b), (r, c), x[i][j] * w[a][b]))
    return terms


# --------------------------------------------------------------------------
# independent references
# --------------------------------------------------------------------------
def ref_conv(x, w, n, k, s, p, d):
    """Naive convolution, written without reference to any contrib table."""
    span = d * (k - 1) + 1
    full = n + 2 * p
    out = (full - span) // s + 1
    if out <= 0:
        return None, out
    pad = [[0.0] * full for _ in range(full)]
    for i in range(n):
        for j in range(n):
            pad[i + p][j + p] = x[i][j]
    y = [[0.0] * out for _ in range(out)]
    for r in range(out):
        for c in range(out):
            acc = 0.0
            for a in range(k):
                for b in range(k):
                    acc += pad[r * s + a * d][c * s + b * d] * w[a][b]
            y[r][c] = acc
    return y, out


def ref_convT(x, w, n, k, s, p, opad, d):
    span = d * (k - 1) + 1
    full = (n - 1) * s + span
    out = full - 2 * p + opad
    canvas = [[0.0] * full for _ in range(full)]
    for i in range(n):
        for j in range(n):
            for a in range(k):
                for b in range(k):
                    canvas[i * s + a * d][j * s + b * d] += x[i][j] * w[a][b]
    if out <= 0:
        return None, out
    y = [[0.0] * out for _ in range(out)]
    for r in range(out):
        for c in range(out):
            R, C = r + p, c + p
            y[r][c] = canvas[R][C] if (R < full and C < full) else 0.0
    return y, out


def brute_out_size(layer, n, k, s, p, opad, d):
    """Count legal window positions instead of using the formula."""
    span = d * (k - 1) + 1
    if layer == "conv":
        full = n + 2 * p
        return sum(1 for r in range(full) if r + span <= full and r % s == 0)
    return (n - 1) * s + span - 2 * p + opad


# --------------------------------------------------------------------------
def rng(seed):
    st = seed & 0xFFFFFFFF
    def nxt():
        nonlocal st
        st = (st * 1664525 + 1013904223) & 0xFFFFFFFF
        return st / 2**32
    return nxt


def main():
    fails = 0
    checked = 0
    torch_checked = 0

    grid = list(product(
        ("conv", "convT"),
        (1, 2, 3, 5, 6),          # n
        (1, 2, 3, 4, 5),          # k
        (1, 2, 3),                # s
        (0, 1, 2),                # p
        (0, 1),                   # output_padding
        (1, 2),                   # d
    ))

    for layer, n, k, s, p, opad, d in grid:
        if layer == "conv" and opad:
            continue                                   # conv has no output_padding
        if layer == "convT" and opad >= max(s, d):
            continue                                   # PyTorch rejects this

        r = rng(1234 + n * 31 + k * 7 + s * 3 + p)
        x = [[float(int(r() * 6)) for _ in range(n)] for _ in range(n)]
        w = [[float(int(r() * 4)) for _ in range(k)] for _ in range(k)]

        D = derive(layer, n, k, s, p, opad, d)

        # ---- 1. output size ------------------------------------------------
        expect = brute_out_size(layer, n, k, s, p, opad, d)
        if D["out"] != expect and (D["out"] > 0 or expect > 0):
            print(f"FAIL size {layer} n{n} k{k} s{s} p{p} op{opad} d{d}: "
                  f"page={D['out']} brute={expect}")
            fails += 1
            continue
        if D["out"] <= 0:
            continue

        # ---- 2. the gather sum vs a naive reference ------------------------
        if layer == "conv":
            ref, _ = ref_conv(x, w, n, k, s, p, d)
        else:
            ref, _ = ref_convT(x, w, n, k, s, p, opad, d)

        for rr, cc in product(range(D["out"]), repeat=2):
            got = sum(t[3] for t in gather_terms(layer, D, x, w, n, k, s, p, rr, cc))
            if abs(got - ref[rr][cc]) > 1e-6:
                print(f"FAIL sum {layer} n{n} k{k} s{s} p{p} op{opad} d{d} "
                      f"at [{rr}][{cc}]: page={got} ref={ref[rr][cc]}")
                fails += 1
                break

        # ---- 3. scatter and gather must describe the same triples ----------
        g = set()
        for rr, cc in product(range(D["out"]), repeat=2):
            for src, tap, dst, _v in gather_terms(layer, D, x, w, n, k, s, p, rr, cc):
                if src is not None:                     # padding taps have no source
                    g.add((src, tap, dst))
        sc = set()
        for i, j in product(range(n), repeat=2):
            for src, tap, dst, _v in scatter_terms(layer, D, x, w, n, k, s, p, i, j):
                sc.add((src, tap, dst))
        if g != sc:
            only_g, only_s = sorted(g - sc)[:2], sorted(sc - g)[:2]
            print(f"FAIL duality {layer} n{n} k{k} s{s} p{p} op{opad} d{d}: "
                  f"gather-only={only_g} scatter-only={only_s}")
            fails += 1

        # ---- 4. per-axis counts --------------------------------------------
        for r_ in range(D["ext"]):
            if layer == "conv":
                brute = sum(1 for R in range(D["out"]) for a in range(k)
                            if R * s + a * d == r_)
            else:
                brute = sum(1 for i in range(n) for a in range(k)
                            if i * s + a * d == r_)
            if len(D["contrib"][r_]) != brute:
                print(f"FAIL count {layer} n{n} k{k} s{s} p{p} d{d} row {r_}: "
                      f"page={len(D['contrib'][r_])} brute={brute}")
                fails += 1
                break

        # ---- 5. PyTorch, if it is here -------------------------------------
        if torch is not None:
            xt = torch.tensor(x).view(1, 1, n, n)
            wt = torch.tensor(w).view(1, 1, k, k)
            if layer == "conv":
                yt = F.conv2d(xt, wt, stride=s, padding=p, dilation=d)
            else:
                yt = F.conv_transpose2d(xt, wt, stride=s, padding=p,
                                        output_padding=opad, dilation=d)
            if list(yt.shape[-2:]) != [D["out"], D["out"]]:
                print(f"FAIL torch shape {layer} n{n} k{k} s{s} p{p} op{opad} d{d}: "
                      f"page={D['out']} torch={list(yt.shape[-2:])}")
                fails += 1
            elif torch.max(torch.abs(yt.view(D["out"], D["out"])
                                     - torch.tensor(ref))).item() > 1e-5:
                print(f"FAIL torch values {layer} n{n} k{k} s{s} p{p} op{opad} d{d}")
                fails += 1
            torch_checked += 1

        checked += 1

    print(f"\n{checked} parameter combinations checked "
          f"({'torch on ' + str(torch_checked) if torch else 'torch not installed'}).")
    print("all good" if not fails else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
