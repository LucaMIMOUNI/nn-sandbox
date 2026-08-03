#!/usr/bin/env python3
"""Ground-truth check for the backprop visualiser.

The page runs a scalar autograd engine — a graph of nodes, each knowing how to
push a gradient to its inputs — and reads every chip, edge width and diagnostic
straight off it. That engine is the whole promise of the page: if it were wrong,
so would be every number on screen.

This is a port of it. It checks four things, over every activation and a wide
sweep of inputs, weights and targets:

    1. the engine agrees with a central finite difference of the loss,
    2. the engine agrees with the closed form the page prints,

           z    = sum(w_i * x_i) + b
           a    = act(z)
           L    = (a - t)^2
           dL/da = 2(a - t)
           dL/dw_i = dL/da * da/dz * x_i
           dL/db   = dL/da * da/dz
           dL/dx_i = dL/da * da/dz * w_i

    3. a gradient-descent step actually lowers the loss when eta is small
       enough, which is the claim the loss curve makes,
    4. the batch gradient the page applies really is the mean of the per-point
       gradients it draws, and matches a finite difference of the batch loss.

and it compares against PyTorch's own autograd if torch happens to be installed.

    python3 verify/backprop.py
"""

import math
from itertools import product

try:
    import torch
except ImportError:
    torch = None


# ----------------------------------------------------------------------
# the engine, ported node for node from src/backprop.js
# ----------------------------------------------------------------------
class V:
    """One scalar in the graph: a value, a gradient, and how to push it back."""

    def __init__(self, d, prev=(), op=""):
        self.d = float(d)
        self.g = 0.0
        self.prev = tuple(prev)
        self.op = op
        self.local = 0.0
        self._back = lambda: None

    def __add__(self, other):
        out = V(self.d + other.d, (self, other), "+")

        def back():
            self.g += out.g
            other.g += out.g

        out._back = back
        return out

    def __sub__(self, other):
        out = V(self.d - other.d, (self, other), "-")

        def back():
            self.g += out.g
            other.g -= out.g

        out._back = back
        return out

    def __mul__(self, other):
        out = V(self.d * other.d, (self, other), "*")

        def back():
            self.g += other.d * out.g
            other.g += self.d * out.g

        out._back = back
        return out


def unary(z, val, der, op):
    out = V(val, (z,), op)
    out.local = der

    def back():
        z.g += der * out.g

    out._back = back
    return out


def backward(root):
    topo, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n.prev:
            visit(p)
        topo.append(n)

    visit(root)
    for n in topo:
        n.g = 0.0
    root.g = 1.0
    for n in reversed(topo):
        n._back()
    return topo


ACT = {
    "linear": (lambda z: z, lambda z: unary(z, z.d, 1.0, "id")),
    "sigmoid": (
        lambda z: 1.0 / (1.0 + math.exp(-z)),
        lambda z: (lambda s: unary(z, s, s * (1 - s), "sigmoid"))(1.0 / (1.0 + math.exp(-z.d))),
    ),
    "tanh": (
        math.tanh,
        lambda z: (lambda t: unary(z, t, 1 - t * t, "tanh"))(math.tanh(z.d)),
    ),
    "relu": (
        lambda z: max(0.0, z),
        lambda z: unary(z, max(0.0, z.d), 1.0 if z.d > 0 else 0.0, "relu"),
    ),
}


def graph(xs, ws, b, t, act):
    """The page's forward pass, then its backward pass."""
    x = [V(v) for v in xs]
    w = [V(v) for v in ws]
    bb = V(b)
    prod = [w[i] * x[i] for i in range(len(xs))]
    z = prod[0]
    for p in prod[1:]:
        z = z + p
    z = z + bb
    a = ACT[act][1](z)
    e = a - V(t)
    loss = e * e          # nn.MSELoss() on a single output is just the square
    backward(loss)
    return dict(x=x, w=w, b=bb, z=z, a=a, L=loss)


def loss_at(xs, ws, b, t, act):
    """The same forward pass with no graph at all — for differencing."""
    z = b + sum(w * x for w, x in zip(ws, xs))
    return (ACT[act][0](z) - t) ** 2


# ----------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------
def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ----------------------------------------------------------------------
# the datasets, ported from src/backprop.js so this checks what actually
# ships rather than a clean stand-in. Same LCG, same Box-Muller, same call
# order, same rounding — JS Math.round is half-up, Python's round is not.
# ----------------------------------------------------------------------
def r2(v):
    return math.floor(v * 100 + 0.5) / 100


def prng(seed):
    x = [seed & 0xFFFFFFFF]

    def nxt():
        x[0] = (x[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return x[0] / 4294967296

    return nxt


def gauss(r):
    u = max(r(), 1e-9)
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * r())


def make_line():
    r = prng(20250801)
    out = []
    for i in range(13):
        x = r2(-2.1 + i * 0.35)
        out.append(([x, 0.0], r2(0.7 * x + 0.3 + 0.22 * gauss(r))))
    return out


def make_step():
    out = []
    for i in range(14):
        x = r2(-2.1 + i * 0.32)
        out.append(([x, 0.0], 1.0 if x > 0.35 else 0.0))
    return out


def make_plane():
    r = prng(77003)
    out = [([r2(-1.1 + 0.5 * gauss(r)), r2(-0.7 + 0.5 * gauss(r))], 0.0) for _ in range(9)]
    out += [([r2(1.1 + 0.5 * gauss(r)), r2(0.8 + 0.5 * gauss(r))], 1.0) for _ in range(9)]
    return out


SETS = {"line": make_line(), "step": make_step(), "plane": make_plane()}


def check_finite_difference(xs, ws, b, t, act):
    """The engine against (L(p+h) - L(p-h)) / 2h, the check the page shows."""
    g = graph(xs, ws, b, t, act)
    h = 1e-6
    for i in range(len(ws)):
        up, dn = list(ws), list(ws)
        up[i] += h
        dn[i] -= h
        num = (loss_at(xs, up, b, t, act) - loss_at(xs, dn, b, t, act)) / (2 * h)
        assert close(g["w"][i].g, num, 1e-4), (act, "w", i, g["w"][i].g, num)

    num = (loss_at(xs, ws, b + h, t, act) - loss_at(xs, ws, b - h, t, act)) / (2 * h)
    assert close(g["b"].g, num, 1e-4), (act, "b", g["b"].g, num)

    for i in range(len(xs)):
        up, dn = list(xs), list(xs)
        up[i] += h
        dn[i] -= h
        num = (loss_at(up, ws, b, t, act) - loss_at(dn, ws, b, t, act)) / (2 * h)
        assert close(g["x"][i].g, num, 1e-4), (act, "x", i, g["x"][i].g, num)
    return 2 * len(ws) + 1


def check_closed_form(xs, ws, b, t, act):
    """The engine against the factors the readout prints, one by one."""
    g = graph(xs, ws, b, t, act)
    a, dLda, dadz = g["a"].d, g["a"].g, g["a"].local

    assert close(dLda, 2 * (a - t)), (act, dLda, 2 * (a - t))

    z = g["z"].d
    expect = {
        "linear": 1.0,
        "sigmoid": a * (1 - a),
        "tanh": 1 - a * a,
        "relu": 1.0 if z > 0 else 0.0,
    }[act]
    assert close(dadz, expect), (act, "da/dz", dadz, expect)

    for i in range(len(ws)):
        assert close(g["w"][i].g, dLda * dadz * xs[i]), (act, "dL/dw", i)
        assert close(g["x"][i].g, dLda * dadz * ws[i]), (act, "dL/dx", i)
    assert close(g["b"].g, dLda * dadz), (act, "dL/db")
    return 2 * len(ws) + 3


def check_descent(xs, ws, b, t, act, eta=1e-3):
    """A small step against the gradient must not raise the loss."""
    g = graph(xs, ws, b, t, act)
    before = g["L"].d
    ws2 = [w - eta * n.g for w, n in zip(ws, g["w"])]
    b2 = b - eta * g["b"].g
    after = loss_at(xs, ws2, b2, t, act)
    assert after <= before + 1e-12, (act, before, after)
    return 1


def batch(points, ws, b, act):
    """What the page applies: L is the mean of the per-point losses, so dL/dw is
    the mean of the per-point gradients."""
    gs = [graph(xs, ws, b, t, act) for xs, t in points]
    n = len(gs)
    return (sum(g["L"].d for g in gs) / n,
            [sum(g["w"][i].g for g in gs) / n for i in range(len(ws))],
            sum(g["b"].g for g in gs) / n)


def batch_loss_at(points, ws, b, act):
    return sum(loss_at(xs, ws, b, t, act) for xs, t in points) / len(points)


def check_batch(points, ws, b, act):
    """The averaged gradient against a finite difference of the averaged loss."""
    L, gw, gb = batch(points, ws, b, act)
    assert close(L, batch_loss_at(points, ws, b, act)), (act, L)

    h = 1e-6
    for i in range(len(ws)):
        up, dn = list(ws), list(ws)
        up[i] += h
        dn[i] -= h
        num = (batch_loss_at(points, up, b, act) - batch_loss_at(points, dn, b, act)) / (2 * h)
        assert close(gw[i], num, 1e-4), (act, "batch w", i, gw[i], num)

    num = (batch_loss_at(points, ws, b + h, act) - batch_loss_at(points, ws, b - h, act)) / (2 * h)
    assert close(gb, num, 1e-4), (act, "batch b", gb, num)

    # and one small step down must not raise the averaged loss
    eta = 1e-3
    after = batch_loss_at(points, [w - eta * g for w, g in zip(ws, gw)], b - eta * gb, act)
    assert after <= L + 1e-12, (act, L, after)
    return len(ws) + 2


def check_torch(xs, ws, b, t, act):
    """The engine against PyTorch's own autograd."""
    x = torch.tensor(xs, dtype=torch.float64)
    w = torch.tensor(ws, dtype=torch.float64, requires_grad=True)
    bb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    z = (w * x).sum() + bb
    a = {"linear": lambda v: v, "sigmoid": torch.sigmoid,
         "tanh": torch.tanh, "relu": torch.relu}[act](z)
    loss = (a - t) ** 2
    loss.backward()

    g = graph(xs, ws, b, t, act)
    assert close(g["L"].d, loss.item()), (act, g["L"].d, loss.item())
    for i in range(len(ws)):
        assert close(g["w"][i].g, w.grad[i].item()), (act, "w", i)
    assert close(g["b"].g, bb.grad.item()), (act, "b")
    return len(ws) + 2


# ----------------------------------------------------------------------
def main():
    grid = (-2.5, -0.7, 0.0, 0.4, 1.3, 3.0)
    checked = torched = configs = 0

    for act in ACT:
        for x1, w1, b, t in product(grid, grid, (-1.1, 0.0, 0.9), (-1.0, 0.0, 1.0)):
            xs, ws = [x1, -1.6], [w1, 0.55]
            # relu's kink has no derivative, and the difference quotient
            # straddles it — skip only that exact point
            if act == "relu" and abs(ws[0] * xs[0] + ws[1] * xs[1] + b) < 1e-9:
                continue
            configs += 1
            checked += check_finite_difference(xs, ws, b, t, act)
            checked += check_closed_form(xs, ws, b, t, act)
            checked += check_descent(xs, ws, b, t, act)
            if torch is not None:
                torched += check_torch(xs, ws, b, t, act)

    # the batch gradient, over the three datasets the page ships
    for act in ACT:
        for name, points in SETS.items():
            nin = 2 if name == "plane" else 1
            for ws, b in [([0.3, -0.4][:nin], 0.0), ([-0.6, 0.9][:nin], -0.9), ([2.0, 1.5][:nin], 2.0)]:
                pts = [(xs[:nin], t) for xs, t in points]
                if act == "relu" and any(
                        abs(sum(w * x for w, x in zip(ws, xs)) + b) < 1e-9 for xs, _ in pts):
                    continue
                configs += 1
                checked += check_batch(pts, ws, b, act)

    print(f"{configs} configurations, {checked} gradient comparisons — all agree.")
    if torch is not None:
        print(f"{torched} of them also match torch {torch.__version__} autograd.")
    else:
        print("torch not installed — the autograd comparison was skipped "
              "(pip install torch to run it).")

    # each dataset the page ships, trained the way its "train x50" button does.
    # the page claims the curve settles and the residual bars fade; this is that
    # claim, as a number.
    print("\nEach dataset, 50 updates at the eta the page starts it with:")
    for name, points, ws, b, act, eta in [
        ("line",  SETS["line"],  [-0.6],      -0.9, "linear",  0.2),
        ("step",  SETS["step"],  [0.4],        0.1, "sigmoid", 1.5),
        ("plane", SETS["plane"], [0.3, -0.4],  0.0, "sigmoid", 1.0),
    ]:
        nin = len(ws)
        pts = [(xs[:nin], t) for xs, t in points]
        first = batch_loss_at(pts, ws, b, act)
        for _ in range(50):
            _, gw, gb = batch(pts, ws, b, act)
            ws = [w - eta * g for w, g in zip(ws, gw)]
            b -= eta * gb
        last = batch_loss_at(pts, ws, b, act)
        assert last < first, (name, first, last)
        print(f"  {name:<6} eta {eta:<4} L {first:.4f} -> {last:.4f}   "
              f"w {[round(w, 3) for w in ws]}  b {b:.3f}")

    # the two failure modes the diagnostics panel calls out
    print("\nThe two failure modes, and the gradient they produce:")
    for name, xs, ws, b, t, act in [
        ("saturated sigmoid", [3, 3], [2.5, 2.5], 2, 0, "sigmoid"),
        ("dead ReLU", [1, 2], [-1.2, -0.9], -1.5, 1, "relu"),
    ]:
        g = graph(xs, ws, b, t, act)
        print(f"  {name:<19} z = {g['z'].d:>7.3f}   da/dz = {g['a'].local:>10.3e}   "
              f"dL/dw1 = {g['w'][0].g:>10.3e}")


if __name__ == "__main__":
    main()
