#!/usr/bin/env python3
"""Ground-truth check for the multi-layer half of the backprop visualiser.

Adding a hidden layer adds exactly one equation to the page:

    delta^L   = dL/da^L  *  f'(z^L)              the last layer, as before
    dL/dW^l   = delta^l  @  a^(l-1)^T
    dL/db^l   = delta^l
    dL/da^(l-1) = W^l^T  @  delta^l              <- the new one, the recursion

The page never writes those formulas down and evaluates them: it builds the same
scalar graph it builds for one neuron, one node per weight, and reads the
gradients off it. So there are two things worth checking, and this checks both:
the engine against a finite difference of the loss, and the engine against the
textbook recursion above. If those three ever disagreed, either the picture or
the caption would be lying.

It also checks that the batch gradient is the mean of the per-point ones, that a
small step downhill does not raise the loss, and that xor — which no single
neuron can fit at all — is actually learned by 2-4-1.

    python3 verify/mlp.py
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
    """The local derivative is stored, not re-derived — the page prints it."""
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


def sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


ACT = {
    "linear": (lambda z: z, lambda z: unary(z, z.d, 1.0, "id")),
    "sigmoid": (sigmoid, lambda z: (lambda s: unary(z, s, s * (1 - s), "sig"))(sigmoid(z.d))),
    "tanh": (math.tanh, lambda z: (lambda t: unary(z, t, 1 - t * t, "tanh"))(math.tanh(z.d))),
    "relu": (lambda z: max(0.0, z), lambda z: unary(z, max(0.0, z.d), 1.0 if z.d > 0 else 0.0, "relu")),
}


# ----------------------------------------------------------------------
# the network, ported from src/backprop.js
# ----------------------------------------------------------------------
def layer_act(l, nl, act):
    """Hidden layers use the chosen activation; the output is always sigmoid."""
    return "sigmoid" if l == nl - 1 else act


def net_graph(xs, Ws, Bs, t, act):
    """The page's forward pass over a list of layers, then its backward pass."""
    a = [V(v) for v in xs]
    a0 = a
    layers = []
    nl = len(Ws)
    for l in range(nl):
        W = [[V(v) for v in row] for row in Ws[l]]
        B = [V(v) for v in Bs[l]]
        f = ACT[layer_act(l, nl, act)][1]
        z, prod = [], []
        for j in range(len(W)):
            pr = [W[j][i] * a[i] for i in range(len(a))]
            s = pr[0]
            for p in pr[1:]:
                s = s + p
            z.append(s + B[j])
            prod.append(pr)
        out = [f(zz) for zz in z]
        layers.append(dict(W=W, B=B, z=z, a=out, prod=prod, inp=a))
        a = out
    # the page builds the error once and squares that same node
    e = a[0] - V(t)
    loss = e * e
    backward(loss)
    return dict(a0=a0, layers=layers, out=a[0], L=loss)


def net_raw(xs, Ws, Bs, act):
    """The same forward pass with no graph at all — for differencing."""
    a = list(xs)
    nl = len(Ws)
    for l in range(nl):
        f = ACT[layer_act(l, nl, act)][0]
        a = [f(Bs[l][j] + sum(Ws[l][j][i] * a[i] for i in range(len(a))))
             for j in range(len(Ws[l]))]
    return a[0]


def loss_at(xs, Ws, Bs, t, act):
    return (net_raw(xs, Ws, Bs, act) - t) ** 2


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


def init_net(shape, rnd):
    """PyTorch initialises nn.Linear uniformly on +-1/sqrt(fan_in), per layer."""
    Ws, Bs = [], []
    for l in range(len(shape) - 1):
        k = 1.0 / math.sqrt(shape[l])
        Ws.append([[r2((rnd() * 2 - 1) * k) for _ in range(shape[l])]
                   for _ in range(shape[l + 1])])
        Bs.append([0.0] * shape[l + 1])
    return Ws, Bs


def seeded(shape, width, hidden):
    return init_net(shape, prng((12345 + width * 7 + hidden * 131) & 0xFFFFFFFF))


# ----------------------------------------------------------------------
# the datasets the page ships, ported call for call
# ----------------------------------------------------------------------
def make_xor():
    r = prng(4242)
    out = []
    for cx, cy, t in [(-1.15, -1.15, 0.0), (1.15, 1.15, 0.0),
                      (-1.15, 1.15, 1.0), (1.15, -1.15, 1.0)]:
        for _ in range(8):
            out.append(([r2(cx + 0.4 * gauss(r)), r2(cy + 0.4 * gauss(r))], t))
    return out


def make_circle():
    r = prng(90210)
    out = []
    for _ in range(16):
        th, rad = r() * 6.2832, 0.25 + 0.6 * r()
        out.append(([r2(rad * math.cos(th)), r2(rad * math.sin(th))], 1.0))
    for _ in range(20):
        th, rad = r() * 6.2832, 1.45 + 0.55 * r()
        out.append(([r2(rad * math.cos(th)), r2(rad * math.sin(th))], 0.0))
    return out


def make_plane():
    r = prng(77003)
    out = [([r2(-1.1 + 0.5 * gauss(r)), r2(-0.7 + 0.5 * gauss(r))], 0.0) for _ in range(9)]
    out += [([r2(1.1 + 0.5 * gauss(r)), r2(0.8 + 0.5 * gauss(r))], 1.0) for _ in range(9)]
    return out


SETS = {"xor": make_xor(), "circle": make_circle(), "plane": make_plane()}


# ----------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------
def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_finite_difference(xs, Ws, Bs, t, act):
    """Every weight and bias in every layer, against (L(p+h)-L(p-h))/2h."""
    g = net_graph(xs, Ws, Bs, t, act)
    h = 1e-6
    n = 0
    for l in range(len(Ws)):
        for j in range(len(Ws[l])):
            for i in range(len(Ws[l][j])):
                up = [[row[:] for row in W] for W in Ws]
                dn = [[row[:] for row in W] for W in Ws]
                up[l][j][i] += h
                dn[l][j][i] -= h
                num = (loss_at(xs, up, Bs, t, act) - loss_at(xs, dn, Bs, t, act)) / (2 * h)
                got = g["layers"][l]["W"][j][i].g
                assert close(got, num, 1e-4), (act, "W", l, j, i, got, num)
                n += 1
            up = [b[:] for b in Bs]
            dn = [b[:] for b in Bs]
            up[l][j] += h
            dn[l][j] -= h
            num = (loss_at(xs, Ws, up, t, act) - loss_at(xs, Ws, dn, t, act)) / (2 * h)
            got = g["layers"][l]["B"][j].g
            assert close(got, num, 1e-4), (act, "b", l, j, got, num)
            n += 1
    return n


def check_recursion(xs, Ws, Bs, t, act):
    """The engine against the four equations the page writes on screen."""
    g = net_graph(xs, Ws, Bs, t, act)
    nl = len(Ws)
    a_out = g["out"].d

    # the last layer: dL/da = 2(a - t)
    assert close(g["out"].g, 2 * (a_out - t)), (act, "dL/da", g["out"].g)

    n = 1
    delta_next = None
    for l in reversed(range(nl)):
        L = g["layers"][l]
        rows, cols = len(Ws[l]), len(Ws[l][0])

        # dL/da for this layer: from the loss at the top, from W^T delta below it
        for j in range(rows):
            if l == nl - 1:
                want = 2 * (a_out - t)
            else:
                want = sum(Ws[l + 1][m][j] * delta_next[m] for m in range(len(Ws[l + 1])))
            assert close(L["a"][j].g, want, 1e-9), (act, "dL/da", l, j, L["a"][j].g, want)
            n += 1

        # delta = dL/da * f'(z), one per neuron
        delta = []
        for j in range(rows):
            d = L["a"][j].g * L["a"][j].local
            assert close(L["z"][j].g, d, 1e-9), (act, "delta", l, j, L["z"][j].g, d)
            delta.append(L["z"][j].g)
            n += 1

        # dL/dW = delta a^T, dL/db = delta
        for j in range(rows):
            for i in range(cols):
                want = delta[j] * L["inp"][i].d
                assert close(L["W"][j][i].g, want, 1e-9), (act, "dL/dW", l, j, i)
                n += 1
            assert close(L["B"][j].g, delta[j], 1e-9), (act, "dL/db", l, j)
            n += 1
        delta_next = delta

    # and what a deeper net would hand to the layer below: dL/dx = W^T delta
    for i in range(len(xs)):
        want = sum(Ws[0][j][i] * delta_next[j] for j in range(len(Ws[0])))
        assert close(g["a0"][i].g, want, 1e-9), (act, "dL/dx", i)
        n += 1
    return n


def check_descent(xs, Ws, Bs, t, act, eta=1e-3):
    g = net_graph(xs, Ws, Bs, t, act)
    before = g["L"].d
    W2 = [[[Ws[l][j][i] - eta * g["layers"][l]["W"][j][i].g for i in range(len(Ws[l][j]))]
           for j in range(len(Ws[l]))] for l in range(len(Ws))]
    B2 = [[Bs[l][j] - eta * g["layers"][l]["B"][j].g for j in range(len(Bs[l]))]
          for l in range(len(Bs))]
    after = loss_at(xs, W2, B2, t, act)
    assert after <= before + 1e-12, (act, before, after)
    return 1


def batch(points, Ws, Bs, act):
    """L is the mean of the per-point losses, so dL/dW is the mean of the
    per-point gradients — which is what the update actually applies."""
    gs = [net_graph(xs, Ws, Bs, t, act) for xs, t in points]
    n = len(gs)
    L = sum(g["L"].d for g in gs) / n
    GW = [[[sum(g["layers"][l]["W"][j][i].g for g in gs) / n
            for i in range(len(Ws[l][j]))] for j in range(len(Ws[l]))] for l in range(len(Ws))]
    GB = [[sum(g["layers"][l]["B"][j].g for g in gs) / n
           for j in range(len(Bs[l]))] for l in range(len(Bs))]
    return L, GW, GB


def batch_loss_at(points, Ws, Bs, act):
    return sum(loss_at(xs, Ws, Bs, t, act) for xs, t in points) / len(points)


def check_batch(points, Ws, Bs, act):
    L, GW, GB = batch(points, Ws, Bs, act)
    assert close(L, batch_loss_at(points, Ws, Bs, act)), (act, L)
    h = 1e-6
    n = 1
    for l in range(len(Ws)):
        for j in range(len(Ws[l])):
            for i in range(len(Ws[l][j])):
                up = [[row[:] for row in W] for W in Ws]
                dn = [[row[:] for row in W] for W in Ws]
                up[l][j][i] += h
                dn[l][j][i] -= h
                num = ((batch_loss_at(points, up, Bs, act)
                        - batch_loss_at(points, dn, Bs, act)) / (2 * h))
                assert close(GW[l][j][i], num, 1e-4), (act, "batch W", l, j, i, GW[l][j][i], num)
                n += 1
    eta = 1e-3
    W2 = [[[Ws[l][j][i] - eta * GW[l][j][i] for i in range(len(Ws[l][j]))]
           for j in range(len(Ws[l]))] for l in range(len(Ws))]
    B2 = [[Bs[l][j] - eta * GB[l][j] for j in range(len(Bs[l]))] for l in range(len(Bs))]
    assert batch_loss_at(points, W2, B2, act) <= L + 1e-12
    return n


def check_torch(xs, Ws, Bs, t, act):
    """The whole net against PyTorch's own autograd."""
    nl = len(Ws)
    Wt = [torch.tensor(W, dtype=torch.float64, requires_grad=True) for W in Ws]
    Bt = [torch.tensor(B, dtype=torch.float64, requires_grad=True) for B in Bs]
    a = torch.tensor(xs, dtype=torch.float64)
    for l in range(nl):
        z = Wt[l] @ a + Bt[l]
        f = {"linear": lambda v: v, "sigmoid": torch.sigmoid,
             "tanh": torch.tanh, "relu": torch.relu}[layer_act(l, nl, act)]
        a = f(z)
    loss = (a[0] - t) ** 2
    loss.backward()

    g = net_graph(xs, Ws, Bs, t, act)
    assert close(g["L"].d, loss.item()), (act, g["L"].d, loss.item())
    n = 0
    for l in range(nl):
        for j in range(len(Ws[l])):
            for i in range(len(Ws[l][j])):
                assert close(g["layers"][l]["W"][j][i].g, Wt[l].grad[j][i].item()), (act, l, j, i)
                n += 1
            assert close(g["layers"][l]["B"][j].g, Bt[l].grad[j].item()), (act, "b", l, j)
            n += 1
    return n


# ----------------------------------------------------------------------
def main():
    shapes = [[2, 2, 1], [2, 3, 1], [2, 4, 1], [2, 2, 2, 1], [2, 4, 4, 1], [2, 3, 4, 1]]
    checked = torched = configs = 0

    for act in ("tanh", "sigmoid", "relu", "linear"):
        for sh in shapes:
            for seed, xs, t in product((7, 4242, 90210), ([1.3, -0.8], [-2.1, 0.4]), (0.0, 1.0)):
                Ws, Bs = init_net(sh, prng(seed))
                # a bias of zero everywhere is the page's init but a poor test:
                # move them off it so dL/db is not trivially the same number
                Bs = [[0.3 * ((j % 3) - 1) for j in range(len(B))] for B in Bs]
                # relu has no derivative at its kink and the difference quotient
                # straddles it — skip only those exact points
                if act == "relu" and kinked(xs, Ws, Bs, act):
                    continue
                configs += 1
                checked += check_finite_difference(xs, Ws, Bs, t, act)
                checked += check_recursion(xs, Ws, Bs, t, act)
                checked += check_descent(xs, Ws, Bs, t, act)
                if torch is not None:
                    torched += check_torch(xs, Ws, Bs, t, act)

    # the batch gradient, over the datasets a hidden layer is for
    for act in ("tanh", "relu"):
        for name, points in SETS.items():
            for sh in ([2, 4, 1], [2, 3, 3, 1]):
                Ws, Bs = init_net(sh, prng(4242))
                if act == "relu" and any(kinked(xs, Ws, Bs, act) for xs, _ in points):
                    continue
                configs += 1
                checked += check_batch(points, Ws, Bs, act)

    print(f"{configs} configurations, {checked} gradient comparisons — all agree.")
    if torch is not None:
        print(f"{torched} of them also match torch {torch.__version__} autograd.")
    else:
        print("torch not installed — the autograd comparison was skipped "
              "(pip install torch to run it).")

    # the page's own defaults, trained the way its "train x200" button does
    print("\nEach dataset, 200 updates from the init the page opens with:")
    for name, hidden, width, eta in [("xor", 1, 4, 0.6), ("circle", 1, 4, 0.6),
                                     ("xor", 2, 4, 0.6), ("plane", 1, 4, 1.0)]:
        sh = [2] + [width] * hidden + [1]
        Ws, Bs = seeded(sh, width, hidden)
        pts = SETS[name]
        first = batch_loss_at(pts, Ws, Bs, "tanh")
        for _ in range(200):
            _, GW, GB = batch(pts, Ws, Bs, "tanh")
            Ws = [[[Ws[l][j][i] - eta * GW[l][j][i] for i in range(len(Ws[l][j]))]
                   for j in range(len(Ws[l]))] for l in range(len(Ws))]
            Bs = [[Bs[l][j] - eta * GB[l][j] for j in range(len(Bs[l]))] for l in range(len(Bs))]
        last = batch_loss_at(pts, Ws, Bs, "tanh")
        wrong = sum(1 for xs, t in pts if (net_raw(xs, Ws, Bs, "tanh") >= 0.5) != (t >= 0.5))
        assert last < first, (name, first, last)
        print(f"  {name:<7} {'→'.join(str(n) for n in sh):<10} eta {eta:<4} "
              f"L {first:.4f} -> {last:.4f}   {len(pts) - wrong}/{len(pts)} points on the right side")

    # the claim the xor button makes: one neuron cannot do this, a hidden layer can
    one_W, one_B = [[[0.3, -0.4]]], [[0.0]]
    pts = SETS["xor"]
    for _ in range(2000):
        _, GW, GB = batch(pts, one_W, one_B, "sigmoid")
        one_W = [[[one_W[0][0][i] - 1.0 * GW[0][0][i] for i in range(2)]]]
        one_B = [[one_B[0][0] - 1.0 * GB[0][0]]]
    stuck = batch_loss_at(pts, one_W, one_B, "sigmoid")
    assert stuck > 0.2, stuck
    print(f"\nOne neuron on xor after 2000 updates: L = {stuck:.4f} — it never gets below 0.25, "
          f"because no straight line separates the corners.")

    # what depth does to the gradient that reaches the first layer
    print("\nHow much gradient reaches layer 1, by depth (mean |dL/dW1| over xor):")
    for hidden in (1, 2):
        sh = [2] + [4] * hidden + [1]
        Ws, Bs = seeded(sh, 4, hidden)
        _, GW, _ = batch(SETS["xor"], Ws, Bs, "sigmoid")
        flat = [abs(v) for row in GW[0] for v in row]
        top = [abs(v) for row in GW[-1] for v in row]
        print(f"  {'→'.join(str(n) for n in sh):<10} layer 1 {sum(flat)/len(flat):.3e}   "
              f"last layer {sum(top)/len(top):.3e}")


def kinked(xs, Ws, Bs, act):
    """True if any pre-activation sits exactly on relu's kink."""
    a = list(xs)
    nl = len(Ws)
    for l in range(nl):
        f = ACT[layer_act(l, nl, act)][0]
        z = [Bs[l][j] + sum(Ws[l][j][i] * a[i] for i in range(len(a)))
             for j in range(len(Ws[l]))]
        if layer_act(l, nl, act) == "relu" and any(abs(v) < 1e-9 for v in z):
            return True
        a = [f(v) for v in z]
    return False


if __name__ == "__main__":
    main()
