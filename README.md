# NN sandbox

**→ [Open it in your browser](https://lucamimouni.github.io/nn-sandbox/)**

Interactive visualisations of the tensor transformations inside a neural network —
built to answer "what does *this* parameter actually do to my data?" without
guessing from a formula.

No build step, no dependencies, no network, no tracking. It is one HTML file: save
it and it still works on a plane.

```
git clone https://github.com/LucaMIMOUNI/nn-sandbox
xdg-open nn-sandbox/index.html
```

Everything lives in **[`index.html`](index.html)**, with a menu at the top:

| Tab | Covers |
| --- | --- |
| **Conv2d** | `nn.Conv2d`: `kernel_size`, `stride`, `padding`, `dilation` |
| **ConvTranspose2d** | `nn.ConvTranspose2d`: the same, plus `output_padding` |
| **Normalisation** | `nn.BatchNorm2d` and its siblings `LayerNorm`, `InstanceNorm2d`, `GroupNorm` |

Deep links work: `index.html#conv`, `#convT`, `#norm`.

Every tab shares one layout — parameters on the left, the data being transformed in
the middle, shape and diagnostics on the right, collapsing to one column on narrow
screens — and one interaction: **hover a number and the page works out that number**,
greying out everything that is not part of it.

Every tab also has a **shape panel that takes any input size you type** — 224, 512,
whatever your real tensor is. Sizes small enough to draw are applied to the grids
as well; anything bigger stays in the calculator, and `↺` puts the two back in sync.

## Conv2d and ConvTranspose2d

The two are the same page in two directions, so you can flip between them with one
click and watch which arrows reverse. Both draw four grids:

```
Conv2d                          ConvTranspose2d
input   ⊛  kernel               input   ×  kernel
padded  →  output               canvas  →  output
        slide by s                      crop −p
```

**Hover any number and the page works out that number.** Everything not involved in
that one sum is greyed out, so what is left on screen is exactly the arithmetic being
done, and it is written underneath:

```
THIS CELL   output[3][3]  =  canvas[4][4]
FORMULA     input[1][1]·w[2][2] + input[1][2]·w[2][0] + input[2][1]·w[0][2] + input[2][2]·w[0][0]
VALUES      1·0 + 3·3 + 4·3 + 0·1
MATHS       0 + 9 + 12 + 0  =  21
WHY THESE   only these 4 input pixels have a stamp — anchored at i·2 — that covers row 4, col 4.
```

Input values are amber, kernel weights are blue, in the text and in the grids alike.
Hover a single **kernel weight** to isolate the one term of the current sum that uses it.

### The two directions of hovering

Hovering an **output** cell asks *where did this number come from*: for `Conv2d` the
`k×k` window lights up on the padded input, with a dashed box showing its full span
(dilation holes included); for `ConvTranspose2d` the input pixels whose stamps
reached that cell light up.

Hovering an **input** pixel asks the opposite — *where does this number go*:

- `Conv2d` — every output cell that reads this pixel, with `output[R][C] += v·w` for each.
- `ConvTranspose2d` — the `k²` cells this pixel stamps into.

In both cases the destination cells are repainted **amber with the contribution
itself** — `v·w`, this one pixel's share — rather than leaving the accumulated cell
value on screen. That distinction is the whole point of the scatter picture: a
canvas cell normally holds the sum of *several* stamps, so the number it shows is
larger than the amber number sitting on it, and a caption under the grids says so.
Tick **build up progressively** to watch the totals accumulate stamp by stamp.

### Presets

The preset buttons are parameters lifted straight out of real networks, each
labelled with the shape change at that architecture's real input size:

| Conv2d | | ConvTranspose2d | |
| --- | --- | --- | --- |
| LeNet-5 conv1 | `k5 s1 p0` | DCGAN first layer | `k4 s1 p0`, `1² → 4²` |
| AlexNet conv1 | `k11 s4 p2` | DCGAN / pix2pix ×2 block | `k4 s2 p1` |
| VGG 3×3 | `k3 s1 p1` | U-Net up-conv | `k2 s2 p0` |
| ResNet stem | `k7 s2 p3` | Odena checkerboard | `k3 s2 p1 op1` |
| Inception 1×1 bottleneck | `k1 s1 p0` | FSRCNN ×3 deconv | `k9 s3 p3` |
| MobileNet depthwise | `k3 s2 p1` | size-preserving | `k3 s1 p1` |
| DeepLab dilated | `k3 s1 p2 d2` | | |

A preset sets the sliders, picks a grid small enough to read, and puts the real
input size (224, 112, 32 …) into the shape calculator — so you get both the toy you
can count and the number the paper reports.

### Input modes

- **Matrix** — 1×1 to 12×12 of random integers 0–5 (`size` picks it, `↻ new values`
  rerolls). Small enough that every cell shows its number and you can check the
  arithmetic by hand.
- **MNIST digit** — four real samples from the MNIST test set, bundled as raw bytes.
- **Photo** — a real photograph, box-downsampled to 8 / 14 / 28 px.
  Drop your own image on the panel, or use the file picker.

### Walking through it

`◀ / ▶` (or the arrow keys) step one cell at a time and `▶ play` runs the whole scan
— over **window positions** for `Conv2d` (the classic sliding-window animation) and
over **input pixels** for `ConvTranspose2d` (one stamp at a time), because those are
the units each layer actually iterates.

Every parameter slider spans the same 10 steps, so equivalent settings sit at the
same place on the track. Combinations PyTorch would reject (e.g.
`output_padding >= max(stride, dilation)`) are reachable and called out in
**Diagnostics** rather than being silently clamped away.

The **kernel weights** selector decides what gets convolved or stamped: `box` (all
1, so a value is literally the sum of what landed on it), `gauss`, `edge` (a
Laplacian — negative surround, positive centre, so flat regions cancel to 0), and
`random` small integers so the arithmetic stays readable.

### Conv2d — what the page makes visible

```
H_out = ⌊(H_in + 2·padding − dilation·(kernel_size − 1) − 1) / stride⌋ + 1
```

- **`padding` adds, `stride` throws away.** Tick *colour by how often each pixel is
  read* to see the coverage: border pixels are read fewer times than interior ones,
  and once `stride` exceeds the window span some pixels are read **zero** times and
  their information never reaches the output at all.
- **The `⌊floor⌋` silently drops rows.** When `(H + 2p − span)` is not a multiple of
  `stride`, the last row(s) and column(s) are never covered by any window. The shape
  panel names exactly how many.
- **`dilation` buys receptive field for free** — the same `k²` weights see a
  `d(k−1)+1` patch, at the cost of the holes you can see in the dashed window box.

### The mental model for ConvTranspose2d

The reason it feels confusing is that it is **not** a convolution run backwards over
the input. It is a *scatter*, not a *gather*:

1. Every **input** pixel multiplies the whole kernel and **stamps** the result into a
   larger canvas, anchored at position `i * stride`.
2. Overlapping stamps **add up**.
3. `padding` then **crops** `padding` rows/cols off every side of that canvas —
   it removes output, it does not add any.
4. `output_padding` gives back rows/cols on the **bottom and right only**, to break
   the tie between the several input sizes a forward `Conv2d` could have come from.

```
H_out = (H_in − 1)·stride − 2·padding + dilation·(kernel_size − 1) + output_padding + 1
```

Three things the page makes visible that the formula hides:

- **`stride` is the upsampling factor**, because it is the spacing between stamps —
  not a subsampling factor as in `Conv2d`. Every grid shares one cell size, so you
  see the data physically grow. Flip to the Conv2d tab with the same `stride` to see
  the same number shrink it instead.
- **Checkerboard artifacts** are structural. When `kernel_size % stride != 0` the
  stamps overlap unevenly, so some output pixels sum more terms than their
  neighbours in a repeating pattern. Tick *colour by number of terms* to see it —
  that repaints the background only, the numbers in the cells stay the values.
  `k = 4, s = 2` is even; `k = 3, s = 2` is not.
- **`padding` is a crop.** Increasing it shrinks the output, which is the exact
  opposite of `Conv2d`.

## Normalisation

Every normalisation layer runs the same arithmetic:

```
y = γ · (x − μ) / √(σ² + ε) + β
```

The only thing that separates BatchNorm, LayerNorm, InstanceNorm and GroupNorm is
**which numbers get pooled into one `μ`**. The page draws the tensor as an
`N × C` grid of `H × W` maps, twice — input and output — and hovering a cell greys
out everything outside its pooling group:

| | one statistic per | pooled over | batch-dependent |
| --- | --- | --- | --- |
| `BatchNorm2d` | channel | `N·H·W` | yes |
| `LayerNorm` | sample | `C·H·W` | no |
| `InstanceNorm2d` | sample × channel | `H·W` | no |
| `GroupNorm(G)` | sample × group | `(C/G)·H·W` | no |

`GroupNorm(1, C)` is LayerNorm and `GroupNorm(C, C)` is InstanceNorm — switch
between them and watch the highlighted group grow or shrink.

The strip underneath plots every value on one axis, one row per group, before and
after. That is the whole operation in one picture: rows sitting at different
centres with different spreads on the left, all stacked on zero with the same
spread on the right.

Three things worth playing with:

- **`train()` vs `eval()`.** Only BatchNorm has the split, because only BatchNorm's
  answer depends on the other samples in the batch. `feed another batch` runs
  `running = (1−momentum)·running + momentum·batch` a step at a time. Switch to
  `eval()` before feeding anything and the output is the input unchanged — the
  buffers still hold their `0` and `1` initialisation.
- **Batch size.** The `each pools` figure in the shape panel is the number that
  matters: at `N=1, H=W=1` PyTorch refuses to run at all, and the page says so.
- **`γ`, `β`.** Set them to `learned` and the output stops being zero-mean —
  the layer can undo its own normalisation, which is why adding it costs nothing
  in expressiveness.

## Verifying

Plain-Python references sit next to the page. All run without dependencies; all
additionally compare against PyTorch if `torch` is installed.

```
python3 verify_conv2d.py
python3 verify_convtranspose2d.py
python3 verify_batchnorm.py
```

- `verify_conv2d.py` — covers **both** convolution tabs over 1086 parameter
  combinations: the output-size formulas against a brute-force count of legal
  window positions, the sums against a naive reference convolution, the per-axis
  read/write tallies, and — the one that matters for the hovering — that the
  *scatter* and *gather* code paths describe the identical set of
  `(input pixel, weight, output cell)` triples. If hovering an input and hovering
  an output ever disagreed, this test would fail.
- `verify_convtranspose2d.py` — the stamping picture, checked against the shape
  formula over 714 parameter combinations and against `F.conv_transpose2d`.
- `verify_batchnorm.py` — the four pooling patterns and the running-statistic
  update rule, over 297 combinations; 258 compared against `nn.BatchNorm2d`,
  `nn.LayerNorm`, `nn.InstanceNorm2d` and `nn.GroupNorm` in float64 (the rest are
  shapes PyTorch refuses to run, such as one value per channel). It also confirms
  the biased/unbiased split: normalisation divides by `n`, `running_var` by `n−1`.

## Ideas for next tabs

- pooling vs strided conv, side by side
- receptive-field growth across a stack of layers
- attention: `(B, T, C)` reshapes through multi-head projection
- broadcasting and `einsum` index bookkeeping
- a whole-network shape tracker: stack layers and watch the tensor change size

Issues and pull requests welcome — especially "this explanation is wrong" ones.

## Licence

Code — `index.html` and the `verify_*.py` references — is **MIT**, see
[`LICENSE`](LICENSE).

The page also embeds two datasets as base64 so it runs offline, and they carry
their own terms, listed in [`THIRD-PARTY.md`](THIRD-PARTY.md):

- **MNIST digits** — 4 samples from the official `t10k` test set
  (Yann LeCun & Corinna Cortes), public domain.
- **Default photo** — Yann LeCun, by Jérémy Barande / Institut Polytechnique de Paris,
  [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0), via
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Laura_Chaubard_%26_Yann_Le_Cun_-_2024_(53814052697)_(cropped).jpg).
  Cropped and converted to greyscale. **That one asset stays CC BY-SA 2.0, not MIT** —
  share-alike, so keep the attribution if you redistribute it. Attribution is
  displayed in the page. Swap the `PHOTO_B64` constant for a public-domain image if
  you want a fork under a single licence.
