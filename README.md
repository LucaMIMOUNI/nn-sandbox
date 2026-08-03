# NN sandbox

**[→ lucamimouni.github.io/nn-sandbox](https://lucamimouni.github.io/nn-sandbox/)**

Interactive visualisations of what a layer actually does to your tensors. Move a
slider, hover a number, and the page works out that number in front of you.

| Tab | |
| --- | --- |
| **Conv2d** | `kernel_size`, `stride`, `padding`, `dilation` |
| **ConvTranspose2d** | the same, plus `output_padding` |
| **Normalisation** | BatchNorm, LayerNorm, InstanceNorm, GroupNorm, and the covariate shift they answer |
| **Backprop** | a neuron fitting itself to points, then the same neuron many times over as an MLP — with a validation split, so overfitting shows up as two curves coming apart |

Feed it random integers you can check by hand, an MNIST digit, or your own photo.
Presets load the parameters of real networks — LeNet, AlexNet, VGG, ResNet,
MobileNet, DeepLab, DCGAN, U-Net, FSRCNN.

No dependencies, no build step, no network, no tracking. It is one HTML file:
save it and it still works on a plane.

## Hacking on it

`index.html` is generated from `src/` and committed, so you only rebuild if you
changed a source file.

```
src/style.css    src/page.html     the look and the markup
src/assets.js    the bundled MNIST + photo, base64
src/conv.js      Conv2d / ConvTranspose2d
src/norm.js      the normalisation layers
src/backprop.js  the autograd engine, the chain rule, and the MLP
src/tabs.js      the menu

python3 build.py
```

## Verifying

Plain Python, no dependencies, and they compare against PyTorch too if it is
installed.

```
python3 verify/conv2d.py            # both convolution tabs, 1086 combinations
python3 verify/convtranspose2d.py
python3 verify/batchnorm.py
python3 verify/backprop.py          # the autograd engine, 1331 configurations
python3 verify/mlp.py               # the same engine over layers, the delta recursion,
                                    # and where the validation loss turns back up
```

`verify/conv2d.py` also checks that hovering an input and hovering an output agree
on the identical set of `(input pixel, weight, output cell)` triples — if the two
directions ever disagreed, the page would be teaching something false, and the test
fails.

## Licence

Code is [MIT](LICENSE). The bundled MNIST samples are public domain; the default
photo is CC BY-SA 2.0 and stays that way — details in
[THIRD-PARTY.md](THIRD-PARTY.md).

Issues and pull requests welcome, especially "this explanation is wrong" ones.
