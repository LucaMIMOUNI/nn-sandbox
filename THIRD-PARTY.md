# Third-party assets

The MIT licence in [`LICENSE`](LICENSE) covers the code — `index.html` and the
`verify_*.py` references.

`index.html` also **embeds two datasets as base64**, so that the page works with no
network and no build step. They are not mine, and they carry their own terms.

## MNIST digits — public domain

Four samples from the official MNIST `t10k` test set, by Yann LeCun and Corinna
Cortes. Stored as raw 8-bit greyscale bytes.

MNIST is distributed as a public-domain dataset. No attribution is required; it is
given in the page anyway.

## Default photo — CC BY-SA 2.0 ⚠️ share-alike

*Laura Chaubard & Yann Le Cun (2024)*, by **Jérémy Barande / Institut Polytechnique
de Paris**, licensed
[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0), via
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Laura_Chaubard_%26_Yann_Le_Cun_-_2024_(53814052697)_(cropped).jpg).

The copy bundled here is an **adaptation**: cropped to a square and converted to
112×112 greyscale.

**That adaptation remains under CC BY-SA 2.0 — it is not MIT.** CC BY-SA is a
copyleft licence, so if you redistribute the image, or a work derived from it, you
must keep it under CC BY-SA 2.0 (or a compatible licence) and keep the attribution.
Attribution is displayed in the page itself, under the Photo tab.

### What this means if you reuse this project

- **Reusing the code** — the layout, the maths, the rendering: MIT, do as you like.
- **Redistributing `index.html` as-is** — you are also redistributing that photo, so
  keep the attribution and honour CC BY-SA 2.0 for it.
- **Want a clean single-licence fork?** Replace the `PHOTO_B64` constant in
  `index.html` with a public-domain or CC0 image, or with an empty string — the
  upload/drag-and-drop path works regardless, and nothing else depends on it.

If you are the rights holder for any asset here and would like it changed or
removed, please open an issue.
