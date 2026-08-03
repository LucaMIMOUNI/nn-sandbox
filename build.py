#!/usr/bin/env python3
"""Build index.html from src/.

The page ships as one self-contained file — no CDN, no build step to *use* it,
works offline. That file is generated, so edit src/ and run this:

    python3 build.py

index.html is committed, so you only need to run this if you changed src/.
"""

import pathlib

ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "src"
OUT = ROOT / "index.html"

TITLE = "NN Sandbox — see what each layer actually does"


def part(name):
    return (SRC / name).read_text().strip("\n")


page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
{part("style.css")}
</style>
</head>
<body>

{part("page.html")}

<script>
"use strict";

{part("assets.js")}

{part("conv.js")}

{part("norm.js")}

{part("backprop.js")}

{part("tabs.js")}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    OUT.write_text(page)
    print(f"{OUT.name} — {len(page):,} bytes")
