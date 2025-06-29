#!/usr/bin/env python3
"""
combine_fitness.py  –  side‑by‑side version

Creates a 1×2 figure (full‑run plot on the left, zoom plot on the right)
for every experiment directory under `results/`.

Output:
    results/<param>/<value>/fitness_combined_<param>_<value>.png
"""

from pathlib import Path
from matplotlib.gridspec import GridSpec
import sys
from PIL import Image
import matplotlib.pyplot as plt

def combine_one(run_dir: Path):
    full  = run_dir / "fitness_full.png"
    zoom  = run_dir / "fitness_zoom.png"
    param = run_dir.parent.name
    value = run_dir.name
    out   = run_dir / f"fitness_combined_{param}_{value}.png"

    """
    if out.exists():
        print(f"[skip] {out.name} already exists")
        return
    if not (full.exists() and zoom.exists()):
        print(f"[skip] Missing fitness plots in {run_dir}")
        return
    """
    # ---------- layout tweaks ---------------------------------
    fig = plt.figure(figsize=(13, 4.5))        # wider & a bit taller
    gs  = GridSpec(1, 2, wspace=0.015,         # tiny gap between plots
                   left=0.02, right=0.98,      # thin side margins
                   top=0.88, bottom=0.05)      # pull title closer

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.imshow(Image.open(full));  ax1.axis("off")
    ax2.imshow(Image.open(zoom));  ax2.axis("off")

    fig.suptitle(f"{param} = {value}", fontsize=18, y=0.93)

    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    root = Path("results")
    if not root.exists():
        print("Error: 'results/' directory not found.", file=sys.stderr)
        sys.exit(1)

    for param_dir in sorted(root.iterdir()):
        if not param_dir.is_dir():
            continue
        for val_dir in sorted(param_dir.iterdir(), key=lambda d: d.name):
            if val_dir.is_dir():
                combine_one(val_dir)

if __name__ == "__main__":
    main()
