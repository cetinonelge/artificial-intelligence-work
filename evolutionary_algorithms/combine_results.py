#!/usr/bin/env python3
"""
combine_results.py

Traverse the directory structure:

    results/<parameter>/<value>/

and for each `value` folder containing exactly the files
`best_gen_01000.png` … `best_gen_10000.png`, produce a
2×5 montage saved as `montage_<parameter>_<value>.png` in the same folder.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Generation checkpoints
GENS = list(range(1000, 10001, 1000))

# ------------------------------------------------------------
# Combine best_gen images in a single value directory
# ------------------------------------------------------------
def combine_one(val_dir: Path):
    # Check required files exist
    img_paths = [val_dir / f"best_gen_{gen:05d}.png" for gen in GENS]
    if not all(p.exists() for p in img_paths):
        print(f"[skip] Missing images in {val_dir}")
        return

    # Create 2x5 montage
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, gen, path in zip(axes.flat, GENS, img_paths):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"Gen {gen}")
        ax.axis('off')

    # Super-title with parameter and value
    param = val_dir.parent.name
    value = val_dir.name
    fig.suptitle(f"{param} = {value}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save montage
    out_file = val_dir / f"montage_{param}_{value}.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_file}")

# ------------------------------------------------------------
# Main traversal
# ------------------------------------------------------------
def main():
    root = Path("results")
    if not root.exists() or not root.is_dir():
        print("Error: 'results/' directory not found.", file=sys.stderr)
        sys.exit(1)

    # Loop over parameter directories
    for param_dir in sorted(root.iterdir()):
        if not param_dir.is_dir():
            continue
        # Loop over value subdirectories
        for val_dir in sorted(param_dir.iterdir(), key=lambda d: d.name):
            if not val_dir.is_dir():
                continue
            combine_one(val_dir)

if __name__ == "__main__":
    main()
