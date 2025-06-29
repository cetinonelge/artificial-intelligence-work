# ── generate_all_plots.py ──────────────────────────────────────────────
import re, os, glob, json, utils

# Where all your results_*.json live
RESULTS_DIR = os.path.expanduser("~/EE449/HW2")

# Parameter names exactly as you used in the filenames
PARAMS = ["lr", "gamma", "epsilon_decay",
          "target_update_freq", "hidden_dims"]

# ───────────────────────────────────────────────────────────────────────
def decode_value(raw: str) -> str:
    """
    Convert the safe filename encoding back to a readable label.
      0p001    → 1e-3
      0p0001   → 1e-4
      128x128  → [128,128]
      256      → 256
    """
    # lists (hidden_dims): 128x128 → [128,128]
    if "x" in raw and raw[0].isdigit():
        return "[" + raw.replace("x", ",") + "]"

    # floats: 0p001 → 0.001
    if "p" in raw:
        s = raw.replace("p", ".")
        # optional: express small numbers in scientific notation
        try:
            as_float = float(s)
            if 0 < abs(as_float) < 1e-2:
                return f"{as_float:.0e}"      # 0.0001 → 1e-4
            return str(as_float)
        except ValueError:
            pass
    return raw  # fallback


# ───────────────────────────────────────────────────────────────────────
def collect_jsons(param):
    patt = os.path.join(RESULTS_DIR, f"results_{param}_*.json")
    return sorted(glob.glob(patt))


def make_pretty_plots(param):
    paths = collect_jsons(param)
    if not paths:
        print(f"⚠  No JSON files found for “{param}”; skipping.")
        return

    # Build clean labels from the filename chunk after param_ and before timestamp
    labels = []
    rgx = re.compile(rf"results_{param}_(.+?)_\d{{8}}T\d{{6}}Z\.json")
    for p in paths:
        m = rgx.search(os.path.basename(p))
        labels.append(decode_value(m.group(1) if m else p))

    # Learning curves
    utils.plot_learning_curves(
        paths,
        labels=labels,
        output_file=f"{param}_learning_curves_pretty.png",
    )

    # Solved-episode bars
    utils.plot_solved_episodes(
        paths,
        labels=labels,
        output_file=f"{param}_solved_episodes_pretty.png",
    )
    print(f"✓  Plots for “{param}” saved.")


# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for p in PARAMS:
        make_pretty_plots(p)
# ───────────────────────────────────────────────────────────────────────
