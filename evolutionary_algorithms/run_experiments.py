############################################################
# EE449 – HW 3  Evolutionary Art  •  experiment driver      #
############################################################
# Adds a --start VALUE flag so you can resume an unfinished
# sweep, e.g.                                              
#   python run_experiments.py --param frac_parents --start 0.4
# Will iterate over 0.4, 0.6, 0.8 (skipping 0.2).          #
# It also skips a value automatically if the folder already
# contains a complete fitness.npy with NUM_GENERATIONS      
# entries – so re‑runs are idempotent.                      #
############################################################

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evolutionary_art import Population

# ---------------------------------------------------------
# Default hyper‑parameters (bold values in Table 1)
# ---------------------------------------------------------
BASE_PARAMS: Dict[str, float] = {
    "num_inds": 20,
    "num_genes": 50,
    "tm_size": 5,
    "frac_elites": 0.20,
    "frac_parents": 0.40,
    "mutation_prob": 0.20,
    "guided": True,   # True = guided, False = unguided
}

SWEEP_VALUES: Dict[str, List] = {
    "num_inds":      [5, 10, 20, 50, 75],
    "num_genes":     [10, 25, 50, 100, 150],
    "tm_size":       [2, 5, 10, 20],
    "frac_elites":   [0.05, 0.20, 0.40],
    "frac_parents":  [0.20, 0.40, 0.60, 0.80],
    "mutation_prob": [0.10, 0.20, 0.50, 0.80],
    "guided":        [True, False],
}

NUM_GENERATIONS = 10_000
SNAPSHOT_STEP   = 1_000  # save best individual every 1000 gens

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_fitness(history: List[float], out_full: Path, out_zoom: Path):
    gens = np.arange(1, len(history) + 1)
    # full run
    plt.figure(); plt.plot(gens, history)
    plt.xlabel("Generation"); plt.ylabel("Best SSIM"); plt.title("Fitness – full run")
    plt.tight_layout(); plt.savefig(out_full, dpi=300); plt.close()
    # zoom
    plt.figure(); mask = gens >= 1000
    plt.plot(gens[mask], np.array(history)[mask])
    plt.xlabel("Generation"); plt.ylabel("Best SSIM"); plt.title("Fitness – generations 1000‑10000")
    plt.tight_layout(); plt.savefig(out_zoom, dpi=300); plt.close()

# ---------------------------------------------------------
# Core experiment
# ---------------------------------------------------------

def result_complete(folder: Path) -> bool:
    """Return True if fitness.npy exists and has NUM_GENERATIONS entries."""
    f = folder / "fitness.npy"
    return f.exists() and np.load(f).shape[0] == NUM_GENERATIONS


def run_experiment(param_name: str, value):
    target = cv2.imread("painting.png", cv2.IMREAD_UNCHANGED)
    if target is None:
        raise FileNotFoundError("painting.png not found")

    out_dir = Path("results") / param_name / str(value)
    if result_complete(out_dir):
        print(f"[skip] {param_name}={value} already done")
        return
    ensure_dir(out_dir)

    p = BASE_PARAMS.copy(); p[param_name] = value
    pop = Population(p["num_inds"], p["num_genes"], target.shape)

    history: List[float] = []
    for chunk in tqdm(range(NUM_GENERATIONS // SNAPSHOT_STEP), desc=f"{param_name}={value}"):
        history.extend(pop.evolve(target, SNAPSHOT_STEP, p["tm_size"], p["frac_elites"],
                                  p["frac_parents"], p["mutation_prob"], p["guided"]))
        pop.evaluate(target)
        best = pop.individuals[0].draw(target.shape)
        cv2.imwrite(str(out_dir / f"best_gen_{(chunk+1)*SNAPSHOT_STEP:05d}.png"), best)

    plot_fitness(history, out_dir / "fitness_full.png", out_dir / "fitness_zoom.png")
    np.save(out_dir / "fitness.npy", np.array(history))

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EE449 HW3 parameter sweeps – resume capable")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--param", choices=SWEEP_VALUES.keys(), help="Hyper‑parameter to sweep")
    grp.add_argument("--all", action="store_true", help="Run every sweep sequentially")
    parser.add_argument("--start", type=str, help="Value to start/resume from (inclusive)")
    args = parser.parse_args()

    def should_run(val):
        if args.start is None:
            return True
        # convert both to string for generality (works for True/False too)
        return str(val) >= args.start

    if args.all:
        for pname, vals in SWEEP_VALUES.items():
            for v in vals:
                if should_run(v):
                    run_experiment(pname, v)
    else:
        for v in SWEEP_VALUES[args.param]:
            if should_run(v):
                run_experiment(args.param, v)

############################################################
# End of file                                              #
############################################################
