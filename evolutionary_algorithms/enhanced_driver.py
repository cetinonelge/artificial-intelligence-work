#!/usr/bin/env python3
"""
Enhanced EA driver for EE‑449 HW‑3
=================================
Implements three structural improvements **and** uses the *best* hyper‑parameters
identified in our earlier sweeps:
    num_inds       = 75
    num_genes      = 100
    tm_size        = 5
    frac_elites    = 0.05
    frac_parents   = 0.60
    mutation_prob  = 0.20
    guided         = True
The script mirrors the homework artefacts, saving 10 snapshots, two fitness
plots, the best image, and the fitness history under `results/enhanced/`.
Runtime: ≈15 min on a quad‑core CPU (tqdm progress bar shown).
"""

from pathlib import Path
from copy import deepcopy
import math, random, sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evolutionary_art import Population, Individual

# -----------------------------------------------------------
# Best hyper‑parameter set (derived from previous experiments)
# -----------------------------------------------------------
BEST_PARAMS = {
    "num_inds": 75,
    "num_genes": 100,
    "tm_size": 5,
    "frac_elites": 0.05,
    "frac_parents": 0.60,
    "mutation_prob": 0.20,
    "guided": True,
}

GEN_CHECKS = list(range(1000, 10001, 1000))  # snapshot generations

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_history(hist: np.ndarray, out_full: Path, out_zoom: Path):
    gens = np.arange(1, len(hist) + 1)
    plt.figure(); plt.plot(gens, hist); plt.xlabel("Generation"); plt.ylabel("Best SSIM")
    plt.title("Fitness – full run"); plt.tight_layout(); plt.savefig(out_full, dpi=300); plt.close()
    plt.figure(); mask = gens >= 1000; plt.plot(gens[mask], hist[mask])
    plt.xlabel("Generation"); plt.ylabel("Best SSIM"); plt.title("Fitness – generations 1000‑10000")
    plt.tight_layout(); plt.savefig(out_zoom, dpi=300); plt.close()

# -----------------------------------------------------------
# 1) Self‑adaptive individual
# -----------------------------------------------------------
class SAIndividual(Individual):
    """Individual with log‑normal self‑adaptive step‑size σ."""
    def __init__(self, genes, sigma: float = 0.5):
        super().__init__(genes)
        self.sigma = sigma

    def copy(self):
        return SAIndividual([g.copy() for g in self.genes], self.sigma)

    def crossover(self, other):
        c1, c2 = super().crossover(other)
        mean_sigma = (self.sigma + other.sigma) / 2
        return SAIndividual(c1.genes, mean_sigma), SAIndividual(c2.genes, mean_sigma)

    def mutate(self, prob, guided, img_shape):
        # adapt σ
        self.sigma = float(np.clip(self.sigma * math.exp(np.random.randn() / math.sqrt(len(self.genes))), 0.01, 1.0))
        if random.random() < prob:
            h, w, _ = img_shape
            g = random.choice(self.genes)
            dx, dy = int(self.sigma * w), int(self.sigma * h)
            for v in g.vertices:
                v[:] = np.clip(v + np.random.randint(-dx, dx + 1, 2), [0, 0], [w - 1, h - 1])
            r, g_c, b, a = g.color
            jitter = lambda c: int(np.clip(c + self.sigma * np.random.randint(-128, 128), 0, 255))
            g.color = (jitter(r), jitter(g_c), jitter(b), float(np.clip(a + self.sigma * np.random.randn(), 0, 1)))
            self.invalidate()

# -----------------------------------------------------------
# 2) Island model EA
# -----------------------------------------------------------
class IslandEA:
    def __init__(self, n_islands: int, pop_size: int, num_genes: int, img_shape):
        self.islands = [Population(pop_size, num_genes, img_shape) for _ in range(n_islands)]
        for isl in self.islands:
            isl.individuals = [SAIndividual(ind.genes) for ind in isl.individuals]
        self.shape = img_shape

    def evolve(self, target, params, generations: int, migrate_every: int, migrants: int, out_dir: Path):
        history = []
        for gen in tqdm(range(generations), desc="Enhanced EA"):
            for isl in self.islands:
                isl.evolve(target, 1, params["tm_size"], params["frac_elites"], params["frac_parents"],
                           params["mutation_prob"], params["guided"])
                isl.evaluate(target)
            # record global best
            best_island = max(self.islands, key=lambda i: i.individuals[0]._fitness)
            history.append(best_island.individuals[0]._fitness)
            # snapshot every 1000 gens
            if (gen + 1) in GEN_CHECKS:
                img = best_island.individuals[0].draw(self.shape)
                cv2.imwrite(str(out_dir / f"best_gen_{gen+1:05d}.png"), img)
            # ring migration
            if (gen + 1) % migrate_every == 0:
                for i, isl in enumerate(self.islands):
                    src = isl.individuals[:migrants]
                    dest = self.islands[(i + 1) % len(self.islands)].individuals
                    dest[-migrants:] = [deepcopy(s) for s in src]
        return np.array(history)

# -----------------------------------------------------------
# 3) Lamarckian micro‑search
# -----------------------------------------------------------

def local_search(ind: SAIndividual, target: np.ndarray, steps: int = 8, lr: int = 1):
    h, w, _ = target.shape
    for _ in range(steps):
        img = ind.draw(target.shape)
        err = (target[:, :, :3].astype(int) - img[:, :, :3].astype(int)).mean(2)
        for g in ind.genes:
            for v in g.vertices:
                v[0] = int(np.clip(v[0] + lr * np.sign(err[v[1], v[0]]), 0, w - 1))
                v[1] = int(np.clip(v[1] + lr * np.sign(err[v[1], v[0]]), 0, h - 1))
        ind.invalidate()
    return ind

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    target = cv2.imread("painting.png", cv2.IMREAD_UNCHANGED)
    if target is None:
        print("painting.png not found", file=sys.stderr); sys.exit(1)

    out_dir = Path("results") / "enhanced"; ensure_dir(out_dir)

    # split 75 individuals into 4 roughly equal islands
    ea = IslandEA(4, pop_size=BEST_PARAMS["num_inds"] // 4 + 1,  # 19‑19‑19‑18
                  num_genes=BEST_PARAMS["num_genes"], img_shape=target.shape)

    history = ea.evolve(target, BEST_PARAMS, generations=10_000,
                        migrate_every=500, migrants=2, out_dir=out_dir)

    # Lamarckian tweak every 1000 gens
    for gen in GEN_CHECKS:
        best_island = max(ea.islands, key=lambda i: i.individuals[0]._fitness)
        best_island.individuals[0] = local_search(best_island.individuals[0], target)
        best_island.evaluate(target)
        img = best_island.individuals[0].draw(target.shape)
        cv2.imwrite(str(out_dir / f"best_gen_{gen:05d}.png"), img)
        history[gen - 1] = best_island.individuals[0]._fitness

    # final best
        # final best and artefacts
    for isl in ea.islands:
        isl.evaluate(target)
    best_global = max(ea.islands, key=lambda i: i.individuals[0]._fitness).individuals[0]
    best_path = out_dir / "enhanced_best.png"
    cv2.imwrite(str(best_path), best_global.draw(target.shape))

    # save history and plots
    np.save(out_dir / "enhanced_history.npy", history)
    plot_history(history, out_dir / "fitness_full.png", out_dir / "fitness_zoom.png")

    print("Final SSIM:", best_global._fitness)
    print("All artefacts saved to", out_dir)

if __name__ == "__main__":
    main()
