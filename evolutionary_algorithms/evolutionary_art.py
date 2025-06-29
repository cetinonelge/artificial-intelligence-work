import random
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

"""
EE 449 – Homework 3  ·  Evolutionary Art engine
=================================================
Fully self‑contained evolutionary algorithm implementation.
Public API compatible with `run_experiments.py`.
"""

# ==================================================================
# Constants
# ==================================================================
C1 = 6.5025
C2 = 58.5225

# ==================================================================
# Gene  (one triangle)
# ==================================================================
@dataclass
class Gene:
    vertices: np.ndarray                      # (3,2) int32
    color: Tuple[int, int, int, float]        # R,G,B,A  (A∈[0,1])

    def area(self) -> float:
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    def mutate(self, guided: bool, img_w: int, img_h: int):
        (self._mutate_guided if guided else self._mutate_unguided)(img_w, img_h)

    def _mutate_guided(self, img_w: int, img_h: int):
        dx, dy = img_w // 4, img_h // 4
        for i in range(3):
            self.vertices[i, 0] = np.clip(self.vertices[i, 0] + random.randint(-dx, dx), 0, img_w - 1)
            self.vertices[i, 1] = np.clip(self.vertices[i, 1] + random.randint(-dy, dy), 0, img_h - 1)
        r, g, b, a = self.color
        self.color = (
            int(np.clip(r + random.randint(-64, 64), 0, 255)),
            int(np.clip(g + random.randint(-64, 64), 0, 255)),
            int(np.clip(b + random.randint(-64, 64), 0, 255)),
            float(np.clip(a + random.uniform(-0.25, 0.25), 0.0, 1.0)),
        )

    def _mutate_unguided(self, img_w: int, img_h: int):
        self.vertices = np.column_stack((np.random.randint(0, img_w, 3), np.random.randint(0, img_h, 3))).astype(np.int32)
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.random(),
        )

    def copy(self) -> "Gene":
        return Gene(self.vertices.copy(), tuple(self.color))

# ==================================================================
# Individual  (chromosome)
# ==================================================================
@dataclass
class Individual:
    genes: List[Gene]
    _fitness: float = field(init=False, default=None, repr=False)

    # -------------- evaluation ------------------
    def fitness(self, target: np.ndarray) -> float:
        if self._fitness is None:
            self.genes.sort(key=lambda g: g.area(), reverse=True)
            self._fitness = ssim(self.draw(target.shape), target)
        return self._fitness

    def invalidate(self):
        self._fitness = None

    # -------------- operators -------------------
    def crossover(self, other: "Individual") -> Tuple["Individual", "Individual"]:
        c1, c2 = [], []
        for g1, g2 in zip(self.genes, other.genes):
            if random.random() < 0.5:
                c1.append(g1.copy()); c2.append(g2.copy())
            else:
                c1.append(g2.copy()); c2.append(g1.copy())
        return Individual(c1), Individual(c2)

    def mutate(self, prob: float, guided: bool, img_shape):
        if random.random() < prob:
            random.choice(self.genes).mutate(guided, img_shape[1], img_shape[0])
            self.invalidate()

    # -------------- rendering -------------------
    def draw(self, shape: Tuple[int, int, int]) -> np.ndarray:
        h, w, _ = shape
        img = np.ones((h, w, 4), dtype=np.uint8) * 255  # opaque white
        for gene in self.genes:
            r, g, b, a = gene.color
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [gene.vertices.astype(np.int32)], 255)
            layer = img.copy()
            cv2.fillPoly(layer, [gene.vertices.astype(np.int32)], (b, g, r, 255))
            alpha = a * (mask[:, :, None] / 255.0)
            img[:, :, :3] = (layer[:, :, :3] * alpha + img[:, :, :3] * (1 - alpha)).astype(np.uint8)
        return img

# ==================================================================
# Population  (μ + λ EA)
# ==================================================================
class Population:
    def __init__(self, num_inds: int, num_genes: int, img_shape):
        self.img_shape = img_shape
        self.individuals = [self._rand_ind(num_genes, img_shape) for _ in range(num_inds)]

    def evaluate(self, target):
        for ind in self.individuals:
            ind.fitness(target)
        self.individuals.sort(key=lambda i: i._fitness, reverse=True)

    def _tournament(self, k):
        return max(random.sample(self.individuals, k), key=lambda i: i._fitness)

    def evolve(self, target, gens, tm, fe, fp, mp, guided):
        history = []
        k_elite = max(1, int(fe * len(self.individuals)))
        for _ in range(gens):
            self.evaluate(target)
            history.append(self.individuals[0]._fitness)
            next_pop = self.individuals[:k_elite]
            parents = [self._tournament(tm) for _ in range(max(2, int(fp * len(self.individuals))))]
            random.shuffle(parents)
            for p1, p2 in zip(parents[::2], parents[1::2]):
                next_pop.extend(p1.crossover(p2))
            while len(next_pop) < len(self.individuals):
                next_pop.append(self._tournament(tm))
            for ind in next_pop[k_elite:]:
                ind.mutate(mp, guided, self.img_shape)
            self.individuals = next_pop
        return history

    @staticmethod
    def _rand_ind(num_genes, img_shape):
        h, w, _ = img_shape
        return Individual([
            Gene(
                np.column_stack((np.random.randint(0, w, 3), np.random.randint(0, h, 3))).astype(np.int32),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.random()),
            )
            for _ in range(num_genes)
        ])

# ==================================================================
# SSIM  (hand‑coded, channel‑wise)
# ==================================================================

def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    x = img1[:, :, :3].astype(np.float64)
    y = img2[:, :, :3].astype(np.float64)
    score = 0.0
    for k in range(3):
        xs, ys = x[:, :, k], y[:, :, k]
        mu_x, mu_y = xs.mean(), ys.mean()
        sig_x = ((xs - mu_x) ** 2).mean()
        sig_y = ((ys - mu_y) ** 2).mean()
        sig_xy = ((xs - mu_x) * (ys - mu_y)).mean()
        num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
        score += num / den
    return score / 3.0
