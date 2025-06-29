# EE 449 – Computational Intelligence &mdash; Homeworks 1-3

Welcome!  
This repository collects my solutions to the three major assignments of METU **EE 449 – Computational Intelligence** (Spring 2025):

| HW&nbsp;# | Theme | Core Techniques |
|-----------|-------|-----------------|
| **1** | *Training Artificial Neural Networks* | NumPy MLP from scratch · NumPy Conv2D · PyTorch MLP/CNN experiments |
| **2** | *Reinforcement Learning* | Tabular TD(0) in a stochastic maze · Deep Q-Network for LunarLander-v3 |
| **3** | *Evolutionary Algorithms* | μ + λ evolutionary art engine that rebuilds a target image with opaque triangles |

## 1 Training Artificial Neural Networks  

### What I built  
* **Part 1 – NumPy MLP**  
  * Derived and coded the forward/back-prop of a **1-hidden-layer MLP** (supports Sigmoid / Tanh / ReLU).  
  * Solved the XOR toy problem and plotted decision boundaries.

* **Part 2 – NumPy Conv2D**  
  * Implemented a minimal `my_conv2d` (valid padding, stride = 1) and visualised feature maps on MNIST snippets.

* **Part 3 – PyTorch architecture sweep**  
  * Evaluated two fully-connected and three convolutional configurations on a *real* image task.  
  * Logged loss/accuracy, visualised first–layer weights, and discussed depth/width trade-offs.

* **Part 4 – Activation-function study**  
  * Compared gradient flow and convergence of ReLU vs. Sigmoid across all five nets.

* **Part 5 – Learning-rate study**  
  * Bench-marked SGD with {0.1, 0.01, 0.001} and designed a simple stepped LR-schedule; contrasted with Adam.

### Dataset  
All classification experiments (Parts 3-5) use the **CIFAR-10** dataset: 60 000 RGB images (32 × 32) across 10 classes.  
The dataset is **not stored in this repo**; it is fetched automatically by the code via `torchvision.datasets.CIFAR10`, which will download to `./data/` on first run. :contentReference[oaicite:0]{index=0}

## 2 Reinforcement Learning  

### Maze with TD(0)  
* Coded a **stochastic 11 × 11 maze** (75 % intended, 5 % opposite, 2 × 10 % perpendicular slips).  
* Implemented tabular TD(0) with ε-greedy exploration and produced heat-maps, policy arrows and convergence plots for sweeps over α, γ, ε.

### LunarLander-v3 with DQN  
* Wrote a lightweight replay buffer & target-network DQN in PyTorch.  
* Hyper-parameter grid: learning-rate, discount, ε-decay, target-update frequency, and five network footprints.  
* Each run logs raw episode rewards, 100-ep moving averages and the “first solved episode”.

## 3 Evolutionary Art  

### Engine  
* Individuals are lists of coloured triangles; fitness is **SSIM** to a target painting.  
* Supports tournament selection, uniform crossover, guided/unguided mutation, elitism and parental fractions.

### Experiments  
* Conducted systematic sweeps over population size, genome length, tournament size, elitism %, parent ratio, mutation prob/type.  
* For every sweep stored best-of-generation images every 1 000 generations and dual-scale fitness plots.

### Algorithmic Enhancements  
Implemented an “island-model + self-adaptive mutation + Lamarckian refinement” variant that pushed SSIM above 0.91.
