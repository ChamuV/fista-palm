# FISTA–PALM

## Accelerated Block-Coordinate Optimisation for Nonconvex Problems

This repository contains the implementation and experimental study of **FISTA–PALM**, a hybrid optimisation algorithm that combines the block-coordinate structure of PALM with FISTA-style acceleration for structured nonconvex problems.

The work was developed as part of an MSc project and investigates how acceleration techniques can be integrated into proximal alternating methods.

The project focuses on:

- algorithmic design
- evaluation across a range of problem settings


## 1. Problem Setting

We consider optimisation problems of the form:

$$
\min_{x,y} \; f(x) + g(y) + H(x,y)
$$

where:

- $f, g$ are nonsmooth but proximable  
- $H$ is smooth and couples the variables  

This formulation appears in:

- sparse matrix factorisation  
- image reconstruction and denoising  
- representation learning  

Two models studied in this repository are:

### Two-block factorisation

$$
A \approx X Y^T
$$

### Three-block factorisation

$$
A \approx X B Y^T
$$


## 2. Algorithms

The repository implements:

- **PALM** — Proximal Alternating Linearised Minimisation  
- **iPALM** — Inertial extension of PALM  
- **FISTA–PALM** — accelerated variant introduced in this project  


## 3. FISTA–PALM Algorithm

FISTA–PALM combines:

- the block-coordinate structure of PALM  
- Nesterov acceleration from FISTA  

The key idea is to perform updates at an extrapolated point rather than the current iterate.

### Algorithm Outline

Given initial variables $x^0, y^0$, set $t_0 = 1$. For $k = 0,1,\dots$:

1. **Compute extrapolation factor**

$$
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}
$$

2. **Extrapolate variables**

$$
\tilde{x}_k = x_k + \frac{t_k - 1}{t_{k+1}} (x_k - x_{k-1})
$$

$$
\tilde{y}_k = y_k + \frac{t_k - 1}{t_{k+1}} (y_k - y_{k-1})
$$

3. **Update first block**

$$
x_{k+1} =
\mathrm{prox}^{f}_{\eta_k}
\left(
\tilde{x}_k - \frac{1}{\eta_k} \nabla_x H(\tilde{x}_k, y_k)
\right)
$$

4. **Update second block**

$$
y_{k+1} =
\mathrm{prox}^{g}_{\tau_k}
\left(
\tilde{y}_k - \frac{1}{\tau_k} \nabla_y H(x_{k+1}, \tilde{y}_k)
\right)
$$

### Interpretation

- PALM performs alternating proximal gradient steps  
- FISTA–PALM introduces Nestorov-like acceleration across iterations  
- This leads to:
  - faster objective decrease  
  - improved practical performance  
  - minimal additional tuning  


## 4. Repository Overview

```bash
src/
├── problems/    # optimisation problem definitions
├── solvers/     # PALM, iPALM, FISTA–PALM implementations
├── utils/       # data loading, plotting, and experiment utilities
└── cli/         # command-line interface

notebooks/       # experiments
tests/           # validation and correctness checks
```

The `notebooks/` directory contains the experiments, while `src/` provides reusable implementations of the optimisation algorithms.



## 5. Experiments

The repository contains a structured experimental study covering:

- Synthetic sparse matrix factorisation  
- Initialisation strategies (random vs SVD)  
- Time vs objective scaling  
- Conditioning and singular value effects  
- ORL face reconstruction  
- BSDS500 image denoising  
- COIL-20 three-block sparse factorisation  

These experiments evaluate:

- convergence speed  
- stability under different regimes  
- reconstruction quality in imaging tasks  


## 6. Key Findings

Across experiments, the following patterns are observed:

- FISTA–PALM achieves faster objective reduction than PALM, particularly in early iterations  
- Acceleration remains effective in both synthetic and real-data settings  
- Multi-block problems (e.g. three-block factorisation) are significantly more challenging for standard PALM  
- FISTA–PALM produces improved reconstructions under fixed iteration budgets  
- Acceleration may introduce mild instability near convergence, consistent with known behaviour of inertial methods  


## 7. Usage

### Installation

```bash
pip install -r requirements.txt
```

### Run FISTA-PALM from CLI

```bash
python -m src.cli --m 100 --n 80 --rank 10 --plot
```

## 8. References

Bolte, Sabach, Teboulle (2014)
Proximal alternating linearized minimization for nonconvex and nonsmooth problems
https://doi.org/10.1007/s10107-013-0701-9

Pock, Sabach (2016)
Inertial Proximal Alternating Linearized Minimization (iPALM) for Nonconvex and Nonsmooth Problems
https://doi.org/10.1137/16M1064064

Beck, Teboulle (2009)
A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
https://doi.org/10.1137/080716542


## License

This project is licensed under the MIT License. See the LICENSE file for details.