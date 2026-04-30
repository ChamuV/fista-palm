## ✨ Notes

This project was developed as a research-style exploration of modern optimisation methods, with an emphasis on **clarity, extensibility, and empirical insight** rather than black-box usage.
# FISTA–PALM

## Accelerated Block-Coordinate Optimisation for Nonconvex Problems

This repository contains the implementation and experimental study of the **FISTA–PALM algorithm**, a hybrid optimisation method developed as part of an MSc research project at the University of Oxford.

The project investigates how acceleration techniques interact with block-coordinate methods in nonconvex, nonsmooth optimisation settings, with a focus on both theoretical motivation and empirical performance.


## 1. Problem Setting

We consider optimisation problems of the form:

\[
\min_{x,y} \; f(x) + g(y) + H(x,y)
\]

where:
- \( f \) and \( g \) are possibly nonsmooth but proximable
- \( H \) is smooth and couples the variables

This structure naturally appears in applications such as:
- sparse matrix factorisation
- image reconstruction and denoising
- representation learning

Two key models studied in this repository are:

### Two-block factorisation
\[
A \approx X Y^T
\]

### Three-block factorisation
\[
A \approx X B Y^T
\]


## 2. Algorithms

The repository implements and compares the following optimisation methods:

- **PALM** (Proximal Alternating Linearised Minimisation)  
- **iPALM** (Inertial PALM)  
- **FISTA–PALM** (accelerated variant proposed in this project)

This work was carried out as part of an MSc project at the University of Oxford, with the aim of designing and evaluating an accelerated block-coordinate method for nonconvex optimisation.

### FISTA–PALM

FISTA–PALM combines:
- the block-coordinate structure of PALM
- Nesterov-style acceleration from FISTA

The goal is to retain the simplicity and modularity of PALM while improving convergence speed.

#### Algorithmic structure

Given variables $x$ and $y$, each iteration consists of:

1. **Extrapolation (inertial step)**

\[
\tilde{x}_k = x_k + \beta_k (x_k - x_{k-1}), \qquad
\tilde{y}_k = y_k + \beta_k (y_k - y_{k-1})
\]

where the acceleration parameter is defined via

\[
t_k = \frac{1 + \sqrt{1 + 4 t_{k-1}^2}}{2}, \qquad
\beta_k = \frac{t_{k-1} - 1}{t_k}
\]

2. **Block updates (proximal gradient steps)**

\[
x_{k+1} = \operatorname{prox}_{f/L_x}\big(\tilde{x}_k - \tfrac{1}{L_x} \nabla_x H(\tilde{x}_k, y_k)\big)
\]

\[
y_{k+1} = \operatorname{prox}_{g/L_y}\big(\tilde{y}_k - \tfrac{1}{L_y} \nabla_y H(x_{k+1}, \tilde{y}_k)\big)
\]

where $L_x$ and $L_y$ are Lipschitz constants for the partial gradients.

3. **Iteration**

The process is repeated until convergence, typically monitored via the objective value.

#### Extension to multi-block problems

The same idea extends naturally to three-block models of the form

\[
A \approx X B Y^T
\]

by applying extrapolation and proximal updates sequentially to each block $(X, B, Y)$.


## 3. Repository Overview

```
src/
  problems/     optimisation problem definitions
  solvers/      PALM, iPALM, FISTA–PALM implementations
  utils/        data loading, plotting, and IO utilities
  cli           command-line interface for running FISTA–PALM

notebooks/      experimental pipeline (10 structured experiments)

results/        generated figures and tables

tests/          validation tests
```


## 4. Experiments

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


## 5. Key Findings

Across experiments, the following patterns are observed:

- FISTA–PALM achieves faster objective reduction than PALM, particularly in early iterations
- Acceleration remains effective in both synthetic and real-data settings
- Multi-block problems (e.g. three-block factorisation) are significantly more challenging for standard PALM
- FISTA–PALM produces improved reconstructions under fixed iteration budgets
- Acceleration may introduce mild instability near convergence, consistent with known behaviour of inertial methods


## 6. Usage

### Installation

```
pip install -r requirements.txt
```

### Run FISTA–PALM from CLI

```
python -m src.cli --m 100 --n 80 --rank 10 --plot
```

### Run experiments

```
jupyter notebook notebooks/
```


## 7. References

Bolte, Sabach, Teboulle (2014)  
*Proximal alternating linearized minimization for nonconvex and nonsmooth problems*  
https://doi.org/10.1007/s10107-013-0701-9  

Pock, Sabach (2016)  
*Inertial Proximal Alternating Linearized Minimization (iPALM) for Nonconvex and Nonsmooth Problems*  
https://doi.org/10.1137/16M1064064  

Liang, Monteiro, Sim (2019)  
*A FISTA-type accelerated gradient algorithm for solving smooth nonconvex composite optimization problems*  
https://arxiv.org/abs/1905.07010  

Beck, Teboulle (2009)  
*A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*  
https://doi.org/10.1137/080716542  


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.