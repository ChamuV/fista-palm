# FISTA–PALM

## Accelerated Block-Coordinate Optimisation for Nonconvex Problems

This repository contains the implementation and experimental study of the **FISTA–PALM algorithm**, a hybrid optimisation method for solving structured nonconvex problems.

The work was developed as part of an MSc project at the University of Oxford, with the aim of investigating how acceleration techniques can be integrated into block-coordinate optimisation methods.

The project focuses on both:

- algorithmic design, and
- empirical evaluation across a range of problem settings

---

## 1. Problem Setting

We consider optimisation problems of the form:

\[
\min_{x,y} \; f(x) + g(y) + H(x,y)
\]

where:

- $f, g$ are nonsmooth but proximable
- $H$ is smooth and couples the variables

This formulation appears in:

- sparse matrix factorisation
- image reconstruction and denoising
- representation learning

Two models studied in this repository are:

### Two-block factorisation

\[
A \approx X Y^T
\]

### Three-block factorisation

\[
A \approx X B Y^T
\]

---

## 2. Algorithms

The repository implements:

- **PALM** — Proximal Alternating Linearised Minimisation
- **iPALM** — Inertial extension of PALM
- **FISTA–PALM** — accelerated variant introduced in this project

---

## 3. FISTA–PALM Algorithm

FISTA–PALM combines:

- the block-coordinate structure of PALM
- Nesterov acceleration from FISTA

The key idea is to perform updates at an extrapolated point rather than the current iterate.

### Algorithm Outline

Given initial variables $x^0, y^0$, set $t_0 = 1$. For $k = 0,1,\dots$:

1. **Compute extrapolation factor**

\[
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}
\]

2. **Extrapolate variables**

\[
\tilde{x}_k = x_k + \frac{t_k - 1}{t_{k+1}} (x_k - x_{k-1})
\]

\[
\tilde{y}_k = y_k + \frac{t_k - 1}{t_{k+1}} (y_k - y_{k-1})
\]

3. **Update first block**

\[
x_{k+1} = \operatorname{prox}_{f/L_x} \left( \tilde{x}_k - \frac{1}{L_x} \nabla_x H(\tilde{x}_k, y_k) \right)
\]

4. **Update second block**

\[
y_{k+1} = \operatorname{prox}_{g/L_y} \left( \tilde{y}_k - \frac{1}{L_y} \nabla_y H(x_{k+1}, \tilde{y}_k) \right)
\]

---

### Interpretation

- PALM performs alternating proximal gradient steps
- FISTA–PALM introduces momentum across iterations
- This leads to:
  - faster objective decrease
  - improved practical performance
  - minimal additional tuning

---

## 4. Repository Overview