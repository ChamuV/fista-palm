

# FISTA–PALM

Implementation of a hybrid **FISTA–PALM optimisation algorithm** for solving problems of the form:

\[
\min_x \; f(x) + g(x)
\]

where:
- \( f \) is smooth and differentiable
- \( g \) is possibly non-smooth but admits a proximal operator

---

## Overview

This project explores a combination of:
- **FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)** for acceleration
- **PALM (Proximal Alternating Linearised Minimisation)** for handling non-smooth terms

The goal is to study how acceleration and proximal splitting interact in practice.

---

## Structure

```
src/           core algorithms and operators  
experiments/   scripts to run examples  
report/        project report  
results/       generated plots/logs  
tests/         basic tests  
```

---

## Quick Start

Install dependencies:

```
pip install -r requirements.txt
```

Run example:

```
python experiments/run_lasso.py
```

---

## Current Features

- FISTA-style acceleration
- Proximal gradient updates
- Example: LASSO reconstruction

---

## To Do

- Add convergence plots  
- Compare with vanilla FISTA / PALM  
- Extend to more problems (matrix completion, etc.)

---

## Notes

This repo is based on a coursework/research-style project and is intended to be extended.