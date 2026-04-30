# FISTA–PALM

A modular and extensible implementation of advanced **nonconvex optimisation algorithms** combining **FISTA-style acceleration** with **PALM-type block coordinate updates**.

This project studies how modern optimisation techniques behave in practice across:
- sparse matrix factorisation
- image reconstruction and denoising
- multi-block nonconvex problems

---

## 🔍 Core Idea

We consider optimisation problems of the form:

\[
\min_{x} \; f(x) + g(x)
\]

where:
- \( f \) is smooth (differentiable)
- \( g \) is non-smooth but proximable

and extend this to **multi-block settings** such as:

\[
A \approx X Y^T \quad \text{(2-block)}
\]

\[
A \approx X B Y^T \quad \text{(3-block)}
\]

---

## 🚀 What This Project Does

This repository goes beyond a simple implementation and provides a **full experimental study** of:

### Algorithms
- **PALM** (Proximal Alternating Linearised Minimisation)
- **iPALM** (inertial PALM)
- **FISTA–PALM** (accelerated variant)

### Experiments
- Synthetic matrix factorisation
- SVD vs random initialisation
- Time vs objective analysis
- Conditioning and singular value scaling
- ORL face reconstruction
- BSDS500 image denoising
- COIL-20 **three-block sparse factorisation**

### Key Contributions
- Clean abstraction of optimisation problems
- Plug-and-play solver interface
- Extension from 2-block → 3-block nonconvex optimisation
- Systematic empirical comparison across regimes

---

## 🧠 Why This Is Interesting

While FISTA is well understood for convex problems, its behaviour in **nonconvex and block-structured settings** is much less clear.

This project explores:
- when acceleration helps
- when it destabilises optimisation
- how inertia interacts with proximal updates
- how complexity increases in multi-block models

---

## 📊 Repository Structure

```
src/
  problems/     problem formulations (2-block, 3-block)
  solvers/      PALM, iPALM, FISTA–PALM implementations
  utils/        data loading, plotting, IO

notebooks/      full experimental pipeline (10 experiments)

results/        generated figures and tables

tests/          basic validation
```

---

## ⚡ Quick Start

Install dependencies:

```
pip install -r requirements.txt
```

Run the CLI (FISTA–PALM on synthetic data):

```
python -m src.cli --m 100 --n 80 --rank 10 --plot
```

Run experiments:

```
jupyter notebook notebooks/
```

---

## 📈 Highlights

- FISTA–PALM consistently achieves **faster objective decay** than PALM
- Acceleration benefits persist in **image reconstruction tasks**
- Multi-block problems (e.g. COIL-20) expose **limitations of naive alternating schemes**
- Synthetic vs real data comparisons reveal **robustness patterns**

---

## 📚 References

**PALM**  
Bolte, Sabach, Teboulle (2014)  
*Proximal Alternating Linearized Minimization for Nonconvex and Nonsmooth Problems*  
https://doi.org/10.1137/140965641

**iPALM**  
Pock, Sabach (2016)  
*Inertial Proximal Alternating Linearized Minimization (iPALM)*  
https://doi.org/10.1137/15M1021839

**FISTA**  
Beck, Teboulle (2009)  
*A Fast Iterative Shrinkage-Thresholding Algorithm*  
https://doi.org/10.1137/080716542

---

## 📄 License

MIT License

---

## ✨ Notes

This project was developed as a research-style exploration of modern optimisation methods, with an emphasis on **clarity, extensibility, and empirical insight** rather than black-box usage.