# Constrained Flow Matching

A PyTorch research repository exploring methods to apply **geometric, structural, and physics-inspired constraints** to pre-trained Continuous Normalizing Flows (Flow Matching) *without* retraining the base unconditional models.

Standard generative models excel at mapping noise to a target distribution, but enforcing strict, hard constraints at inference time (e.g., "generated points must stay within these boundaries" or "this region of the image must be completely black") is notoriously difficult. This repository investigates inference-time trajectory optimization and distillation techniques to solve this problem.

---

## Project Modules

This repository is divided into two main areas of research. **Click on the module titles** to view the detailed experiments, methodologies, and visual results for each.

### 1. [Zero-Shot Constrained Sampling: ECI vs. HardFlow](./eci_vs_hardflow)
*How do we force a pre-trained model to obey rules it never saw during training?*

This module compares two distinct inference-time sampling algorithms designed to enforce hard constraints on standard unconditional Flow Matching models:
* **ECI (Extrapolation, Correction, Interpolation):** A fast, gradient-free geometric projection method.
* **HardFlow (Trajectory Optimization):** A differentiable method that uses gradient guidance to gently steer the vector field around constraints.

**Experiments Include:** * 2D Checkerboard bounded generation.
* High-dimensional MNIST constraints (Inpainting, Total Ink control, PCA Subspace projection, and Structural Symmetry).

### 2. [Dynamic HardFlow Distillation](./constraints_distillation)
*How do we make gradient-guided sampling fast enough for real-time inference?*

Calculating gradients (`requires_grad`) through an ODE solver at every integration step (as done in HardFlow) is computationally expensive. This module explores **distilling** that expensive optimization process into a blazing-fast, lightweight Student MLP.

**Key Features:**
* **1D GMM Environment:** Training a base Flow Matching model on a 2-peak Gaussian Mixture Model.
* **Dynamic Student Training:** By engineering relative-distance features, we train a single Student MLP to predict the correct gradient penalty for *arbitrary, unseen boundaries* in a single forward pass.
* **Feature Pruning:** Using Permutation Feature Importance to heavily optimize the student network's inputs.

---

## References & Inspiration

The methodologies implemented and adapted in this repository are heavily inspired by the following papers:

* **HardFlow:** [Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization](https://arxiv.org/pdf/2511.08425)
* **ECI:** [Gradient-Free Generation for Hard-Constrained Systems](https://arxiv.org/abs/2412.01786)

---
*Developed using PyTorch.*