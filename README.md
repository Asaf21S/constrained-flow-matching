# Constrained Flow Matching

A PyTorch implementation of **Flow Matching**, exploring standard unconditional generation and various zero-shot hard-constrained sampling methods.

This repository focuses on adapting pre-trained Flow Matching models to satisfy geometric and physics-inspired constraints without retraining, currently using the **[ECI (Extrapolation, Correction, Interpolation)](https://arxiv.org/abs/2412.01786)** algorithm.

---

## 1. Checkerboard Experiments

### Standard Generation (Unconstrained)
A baseline implementation of Flow Matching on 2D checkerboard data.
<br>
<a href="https://colab.research.google.com/github/Asaf21S/constrained-flow-matching/blob/main/flow_matching_checkerboard.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<p align="center">
  <img src="images/flow_matching_checkerboard_samples.png" width="500" title="Unconstrained Checkerboard">
</p>

### ECI Constrained Sampling
Applying the ECI algorithm to force samples into the black squares.
<br>
<a href="https://colab.research.google.com/github/Asaf21S/constrained-flow-matching/blob/main/flow_matching_checkerboard_ECI.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<p align="center">
  <img src="images/flow_matching_checkerboard_samples_eci.png" width="500" title="Constrained Checkerboard">
</p>

---

## 2. MNIST Experiments

### Baseline Generation
Standard unconditional generation of MNIST digits using a U-Net based vector field.
<br>
<a href="https://colab.research.google.com/github/Asaf21S/constrained-flow-matching/blob/main/flow_matching_mnist.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<p align="center">
  <img src="images/flow_matching_mnist_samples.png" width="600" title="Standard MNIST Generation">
</p>

---

### Hard-Constrained Generation (ECI)
All constraints below are applied at inference time to the **same** pre-trained unconditional model using the ECI sampling trajectory.
<br>
<a href="https://colab.research.google.com/github/Asaf21S/constrained-flow-matching/blob/main/flow_matching_mnist_ECI.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Experiment A: Inpainting
**Constraint:** Force the center $6 \times 6$ pixels to be black.
<br>

<p align="center">
  <img src="images/flow_matching_mninst_samples_eci_center_hole.png" width="400" title="Center Hole Constraint">
</p>

#### Experiment B: Physics-Inspired Constraints (Total Ink)
**Constraint:** Control the total sum of pixel intensities ("Ink Amount").
<br>
*Left: Low Ink (K = 60). Right: High Ink (K = 150).*

<p align="center">
  <img src="images/flow_matching_mninst_samples_eci_ink_amount_60.png" width="350" style="margin-right: 20px;">
  <img src="images/flow_matching_mninst_samples_eci_ink_amount_150.png" width="350">
</p>

#### Experiment C: Subspace Projection (Classifier-Free Guidance)
**Constraint:** Project the noisy state onto the PCA subspace of a specific digit class.

*Targeting Digit: 3*
<p align="center">
  <img src="images/flow_matching_mninst_samples_eci_specific_digit_3.png" width="400" title="PCA Constraint: 3">
</p>

*Targeting All Digits (0-9)*
<p align="center">
  <img src="images/flow_matching_mninst_samples_eci_specific_digit_all.png" width="600" title="PCA Constraint: All Digits">
</p>
