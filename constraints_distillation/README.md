# Distilling Dynamic HardFlow for 1D Flow Matching

*[← Back to Main Repository](../README.md)*

This module explores applying **HardFlow constraints** to a Flow Matching model through distillation of the expensive gradient-guided sampling process into a blazing-fast, **dynamic Student MLP**.

<a href="https://colab.research.google.com/github/Asaf21S/constrained-flow-matching/blob/main/constraints_distillation/constrained_gaussian_distillation.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 1. The Target Dataset
The ground truth data is a 1D Gaussian Mixture Model (GMM) consisting of two distinct peaks:
* Peak 1: Mean 2, Std 1
* Peak 2: Mean -2, Std 2

![Original GMM Dataset](images/2_peaks/1d_gmm_dataset.png)

## 2. The Unconstrained Base Model
The foundational model is an unconditional Flow Matching MLP trained to map standard Gaussian noise to the unconstrained 1D GMM target distribution. 

![Unconstrained FM Generation](images/2_peaks/1d_gmm_fm.png)

## 3. The Teacher: HardFlow with Gradient Guidance
To constrain the generated data without retraining the base model, we implement a HardFlow-style guidance loop. During the Euler integration steps, we predict the destination (`x1_hat`), calculate a **squared Softplus barrier loss** if it falls outside our desired boundaries, and subtract the gradient of that loss from the velocity. 

Using a squared Softplus provides a smooth, continuous gradient as points approach the boundaries, preventing exploding gradients while firmly pushing wandering points back inside. Combined with a linear guidance schedule (scaling the penalty by time `t`), the trajectories smoothly bend to respect the boundaries. 

**Example: Static Boundaries at [-3, 3]**
*(Achieved 99.44% boundary accuracy)*
![HardFlow Base Model](images/2_peaks/1d_gmm_fm_hardflow.png)

## 4. Dynamic Distillation & Feature Engineering
Calculating `requires_grad` inside an ODE solver is computationally expensive. To solve this, we train a **Student MLP** to predict the necessary gradient correction in a single forward pass.

Instead of training a static student for a single set of boundaries, we made the student **dynamic**. By generating a dataset with randomized boundaries for every sample, we forced the student to learn arbitrary constraints.  
Specifically, we sampled random boundaries in the ranges:
- Min Boundary: (-7.5, 0)
- Max Boundary: (0, 5)

**Teacher Baseline Accuracy:** Across this randomized training dataset, the Teacher model achieved a baseline boundary accuracy of **96.87%**.

### Feature Importance & Pruning
To optimize the student, we engineered relative distance features and computed **Permutation Feature Importance** on a validation set to see what the network actually relied on.

![Feature Importance Chart](images/2_peaks/student_feature_importance.png)

## 5. Distilled Student Results
The distilled dynamic student successfully guides the flow to respect varying, unseen boundaries at inference time with zero backpropagation, closely matching the Teacher's baseline accuracy across drastically different constraints.

**Dynamic Boundaries: [-3, 3]**
*(Accuracy: 99.47%)*
![Student -3 to 3](images/2_peaks/1d_gmm_fm_hardflow_student_-3_3.png)

**Dynamic Boundaries: [-2, 2]**
*(Accuracy: 97.09%)*
![Student -2 to 2](images/2_peaks/1d_gmm_fm_hardflow_student_-2_2.png)

**Dynamic Boundaries: [-1, 1]**
*(Accuracy: 92.58%)*
![Student -1 to 1](images/2_peaks/1d_gmm_fm_hardflow_student_-1_1.png)

**Dynamic Boundaries: [-2, 4]**
*(Accuracy: 99.78%)*
![Student -2 to 4](images/2_peaks/1d_gmm_fm_hardflow_student_-2_4.png)

**Dynamic Boundaries: [-7, 2]**
*(Accuracy: 97.61%)*
![Student -7 to 2](images/2_peaks/1d_gmm_fm_hardflow_student_-7_2.png)

**Dynamic Boundaries: [0, 3]**
*(Accuracy: 99.18%)*
![Student 0 to 3](images/2_peaks/1d_gmm_fm_hardflow_student_0_3.png)

**Dynamic Boundaries: [-3, 0]**
*(Accuracy: 91.72%)*
![Student -3 to 0](images/2_peaks/1d_gmm_fm_hardflow_student_-3_0.png)

### Conclusion
By passing the relative boundary distances as features to a normalized Student MLP, we can distill expensive ODE gradient-guidance into a fast, arbitrary-boundary inference step while maintaining the structural integrity of the underlying Flow Matching distribution.
