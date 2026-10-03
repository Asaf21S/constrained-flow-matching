# Review — Abstract & Figure 1

**Verdict:** the abstract is clear and every claim has a matching experiment. The fixes below are about scoping the claims, not about wrong numbers.

## Must fix

- [ ] **1. "Enforcing hard constraints" overstates what the method does.**
  - Problem: our models reach a median AR of 98.6 % (explicit) and 97.9 % (implicit). ECI and HardFlow reach 100 %. "Outperforming" them is only true for fidelity and likelihood, not for feasibility.
  - Fix: name the axis on which we win (fidelity + exact likelihood). Add that the few infeasible samples can be discarded at negligible cost.

- [ ] **2. "In the low-data regime" needs a threshold.**
  - Problem: the implicit model beats fine-tuning only below roughly 500 samples. The two are at parity around 1000, and fine-tuning is slightly ahead at 2000.
  - Fix: write "when fewer than a few hundred constraint samples are available".

- [ ] **3. The implicit representation is described inaccurately.**
  - Problem: the encoder receives *continuous* values $\tanh C(x)$ at query points, not membership answers. It also runs a 15-step latent fit at test time.
  - Fix: replace "defined only via spatial queries" with "accessed only through pointwise evaluations of a constraint function". Replace "zero-shot" with "without retraining".

- [ ] **4. The Fig. 1 caption is wrong and incomplete.**
  - Problem: it calls all panels "constraint amortization methods", but ECI and HardFlow are not.
  - Problem: it leaves AR, SWD, the dashed overlay and "on the wall" undefined.
  - Problem: the x-axis is the first-order distance $C/\lVert\nabla C\rVert$, not the true signed distance.
  - In the figure itself, change **SR → AR**.

## Should fix

- [ ] **5. State the setting.** The target distribution is fixed and only the constraint varies. This is never said.
- [ ] **6. Add one headline number.** For example: SWD is 6.5–10× lower than ECI/HardFlow over 1000 held-out constraints.
- [ ] **7. Make the last sentence concrete.** Optimizing the constraint means *fitting it to data*. For IS, give the result: up to 5.6× lower error than equal-time rejection.
- [ ] **8. Wording.**
  - "heuristics" → "inference-time correction methods"
  - fix the typo "is is"
  - fix the double space in "by  gradient"

## Optional
- Poly 86 has median mass ($Z\approx0.48$), so Fig. 1 does not show a *rare* constraint. Either say "median-mass" in the caption or swap in a low-mass polynomial.

---

## Suggested abstract (similar length)
```latex
Sampling from a fixed target distribution restricted to a constraint set that changes at test time typically requires inference-time computation. Rejection sampling is exact but its cost grows inversely with the feasible probability mass, while projecting or guiding sampling trajectories guarantees feasibility at the cost of distorting the target, piling mass onto the constraint boundary. We propose \emph{constraint amortization}: a single flow matching model is trained across a family of constraints and receives the constraint as a conditioning input, shifting the cost of constraint satisfaction offline. We show that exact training pairs can be generated without rejection. We introduce an \emph{explicit} representation, conditioned on the constraint's analytic parameters, and an \emph{implicit} one, which accesses the constraint only through pointwise evaluations. Both sample and evaluate exact likelihoods for unseen constraints without retraining, reduce the sliced Wasserstein distance to the ground truth by $6.5$--$10\times$ relative to inference-time correction, and outperform per-constraint fine-tuning when few constraint samples are available. Finally, amortization enables fitting constraints to data by gradient-based optimization and unbiased importance-sampling estimates of constrained expectations, with up to $5.6\times$ lower error than rejection sampling at equal cost.
```

## Suggested Fig. 1 caption
```latex
\captionof{figure}{Inference-time correction vs.\ constraint amortization on a held-out cubic constraint (dashed: boundary). \textbf{Top:} sample densities of rejection sampling (ground truth), ECI, HardFlow, and our explicit model. AR: fraction of feasible samples; SWD: distance to an independent ground-truth draw, so the ground-truth value is the noise floor. \textbf{Bottom:} first-order signed distance $C(x)/\lVert\nabla C(x)\rVert$ (negative inside), with the ground truth dashed; percentages give the mass within $0.02$ of the boundary. ECI and HardFlow are always feasible but pile mass on the boundary.}
```
