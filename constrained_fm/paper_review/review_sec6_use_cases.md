# Review — Section 6: Use Cases

**Verdict:** both experiments are real and solid. The constraint-optimization text describes a different objective and different figures from what the code produced. The IS text credits importance sampling for a gain that comes mostly from amortization.

## Must fix — Constraint Optimization

- [fixed] **1. The objective is wrong.**
  - The code *minimizes the flow-matching loss* on the 3-mode samples. It never computes a likelihood and never integrates a trajectory. The exact likelihood actually diverges for the optimized latents.
  - Replace the paragraph with:
    ```latex
    With the implicit model frozen, we optimize the latent $\psi$ to minimize the conditional flow matching loss on the target samples. The objective is simulation-free: gradients reach $\psi$ through the velocity field at interpolated points, without solving the ODE. To stay on the manifold of valid encodings, $\psi$ is restricted to the principal subspace of the training latents.
    ```
- [fixed] **2. The right panels are not likelihood curves.**
  - `likelihood.pdf` is a single final density map, not an evolution.
  - Fix: swap in `history.pdf` (FM loss + per-mode inside fraction), or rewrite the text and caption to describe a final density map.
- [fixed] **3. The narrative contradicts the strip figure.**
  - At step 0 the region is *fragmented*, not "overly permissive".
  - The final boundary *separates* the excluded mode; it does not wrap the target modes tightly.
  - Report the outcome instead: 94–100 % of each target mode is inside, and 5–9 % of the excluded mode remains.
  - Remove "optimal encapsulation".

## Must fix — Conditional Expectations

- [pass] **4. The figure misattributes the gain.**
  - At $N=1000$, the plain amortized samples are as accurate as IS: in the 0.2 % box, 1.8e-3 vs 1.7e-3. So the ~6× win over rejection comes from **amortization**.
  - IS removes the remaining bias, which matters at large $N$ (up to 4.8× lower error at $N=10^5$).
  - Fix: add one sentence, or add a `q_filtered` curve to the figure.
- [fixed] **5. "Error diverges exponentially" is wrong.** It grows roughly as $P(\mathcal B)^{-1/2}$.
- [ ] **6. The derivation has wrong lines.**
  - "$p(c)=\mathbb E_{p(x|c)}\,p(c)$" should read $Z_\phi=\mathbb E_{q}[\mathds 1[x\in\Omega_\phi]\,p(x)/q(x\mid\phi)]$.
  - The first estimator samples $x_i\sim p(x|c)$; it should be $x_i\sim q$.

## Should fix

- [pass] **7. State when IS wins.**
  - The result holds with 64 ODE steps and for boxes with $P(\mathcal B)\le2\%$.
  - At 10 % mass, equal-time rejection is as good. With ≤ 16 steps, rejection wins.
  - Fix the caption's "even with equal time … diverge" accordingly.
- [fixed] **8. State the premise.** Rejection samples the *learned* unconstrained flow, not the simulator. Add one clause: the simulator is assumed expensive or unavailable.
- [ ] **9. Box training pairs use a different exact construction.**
  - The box centre is drawn uniformly so that the box contains $x$; this is not the sign flip.
  - Add one sentence so the "exact without rejection" claim covers this case.
- [ ] **10. Notation and style.**
  - $c$, $p(c)$, $x\in c$ → $\phi$, $Z_\phi$, $x\in\Omega_\phi$; also write $q_\theta$.
  - `eqnarray*` → `align*`.
  - Use `\mathds{1}` consistently.
  - The derivation is in draft voice ("Assume we have…", "Now we have everything…"). Condense it to one paragraph plus the final estimator.
- [fixed] **11. Discovery figure fixes.**
  - Subfigure widths sum to 1.06\textwidth and overflow.
  - The two subcaptions are identical; label them "mode 1/2 excluded".
  - Legend $f_\theta(x,z_c)$ → $f_\omega(x,\psi)$.
  - Fix "freezed".

## Optional
- "Invariant mass" → "mass scale"; the toy model is not relativistic.
- Section title: "Applications Enabled by Amortization", with one opening line: exact likelihood → IS; differentiable conditioning → constraint optimization.
- Remove the `% PLACEHOLDER` comment.

## Suggested compact derivation (replaces both `eqnarray*` blocks)
```latex
The amortized model $q_\theta(x\mid\phi)$ places almost all samples in $\Omega_\phi$ but is approximate, so averaging over its samples is biased. Since it provides exact likelihoods, the bias is removed by importance sampling with any accurate model $p$ of the unconstrained target:
\begin{equation}
\mathbb{E}_{p(x\mid\phi)}[r(x)] \approx \sum_{i=1}^N \frac{w_i}{\sum_j w_j}\, r(x_i),\qquad
w_i=\frac{\mathds{1}[x_i\in\Omega_\phi]\,p(x_i)}{q_\theta(x_i\mid\phi)},\quad x_i\sim q_\theta(\cdot\mid\phi),
\end{equation}
where $\frac1N\sum_i w_i$ estimates $Z_\phi$. Leaked samples receive zero weight, so imperfect feasibility costs samples but adds no bias.
```
