# Review — Section 5: Implicit Method

**Verdict:** the encoder description matches the code. The main problems are that the figures show a different model from the one behind the results, plus one false comparison.

## Must fix

- [fixed] **1. The figures show the joint polynomial + polygon SIREN, which is *not* used for generation.**
  - All implicit results come from the polynomial-only SIREN.
  - The joint model was also trained on different targets than the $\tanh(P(x))$ the text states.
  - Fix: add something like:
    ```latex
    To illustrate that the encoder is not tied to one constraint family, we also meta-learn a SIREN jointly on polynomials and convex polygons (Figures~\ref{fig:siren_grid},~\ref{fig:siren_interp}); all generative results use the polynomial-only encoder.
    ```

- [fixed] **2. The interpolation figure comes from an outdated checkpoint.**
  - It was made with the old `joint_siren`, while the grid figure uses the newer `joint_siren_sharp`.
  - Fix: regenerate with `--siren-dir constrained_fm/functa_dataset/joint_siren_sharp`, and update the script default.

- [fixed] **3. "Lower … KLD than inference-time baselines" is false.** KLD is undefined for ECI and HardFlow. Remove it from that sentence.

- [fixed] **4. The CAVIA expansion is wrong.** CAVIA is "Fast Context Adaptation via Meta-Learning" (Zintgraf et al., 2019). Cite Functa (Dupont et al., 2022) separately.

## Should fix

- [pass] **5. The encoder has no quantitative result.** Add one sentence:
  ```latex
  On held-out constraints the decoded region matches the true one with median mass-IoU $0.984$ (5th percentile $0.939$), and the implicit model's acceptance rate correlates with this IoU ($r=0.79$).
  ```
- [pass] **6. Replace "closely trails" with numbers.**
  - Implicit vs. explicit: AR 97.9 vs 98.6, SWD +14 %, but **KLD 0.046 vs 0.028**.
  - Add: "the two models are not architecture-matched". The implicit model has about 2× the parameters, while the explicit one has 4× the batch size and fresh constraints at every step.
- [pass] **7. Overclaims.**
  - "Only observable as spatial queries" → the encoder needs *continuous* values $\tanh C(x)$.
  - "Sparse queries" → it uses 1000 uniform points.
  - "Empirical geometries" → never tested.
- [pass and fixed] **8. Architecture inaccuracies.**
  - Time is not concatenated with $\psi$; it enters only through AdaGN.
  - Latents are precomputed once per constraint, not over "a diverse distribution of query sets".
- [fixed] **9. The rationale for the SIREN feature is not backed by evidence.** An ablation on an older encoder showed almost no effect. Soften "to provide localized geometric grounding" to a design choice.
- [fixed] **10. Notation and references.**
  - The grid colorbar $f_\theta(x,z)$ → $f_\omega(x,\psi)$, because $\theta$ is the FM.
  - The path is defined in `sec:explicit_method`, not `sec:explicit_training`.
  - The figure includes both `functa_diagram.tex` and `functa_diagram_v2.tex`; keep only v2.

## Optional
- Put all training details in §5.1 and make §5.3 a short "Results" paragraph.
- Headings: "IMPLICIT METHOD" → "Implicit Constraint Amortization". This title is currently repeated as the subsection title.
- Cut phrases: "Crucially", "Remarkably", "seamlessly", "distilling the boundary", "geometric routing".
- Give the interpolation a purpose. One sentence: a smooth latent space is what makes constraint optimization (Sec. 6) possible.
