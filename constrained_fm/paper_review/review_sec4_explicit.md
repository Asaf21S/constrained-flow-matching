# Review — Section 4: Explicit Method & Main Results

**Verdict:** the method matches the code, and the explicit, ECI and HardFlow numbers are correct. Four factual errors must be fixed before submission.

## Must fix

- [fixed] **1. The table's "Implicit" column holds the few-shot numbers.**
  - Correct implicit values: AR **97.92**, SWD **0.0706**, MMD **0.00083**, KLD **0.0461**.
  - Also change `lrrrrr` to `lrrrr`; the table has 5 columns, not 6.

- [pass] **2. "Our approach stays stable across the entire constraint spectrum" is false.**
  - All methods degrade on low-mass constraints. Explicit SWD goes from 0.05 to 0.22, and its AR drops to about 92 % in the lowest-mass decile.
  - What does hold: our models are best in every mass decile, and HardFlow degrades the most.
  - Replacement sentence:
    ```latex
    All methods deteriorate as the feasible mass decreases, but the amortized models remain the most faithful throughout: in the lowest-mass decile ($Z_\phi<0.12$) the explicit model's SWD is $3\times$ lower than ECI's and $6\times$ lower than HardFlow's, whose fidelity collapses on restrictive constraints.
    ```

- [fixed] **3. The SWD/MMD trend plots are not "a function of target mass".**
  - Their x-axis is the ground-truth SWD/MMD (the noise floor). Only the AR and KLD plots use mass.
  - Fix: either replot SWD/MMD against mass (plot-only change), or correct the caption and text.

- [ ] **4. The GMM density figure (setup, left) is transposed.**
  - The modes appear mirrored across the diagonal compared with all the sample plots.
  - Fix: in `density.py`, use `density_grid.T`, then rerender.

- [fixed] **5. The baselines are misdescribed.**
  - Our HardFlow *projects the predicted terminal state*; it is not "gradient guidance". Both ECI and HardFlow are feasible **by construction** (final projection), so "ECI by design, HardFlow empirically" is inaccurate.
  - Fix the 5-panel figure label "(inference guidance)" to match.

- [ ] **6. The exactness claim is unsupported, and the loss is written incorrectly.**
  - Problem: the text calls the sign-flip trick a "heuristic". It is exact, and the abstract claims "we show" this.
  - Problem: "geometric inversions" are not implemented.
  - Problem: the constraints seen in training are **size-biased**, $q(\phi)\propto\pi(\phi)Z_\phi$, not $p(\phi)$.
  - Fix: add a short proposition:
    ```latex
    \begin{proposition}\label{prop:orientation}
    Let $\pi(\phi)=\pi(-\phi)$ and $C_{-\phi}=-C_\phi$. Draw $x_1\sim p_{\text{data}}$, $\tilde\phi\sim\pi$, and set $\phi=\tilde\phi$ if $C_{\tilde\phi}(x_1)\le0$, else $\phi=-\tilde\phi$. Then $q(x_1\mid\phi)=p(x_1\mid\phi)$ exactly, and $q(\phi)=2\pi(\phi)Z_\phi$.
    \end{proposition}
    ```
    This needs `amsthm` in the preamble.

## Should fix

- [pass] **7. Acknowledge the feasibility gap.**
  - "Uninterrupted ODE pass … without post-hoc artifacting" hides that AR < 100 %.
  - Add one sentence: infeasible samples can be dropped at a cost of $1/\mathrm{AR}$, versus $1/Z_\phi$ for rejection.
- [fixed] **8. Benchmark description.**
  - The mass bins are *equal-width* (50 constraints each), not "equal-depth".
  - The setup figure shows example constraints, not the stratification, so don't cite it for that.
- [pass] **9. Notation.**
  - The benchmark paragraph uses $P(x)$; use $C_\phi(x)$ throughout.
  - Eq. 1 uses `\mathbb{I}`; use `\mathds{1}`.
- [ ] **10. Figure labels.**
  - SR → AR (5-panel figure).
  - "Coefficients/Functa" → "Explicit/Implicit" (trend legends).
  - The setup figure's feasible regions are shaded blue-gray, not "gray".
- [ ] **11. Captions are underspecified.** Define:
  - AR;
  - SWD noise floor;
  - the first-order distance;
  - "on the wall" ($|d|<0.02$);
  - the trend x-axes.
- [pass] **12. Minimal reproducibility (one sentence in the text, the rest in an appendix).**
  - State the path: CondOT, $u_t=x_1-x_0$, prior $\mathcal N(0,I)$.
  - Say that the input also includes $x_t$.
  - Disclose that ECI was tuned on the validation set.

## Optional
- Move "Problem Formulation" and pair generation to a shared section, since the implicit method uses them too.
- Add a noise-floor column to the table (SWD 0.030, MMD 0.0003).
- Mention that our sampler uses about 10× fewer network evaluations than ECI. Wall-clock times are not recorded, though.
- Cut phrases: "dynamically warp", "progressively internalizes", "successfully amortizes", "rigorously supported", "severe density corruption".
