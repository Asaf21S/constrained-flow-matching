# Cross-Section Consistency Log

A running record of the canonical facts, notation and numbers used across reviews. Later sections are checked against it.

## User decisions
- bump2d and kinematics6d are **not** in the paper; do not check them.
- Use **Acceptance Rate (AR)** everywhere. Every "SR" in figures and text becomes AR.
- An Overleaf file named `<name>_vN.pdf` is the same file as repo `<name>.pdf`.
- Few-shot baseline = **fine-tuned** variant (`baselines/few_shot_finetuned_v1k`). The definition of $N$ is consolidated in the appendix (few-shot vs. query points); review that last.
- The implicit method is defined in Section 5 (`sec:implicit_method`). Results that use it before Section 5 must say so.

## Preamble / macros
- Comment macros: `\dan{}`, `\asaf{}`, `\lukas{}` (toggled by `\commentstrue`).
- Packages: amsmath, amssymb, dsfont (`\mathds{1}`), booktabs, subcaption, tikz, natbib (round).
- `amsthm` is **not** loaded; it is needed for the proposition in the Section 4 review.

## Paper notation (as of Section 4)
| Symbol | Meaning | Notes |
| --- | --- | --- |
| $x\in\mathbb R^D$, $p_{\text{data}}$ | unconstrained target | — |
| $\phi\in\Phi$ | constraint parameters | explicit: $\phi\in\mathbb R^{16}$, unit Frobenius norm |
| $C_\phi(x)$ | constraint function, feasible iff $\le 0$ | the Sec. 4 benchmark paragraph uses $P(x)$, which conflicts; use $C_\phi$ |
| $\Omega_\phi$ | feasible set | — |
| $Z_\phi$ | feasible mass | — |
| $p(x\mid\phi)$ | truncated target | Eq. 1 uses `\mathbb{I}`; prefer `\mathds{1}` |
| $v_\theta(x_t,t,\phi,C_\phi(x_t))$ | explicit velocity field | matches code |
| $u_t(x_t\mid x_0,x_1)$ | conditional target | CondOT: $x_1-x_0$ |
| $\psi\in\mathbb R^{512}$ | implicit latent (CAVIA context) | code and figures use $z$; the grid colorbar reads $f_\theta(x,z)$ and must become $f_\omega(x,\psi)$ |
| $f_\omega(x,\psi)$ | modulated SIREN, base weights $\omega$ | — |
| $v_\theta(x_t,t,\psi,f_\omega(x_t,\psi))$ | implicit velocity field | — |

## Canonical terminology
| Concept | Use | Avoid / conflicts seen |
| --- | --- | --- |
| Method | constraint amortization | — |
| Coefficient-conditioned model | Explicit | trend-plot legends "Coefficients (ours)"; Fig. 1 title "Constraint Amortization (Ours)" |
| Functa-conditioned model | Implicit | trend-plot legends "Functa (ours)"; "spatial queries" |
| ECI, HardFlow | inference-time correction methods; both are **projection-based** | "heuristics"; "HardFlow gradient guidance"; 5-panel subtitle "(inference guidance)" |
| Fraction feasible | Acceptance Rate (AR) | "SR" in Fig. 1 and the 5-panel figure |
| Bottom-row x-axis | first-order signed distance $C_\phi/\lVert\nabla C_\phi\rVert$ | "signed distance" |
| Near-boundary mass | fraction with $\lvert d\rvert<0.02$ | "on the wall" (undefined) |
| Training-pair construction | sign-flip / orientation construction (exact) | "orientation heuristic"; "geometric inversions" (not implemented) |

## Setup facts (code-verified)
- **Constraint family:**
  - degree-3 bivariate polynomials, $C_\phi(x)=\sum_{ij}\phi_{ij}\tilde x_1^i\tilde x_2^j$ with $\tilde x=x/4.5$;
  - $\phi\sim\mathcal N(0,I_{16})$, kept if the uniform-area fraction (10k proxy points) is in $[0.05,0.95]$, then Frobenius-normalized;
  - domain $[-4.5,4.5]^2$.
- **Target:** a fixed 4-component GMM.
  - means $(-1.5,-1.5),(1.5,2.0),(2.0,-1.5),(-0.5,0.5)$;
  - weights 0.35, 0.25, 0.15, 0.25.
  - **The `true_gmm_likelihood` figure is transposed** (`indexing='ij'` with no `.T`).
- **Exact pairing (polynomials):** sign flip. The joint is $2\pi(\phi)p_{\text{data}}(x)\mathds 1[x\in\Omega_\phi]$ and the marginal is $q(\phi)=2\pi(\phi)Z_\phi$ (size-biased). `mass_weight_power: 0.0`, so the bias is uncorrected.
- **Explicit model:**
  - input: $x_t$, a 64-d sinusoidal time embedding, $\phi$ (16), and $C_\phi(x_t)$, for 83 dims in total;
  - network: Linear(83→1024)+SiLU, 4 residual MLP blocks (Linear–SiLU–Linear, no normalization), Linear(1024→2);
  - training: Adam, lr 1e-3 cosine-annealed to 1e-5, 15 001 iterations, batch 4096, a fresh $\phi$ per sample;
  - path: CondOT with an $\mathcal N(0,I)$ prior.
- **Implicit model** (run `siren-uniform-8d6375ab`):
  - encoder: 512-d latent, CAVIA with 15 SGD steps, 1000 uniform queries with targets $\tanh P(x)$;
  - FM network: 4 AdaGN blocks of width 1024, 128-d time embedding, conditioned on $z$ and on $\mathrm{SIREN}(x_t,z)$;
  - pool: 100k constraints.
- **Implicit details (Sec. 5):**
  - SIREN: 4 FiLM layers (scale + shift), width 512, $w_0=30$, tanh output.
  - Inner loop: 15 SGD steps, per-shape lr 6.25e-4.
  - Outer loop: Adam 1e-4, wd 1e-5, $\lambda_\psi$ 1e-4, second-order gradients, best epoch 700, meta-val MSE 3.46e-3.
  - Implicit FM: **19.05 M** params vs Explicit **8.48 M**; batch **1024** vs 4096; fixed pool of 100k (× 2 orientations, $\psi^\pm$ re-extracted) vs fresh draws.
- **Encoder fidelity** (100-poly set): mass-IoU median 0.984 / mean 0.976 / p5 0.939; extraction MSE 4.0e-5; corr(AR, IoU) 0.79.
- **Joint SIREN** (figures only; NOT used for FM):
  - `joint_siren_sharp`: $\tau$ 0.9, poly gain 9.06, best epoch 1230; mean IoU poly 0.989 (p5 0.963), polygon 0.980 (p5 0.924).
  - The interpolation figure uses the OLD `joint_siren` (IoU 0.956 / 0.928) and is stale.
- **Ablations** (pre-fix, older encoder): removing the SIREN feature gives AR 90.79 vs 90.90 (near-null). `mass-power-half-u`: worst-5 % AR 86.79 → 88.71, but its KLD is not comparable.
- **Sampling and likelihood (ours):** midpoint solver, step 0.05 (20 steps); exact divergence; NLL on 5000 reference points. Implicit NLL was re-scored after the 2026-09-02 fix.
- **Baselines:**
  - both run on the unconditional base FM (`baselines/base_fm`) with 100 Euler steps;
  - ECI: $M=5$, $R=5$, tuned on val1k;
  - HardFlow: active from $t=0.5$;
  - projection: SQP, 32 iterations, margin 1e-3;
  - both project on the final step, so both are feasible by construction. ECI is below 100 % on 4/1000 constraints (minimum 92.3 %); HardFlow on 1/1000 (minimum 99.83 %).
- **val1k:** 1000 polynomials; 20 equal-*width* mass bins on $[0.02,0.98]$ with 50 each; realized mass range 2.11–97.89 %; mass estimated by Monte Carlo on $10^6$ points; seed 1000.
- **Metrics:**
  - 10k generated samples per constraint;
  - reference = 100k GMM pool (seed 20000) filtered to the constraint, i.e. $\approx Z\cdot10^5$ points;
  - SWD: POT, 50 projections, unseeded;
  - MMD: RBF with $\gamma=1$, subsampled to 5k;
  - SWD and MMD use all samples, including infeasible ones;
  - KLD $=D_{\mathrm{KL}}(p\Vert q_\theta)$ = NLL − MC reference entropy, which can be negative.
- **Trend plots:** sliding window of 100 constraints, step 25, IQR band. **SWD/MMD x-axis = GT noise floor** (log–log, $y=x$ dashed); AR/KLD x-axis = mass.
- **Test sets:** the 100-poly frozen benchmark (used by Fig. 1 poly 86 and the 5-panel poly 13) and val1k (Table and trends).

## Key numbers (newest runs)
- **Fig. 1, poly 86** (100-poly set; $Z=0.476$, the median), $10^5$ samples, 100 steps:
  - AR: GT 100.0, ECI 100.0, HF 100.0, Explicit 99.4 (Implicit 99.2);
  - SWD: 0.034, 0.397, 0.910, 0.056 (Implicit 0.091);
  - wall fraction: 0.6 / 3.9 / 4.9 / 0.6 %.
- **5-panel, poly 13** (100-poly set; $Z=0.464$, rank 49/100), in the order GT / ECI / HF / Explicit / Implicit:
  - AR: 100 / 100 / 100 / 98.9 / 98.8;
  - SWD: 0.007 / 0.289 / 0.608 / 0.027 / 0.040;
  - wall fraction: 1.0 / 3.6 / 22.3 / 1.0 / 0.6 %.
- **val1k medians**, in the order Explicit / Implicit / FT$_{N=1000}$ / ECI / HF / GT floor:
  - AR: 98.61 / 97.92 / 98.10 / 100 / 100 / 100;
  - SWD: 0.0617 / 0.0706 / 0.0712 / 0.4633 / 0.6100 / 0.0300;
  - MMD: 0.00069 / 0.00083 / 0.00108 / 0.02066 / 0.04722 / 0.00030;
  - KLD: 0.0275 / 0.0461 / 0.0495 / n/a / n/a.
  - **The Sec. 4 paper table put the FT values in the "Implicit" column.**
- **val1k by mass decile, lowest → highest** (100 constraints each):
  - SWD, Explicit: 0.222 → 0.046;
  - SWD, Implicit: 0.239 → 0.049;
  - SWD, ECI: 0.695 → 0.405;
  - SWD, HF: 1.266 → 0.128;
  - MMD, Explicit: 0.00350 → 0.00045;
  - MMD, Implicit: 0.00525 → 0.00047;
  - MMD, ECI: 0.0528 → 0.0120;
  - MMD, HF: 0.101 → 0.00175;
  - AR, Explicit: 91.85 → 99.51;
  - AR, Implicit: 88.56 → 99.31;
  - KLD, Explicit: 0.144 → 0.012;
  - KLD, Implicit: 0.206 → 0.016;
  - GT SWD floor: 0.037 → 0.031.
  - Conclusion: **all methods degrade at low mass**; ours remain best in every decile.
- **AR tails over val1k:** Explicit p5 91.2 %, minimum 48.1 %; Implicit p5 87.0 %, minimum 38.4 %.
- **Shared budget** (Implicit vs FT): Implicit wins every metric for $N\le500$; parity at 1000; FT ahead at 2000. The definition of $N$ goes in the appendix.
- **Constraint discovery** (excluded-mode inside fraction): coeffs 0.085 (Adam, 3000 steps); latent 0.091 / 0.054 (Adam, 250 steps, 15-d PCA subspace).
- **decay6d IS**, 64 steps: 5.6× lower RMSE than equal-time rejection on the 0.2 % box; IS wins on every box ≤ 2 %; ESS/N 0.95–0.97.
  - Paper figure: $N=10^3$. In the 0.2 % box at $N=10^3$, IS RMSE ($\lVert\vec p_2\rVert$) $1.73\times10^{-3}$ vs equal-time rejection $1.03\times10^{-2}$ (5.9×), q_raw $1.83\times10^{-3}$, q_filtered $1.81\times10^{-3}$. At $N=10^3$, IS ≈ the raw amortized sampler.
  - IS bias removal at $N=10^5$: q_raw → IS 2.1× (0.2 %, norm) up to 4.8× (10 %, tail).
  - At 10 % mass, equal-time rejection ≥ IS at $N=10^3$.
  - 32 steps: IS wins only for boxes ≤ 1 %. 8–16 steps: rejection wins.
  - All rejection baselines draw from the learned $p_{\text{uncon}}$, not the simulator.
  - Box flow: centre + log half-widths, width 1024, 3 blocks, 100k steps, EMA; training mass 0.1–20 %; **anchored-box** exact pairing (not sign flip); in-box rate 97 %; test leakage 0.6–2.1 %.
  - $D_{\mathrm{KL}}(p\Vert p_{\text{uncon}})=4.3\times10^{-4}$.
- **Constraint discovery** (Sec. 6):
  - Objective = CFM loss (NOT likelihood), simulation-free.
  - $\psi$ lives in a 15-d PCA subspace (explained variance ≈ 1); initialized at the pool-mean latent; Adam lr 1e-2, 250 steps.
  - Final inside fractions: exclude1 [1.00, **0.054**, 0.94, 0.93]; exclude2 [0.99, 0.98, **0.091**, 0.99].
  - The `likelihood.pdf` file is a final density map from samples (the exact likelihood blows up off-manifold); `history.pdf` holds the FM loss and per-mode curves.

## Claims made so far (later sections must support them)
- Abstract: exactness of the training conditional; zero-shot sampling and likelihood (both variants); fidelity superiority over ECI/HardFlow; FT threshold; constraint optimization; IS on decay6d.
- Section 4: proposition on orientation exactness (proposed); size-bias explains the low-mass degradation (a hypothesis with no ablation yet).

## Open flags
1. Provenance of `feasibility_fidelity_4panel_coeff_poly86_v3.pdf` (resolved: `_vN` = repo file).
2. Wall fractions are not persisted to `metrics.json` (Fig. 1 and the 5-panel).
3. `diagrams/coefficient_diagram.tex` is not in the repo.
4. No wall-clock sampling times; NFE counts are inferred from code.
5. No ablation of the $C_\phi(x_t)$ feature or of `mass_weight_power`; single seed.
6. ECI/HardFlow citation keys and appendix labels are placeholders.
7. Sec. 5: `functa_diagram{,_v2}.tex` are not in the repo (both included, likely a duplicate); no val1k encoder IoU; `siren_uniform.pt` provenance (commit efa8cab); the interpolation figure needs regeneration.
8. Sec. 5 text: KLD is wrongly claimed vs ECI/HF; the CAVIA expansion is wrong (it stands for Fast Context Adaptation via Meta-Learning, Zintgraf 2019).
9. Sec. 6: the IS derivation uses $c$, $p(c)$ and $x\in c$; unify to $\phi$, $Z_\phi$, $\Omega_\phi$, $q_\theta$. The Overleaf `_v2` discovery strips are untraced. The likelihood of discovered (off-manifold) latents is unstable.
