# Evaluation and Figure Mechanics

Reference documentation for the programmatic mechanics behind the visualization and
evaluation scripts. Every statement here is taken from the code paths cited alongside it.

Global constants ([`src/consts.py`](src/consts.py)):

| Constant | Value | Meaning |
| --- | --- | --- |
| `PLANE_SCALE` | `4.5` | domain half-width; everything lives on $[-4.5, 4.5]^2$ |
| `POLYNOMIAL_DEGREE` | `3` | constraint is a bivariate cubic, $4 \times 4$ coefficient matrix |
| `GMM_MEANS/COVS/WEIGHTS` | 4 components | the unconstrained target distribution $p_{\mathrm{gmm}}$ |
| `FUNCTA_QUERY_GMM_FRACTION` | `0.0` | CAVIA query points are pure uniform draws |
| `POLY_MIN/MAX_AREA_RATIO` | `0.05 / 0.95` | accepted GMM-mass range for training constraints |
| `VALIDATION_POLY_MIN/MAX_AREA_RATIO` | `0.1 / 0.9` | same, for the legacy 100-poly benchmark |

---

## 1. Boundary generation

Both the ground-truth and the SIREN-predicted boundary are drawn as the **zero level set of a
scalar field sampled on a square lattice**, extracted by `matplotlib.axes.Axes.contour(...,
levels=[0.0])`. Nothing is traced parametrically and marching-squares is never called directly.

### 1.1 Lattice construction

[`src/visualization/siren_encoder.py`](src/visualization/siren_encoder.py):

```python
def _grid(resolution, scale):
    axis = np.linspace(-scale, scale, resolution)
    return np.meshgrid(axis, axis, indexing="xy")
```

The field arrays themselves are built with the `"ij"` convention, so `field[i, j]` is the
value at $(x = \mathrm{axis}[j],\; y = \mathrm{axis}[i])$
([`scripts/plot_siren_encoder.py`](scripts/plot_siren_encoder.py)):

```python
axis = torch.linspace(-cfg.scale, cfg.scale, args.resolution)
grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
lattice = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(device)
```

### 1.2 Ground-truth boundary $\{P(x) = 0\}$

`true_fields()` evaluates the polynomial **exactly** on the lattice through
`compute_poly_features_batched` → `evaluate_poly_batched`, i.e. the Vandermonde contraction

$$P(x, y) = \sum_{i=0}^{3}\sum_{j=0}^{3} C_{ij}\,\tilde{x}^{\,i}\,\tilde{y}^{\,j},
\qquad \tilde{x} = x/\text{scale},\ \tilde{y} = y/\text{scale}$$

reshaped to $(R, R)$ and contoured with **no smoothing** (`sigma = 0`) — an analytic cubic
already has a clean zero set. Style: black, `"--"`, `linewidth 2.0`, `zorder 3`.

### 1.3 SIREN predicted boundary $\{f_\theta(x, z) = 0\}$

`decode_fields()` runs $f_\theta(x/\text{scale}, z)$ over the same lattice in chunks of
`--chunk-size 65536`, then contours level `0.0` on a **blurred** copy:

```python
_draw_boundary(ax, pred_field, scale, pred_color, "-", 2.0, zorder=4, sigma=style.smooth_sigma)
```

`smooth_sigma = 2.0` (lattice cells) routes the field through `smooth_field`
([`src/visualization/diagnostics.py`](src/visualization/diagnostics.py)): a **separable
Gaussian blur**, kernel radius `ceil(3 sigma) = 6`, replicate padding, two `F.conv2d` passes
(vertical then horizontal).

> **Why:** the SIREN uses $w_0 = 30$, so it carries low-amplitude high-frequency ripple. Its
> raw sign flips many times inside a thin band around the boundary and `contour()` returns
> hundreds of disjoint fragments — *denser grids resolve more ripple and look worse, not
> better*. The filter is symmetric, so it does not bias where the crossing sits, and it
> affects **rendering only**: every reported IoU/mass is computed on the raw field.

### 1.4 All boundary-drawing sites

| Site | Grid | GT style | Prediction style |
| --- | --- | --- | --- |
| `siren_encoder.plot_encoder_panel` | 600² | black `--`, lw 2.0 | green solid, lw 2.0, $\sigma$ = 2 |
| `diagnostics.plot_functa_extraction` | 500² | `contourf` on `[-inf, 0]` + black lw 3.0 | lime, $\sigma$ = 2 |
| `diagnostics.plot_boundary_ablation_grid` | 400² | black lw 2.2 | lime, $\sigma$ = 2 |
| `feasibility.polynomial_grid` | 400² | teal, dashes `(0,(7,7))`, lw 1.6, alpha 0.6 | — |
| `scatter.visualize_single_step` | 200² | red `dashed`, lw 2.5 | — |

The feasibility boundary is deliberately **thin, widely dashed and semi-transparent**: an
opaque line drawn on the boundary would hide the one-bin-wide ECI mass pile-up sitting
exactly underneath it.

---

## 2. Point counts

### 2.1 SIREN / CAVIA meta-training

[`scripts/train_functa.py`](scripts/train_functa.py):

```
points_per_shape = 1000       # spatial query points per constraint
batch_size       = 16         # constraints (tasks) per outer step
steps_per_epoch  = 400        # 6,400 shapes per epoch
epochs           = 3000       # early stop: patience 250, min_delta 1e-4
inner_steps      = 15         inner_lr = 1e-2
outer_lr         = 1e-4       lambda_z = 1e-4   (L2 on the context vector)
latent_dim = hidden_dim = 512,  n_layers = 4,  w0 = 30.0
```

**1,000 spatial points per shape**, i.e. 16,000 points per outer optimization step,
**resampled fresh at every step** by `sample_query_points`. With
`FUNCTA_QUERY_GMM_FRACTION = 0.0` these are uniform draws over $[-4.5, 4.5]^2$
(`torch.rand * 2*scale - scale`). Targets are $\tanh(P(x))$. Rejection sampling of the
constraints is backed by a fixed 10,000-point GMM proxy set. The holdout is 100
polynomials × 1,000 points, generated once.

### 2.2 Deployment-time extraction

`ExtractionConfig` in [`src/experiment/config.py`](src/experiment/config.py) mirrors the
training budget exactly:

```python
points_per_shape = 1000
steps            = 15
lr               = 6.25e-4      # = 1e-2 / 16
query_gmm_fraction = 0.0
```

The learning rate differs from the meta-training `1e-2` **by construction**:
`extract_latents_batched` reduces as `((preds - Y)**2).mean(dim=1).sum()` (chunk-invariant),
whereas meta-training used `F.mse_loss` with a mean over `(batch 16, points)`. The
equivalent per-shape step is therefore `1e-2 / 16`. CAVIA initialisations are only optimal
at their trained step size; this is the fix for the historical extraction step-size bug.

### 2.3 The 1D profile graph (`profile_range_quantile`)

Implemented in [`src/visualization/feasibility.py`](src/visualization/feasibility.py) as the
optional second row of `plot_feasibility_row(..., show_profile=True)`.

**Input points:** the plotted sample cloud — `--num-samples`, default **100,000** per panel
(`--metric-samples`, default 10,000, is used only for the captions).

**Pipeline:**

1. `signed_boundary_distance(points, coeffs)` maps each 2D sample to one scalar (§2.4).
2. `_profile_edges` pools the distances of **all** panels and takes
   `np.quantile(pooled, 0.001)` and `np.quantile(pooled, 0.999)` as the range, padded by 2%,
   then `np.linspace(lo, hi, profile_bins + 1)` with `profile_bins = 200`.
3. `_profile_data` histograms into those shared edges and normalizes to a density:
   `counts / (distance.size * bin_width)` — divided by **every** sample, not just the
   in-range ones, so mass outside the axis shows as missing area rather than being silently
   redistributed.
4. `_draw_profile` renders a `step`/`fill_between` pair, a dashed vertical line at $x = 0$,
   and an annotation box reporting the **wall fraction** $\Pr(|d| < 0.02)$.

Two invariants the figure depends on:

- **Shared bin edges and shared y-limit across panels** (`profile_share_y = True`,
  `profile_headroom = 1.35`). Without a shared y-axis a 40× spike and a mild bump render
  identically. The limit is set from panel 0 (ground truth); a panel whose peak exceeds
  `1.05 * top` is annotated `"peak N, clipped"`.
- **A quantile range, not the full range.** Explained in §2.4.

### 2.4 What $P(x)/\lVert \nabla P(x)\rVert$ means

```python
def signed_boundary_distance(points, coeffs, degree, scale):
    pts = torch.as_tensor(...).requires_grad_(True)
    P = evaluate_poly(compute_poly_features(pts, degree, scale), C)
    grad = torch.autograd.grad(P.sum(), pts)[0]
    return (P / grad.norm(dim=1).clamp(min=1e-6)).detach().cpu().numpy()
```

**The problem it solves.** The raw value $P(x)$ tells you *which side* of the boundary a
point is on, but not *how far*. Its magnitude is arbitrary: the coefficient matrices are
normalized only in Frobenius norm (`C / ||C||_F`), so scaling $P$ by any constant leaves the
constraint set $\{P \le 0\}$ unchanged while stretching every $P$ value. Raw $P$ is therefore
not comparable between two constraints, and a histogram of it would not be interpretable.

**The construction.** First-order Taylor expansion of $P$ around a point $x$ near the
boundary, with $x^\star$ the nearest boundary point:

$$0 = P(x^\star) \approx P(x) + \nabla P(x)^\top (x^\star - x)$$

The displacement $x^\star - x$ is, to first order, along $\nabla P(x)$ (the gradient is
normal to the level set). Writing $x^\star - x = -t\,\nabla P(x)/\lVert\nabla P(x)\rVert$
and solving gives $t = P(x)/\lVert\nabla P(x)\rVert$, so

$$d(x) \;=\; \frac{P(x)}{\lVert \nabla P(x) \rVert} \;\approx\; \pm\,\mathrm{dist}\big(x, \{P = 0\}\big)$$

This is the standard **first-order signed distance** (the same normalization used to turn any
implicit function into an approximate SDF). Term by term:

| Term | Meaning |
| --- | --- |
| $P(x)$ | the constraint value; $\le 0$ inside the feasible region, $> 0$ outside |
| $\nabla P(x) = (\partial_x P, \partial_y P)$ | the 2D gradient, obtained by `torch.autograd.grad`, not finite differences |
| $\lVert \nabla P(x) \rVert$ | how fast $P$ changes per unit of *length* — the local conversion factor from "units of $P$" to "metres" |
| $d(x)$ | signed distance in **plane units**, the same units as the $x$/$y$ axes |

**Reading the graph.** $d < 0$ is inside the feasible region, $d > 0$ is outside, $d = 0$ is
the boundary. Because it is now a genuine length, the same axis is meaningful across
different constraints and different samplers, and `wall_tolerance = 0.02` means literally
"within 0.02 plane units of the boundary".

**Why the 0.999 quantile is required.** The approximation is only valid *near* the boundary
and only where the gradient is non-degenerate. A cubic has critical points where
$\nabla P \approx 0$; a sample landing there produces a huge or effectively unbounded $d$
(the `clamp(min=1e-6)` prevents a division by zero but not a large quotient). A handful of
such points would otherwise set the x-axis for every panel and compress the entire
distribution into one bin. Taking the central 99.8% of the *pooled* distances discards those
degenerate outliers while keeping every panel on one common axis.

**What the row proves.** The true truncated density steps down to zero at $d = 0$. A
projection- or guidance-based sampler instead shows a narrow spike immediately to the left of
zero — mass parked *on* the constraint surface. That is the "wall effect" stated numerically:
for validation polynomial 86, **51.6% of ECI mass lies within 0.02 of the boundary** (profile
peak 83 vs. ground truth 1.8), against 0.2% for Functa and 0.7% for the ground truth. This is
also why the density maps use `np.histogram2d` + `imshow` and **never a KDE**: a Gaussian
kernel of any bandwidth spreads that spike back into the interior and erases the evidence.

---

## 3. Metric computation

Implemented in [`src/metrics/distributional.py`](src/metrics/distributional.py),
[`src/metrics/success_rates.py`](src/metrics/success_rates.py) and
[`src/metrics/likelihood.py`](src/metrics/likelihood.py); orchestrated by
`evaluate_single_configuration` in [`src/inference/evaluator.py`](src/inference/evaluator.py).

### 3.0 Notation and sample sizes

| Symbol | Meaning |
| --- | --- |
| $X = \{x_i\}_{i=1}^{n}$ | **generated** samples for one constraint, $x_i \in \mathbb{R}^2$ |
| $Y = \{y_j\}_{j=1}^{m}$ | **ground-truth** samples for the same constraint |
| $C$ | the $4 \times 4$ coefficient matrix of the constraint |
| $\Omega_C = \{x : P_C(x) \le 0\}$ | the feasible region |
| $\text{mass}$ | $\Pr_{p_{\mathrm{gmm}}}[x \in \Omega_C]$, estimated by Monte Carlo |
| $p_{\text{true}}$ | $p_{\mathrm{gmm}}$ truncated to $\Omega_C$, i.e. $p_{\mathrm{gmm}}(x)/\text{mass}$ |
| $p_{\text{model}}$ | the density the flow matcher induces at $t = 1$ |

| Quantity | Legacy 100-poly (`eval_fm.py`) | v1k (`eval_val1k.py`) |
| --- | --- | --- |
| Generated $n$ | `evaluation.num_x0 = 10000` | `--num-x0 10000` |
| GT pool | `gmm_pool_size = 100000` | `--gmm-pool-size 100000`, seeded |
| GT used $m$ | filtered pool: $100{,}000 \times \text{mass}$ (≈ 2k–95k, varies per constraint) | **exactly 10,000**, rejection-resampled |
| MMD internal cap | 5,000 per side | 5,000 per side |
| NLL/KLD points | 5,000 frozen shared | 5,000 |

The legacy asymmetry matters: $n = 10{,}000$ is fixed but $m$ scales with the constraint's
mass, because `filter_true_samples` keeps the subset of the 100k GMM pool with $P(x) \le 0$.
The v1k path removed this by rejection-sampling GT up to exactly `num_x0`.

### 3.1 Success Rate — *feasibility*

**Measures:** the fraction of generated samples that actually satisfy the constraint. Pure
feasibility; it says nothing about *where* inside the region the mass sits.

$$\mathrm{SR} \;=\; \frac{100}{n} \sum_{i=1}^{n} \mathbb{1}\!\left[P_C(x_i) \le 0\right]$$

- Computed on **all $n = 10{,}000$** generated points; the GT side is not involved.
- Range $[0, 100]$, higher is better.

### 3.2 SWD — Sliced Wasserstein Distance — *geometric transport cost*

**Measures:** how far the generated cloud must be moved to become the ground-truth cloud,
averaged over 1D projections. Sensitive to shape, position and spread; a cheap surrogate for
the full optimal-transport cost.

$$\widehat{\mathrm{SWD}}(X, Y) \;=\; \frac{1}{L}\sum_{\ell=1}^{L} W_2^2\!\left(\theta_\ell^\top X,\; \theta_\ell^\top Y\right),
\qquad \theta_\ell \sim \mathrm{Unif}(\mathbb{S}^1)$$

with the 1D Wasserstein computed in closed form from sorted order statistics:

$$W_2^2(a, b) \;=\; \int_0^1 \left| F_a^{-1}(q) - F_b^{-1}(q) \right|^2 dq$$

- $\theta_\ell$ — a random unit direction; projecting reduces the 2D problem to a 1D one that
  has an exact sorting solution.
- $F_a^{-1}$ — the quantile function of the projected samples.
- $L = 50$ (`num_projections=50`). Implementation: `ot.sliced_wasserstein_distance(X, Y,
  n_projections=50)`.
- Uses the **full** $X$ (10k) and $Y$; unequal $n \ne m$ is handled by POT.
- `_finite_rows` drops any row with a non-finite coordinate first; an empty side returns
  `inf`.
- **The projection directions come from the unseeded global NumPy RNG** — see §6.
- Range $[0, \infty)$, lower is better.

### 3.3 MMD — Maximum Mean Discrepancy — *kernel two-sample discrepancy*

**Measures:** the distance between the two clouds' mean embeddings in the RKHS of an RBF
kernel. Zero iff the distributions match (for a characteristic kernel); sensitive to
differences at the kernel's length scale.

$$\widehat{\mathrm{MMD}}^2(X, Y) \;=\; \frac{1}{n^2}\sum_{i,i'} k(x_i, x_{i'}) \;+\; \frac{1}{m^2}\sum_{j,j'} k(y_j, y_{j'}) \;-\; \frac{2}{nm}\sum_{i,j} k(x_i, y_j)$$

$$k(a, b) \;=\; \exp\!\left(-\gamma \lVert a - b \rVert_2^2\right), \qquad \gamma = 1.0$$

- $k$ — RBF (Gaussian) kernel; $\gamma = 1$ fixes the length scale at ~1 plane unit, so
  structure much finer than that is invisible to this metric.
- First two terms: average self-similarity within each cloud. Third: cross-similarity. The
  combination is $\lVert \mu_X - \mu_Y \rVert^2_{\mathcal{H}}$ for the RKHS mean embeddings.
- This is the **biased V-statistic** (diagonal terms $k(x_i, x_i) = 1$ included), so the
  estimate can go slightly negative; the code clamps with `max(0.0, ...)`.
- Both sides are **subsampled to `max_pts = 5000`** with `np.random.choice(..., replace=False)`
  to bound the three dense $N \times N$ Gram matrices built by
  `sklearn.metrics.pairwise.rbf_kernel`. Effective sizes: 5,000 vs 5,000.
- Range $[0, \infty)$, lower is better. Spans ~4 decades between the GT noise floor and a
  projection sampler, hence the `"{:.1e}"` caption format.

### 3.4 JSD — Jensen–Shannon Divergence — *density overlap*

**Measures:** symmetric, bounded disagreement between the two *densities* on a common grid.
Unlike SWD/MMD it is computed from smoothed density estimates, so it penalises getting the
shape of the density wrong even when the support is correct.

$$\mathrm{JSD}(p \Vert q) \;=\; \tfrac{1}{2}\,\mathrm{KL}\!\left(p \,\Vert\, \tfrac{p+q}{2}\right) + \tfrac{1}{2}\,\mathrm{KL}\!\left(q \,\Vert\, \tfrac{p+q}{2}\right)$$

evaluated on discretised densities $p_g$, $q_g$ over grid cells $g$:

$$\mathrm{KL}(p \Vert r) = \sum_g p_g \log \frac{p_g}{r_g}$$

Pipeline in `compute_jsd`:

1. Bounding box = joint min/max of $X$ and $Y$ per axis, padded by $\pm 0.5$.
2. `np.mgrid[x_min:x_max:100j, y_min:y_max:100j]` → **100 × 100 = 10,000** evaluation
   positions (`grid_size=100`).
3. `scipy.stats.gaussian_kde` fitted **separately** on $X^\top$ and $Y^\top$ (Scott's rule
   bandwidth per cloud — the bandwidths are *not* shared), each evaluated on all 10,000
   positions.
4. Each vector normalized to sum 1, making them discrete PMFs over cells (the cell area
   cancels).
5. `scipy.spatial.distance.jensenshannon(Z_gen, Z_true)` returns the JS **distance**
   $\sqrt{\mathrm{JSD}}$; the code **squares it** to report the divergence, in nats (natural
   log base).

- Guards: fewer than 10 points on either side → `inf`; a `LinAlgError`/`ValueError` from a
  rank-deficient covariance (fully collapsed cloud) → `inf`.
- Uses the **full** sample sets, no subsampling.
- Range $[0, \log 2] \approx [0, 0.693]$, lower is better.

### 3.5 NLL — Negative Log-Likelihood — *density assigned to real data*

**Measures:** how much probability mass the learned density puts on genuine
constraint-satisfying points. Unlike the three metrics above it scores the *density*
directly, so a model that covers the right region with the wrong shape cannot hide.

$$\mathrm{NLL} \;=\; -\frac{1}{N}\sum_{k=1}^{N} \log p_{\text{model}}(x_k), \qquad x_k \sim p_{\text{true}}$$

computed by the instantaneous change-of-variables formula along the probability-flow ODE:

$$\log p_{\text{model}}(x_1) \;=\; \log \mathcal{N}(x_0; 0, I) \;-\; \int_{1}^{0} \operatorname{Tr}\!\left(\nabla_x v_t(x_t)\right) dt$$

- $x_1$ — the data point; $x_0$ — its preimage under the backward flow.
- $\log \mathcal{N}(x_0; 0, I)$ — the standard 2D Gaussian prior log-density, supplied as
  `Independent(Normal(0, 1), 1).log_prob`.
- $\operatorname{Tr}(\nabla_x v_t)$ — the divergence of the velocity field, i.e. the
  instantaneous log-volume change. In 2D the **exact** trace costs two backward passes per
  step, so no Hutchinson estimator is used (`exact_divergence=True`).
- Integrated by `ODESolver.compute_likelihood`, `method="midpoint"`, `step_size = 0.05`,
  `chunk_size = 4000`.
- $N = 5000$ **frozen, shared** points (§6.4). Non-finite log-probs are dropped.
- Defined **only for `coeff` and `functa`.** ECI and HardFlow move state outside the
  probability-flow ODE (projection / guidance steps), so change-of-variables no longer
  describes their density. GT's is the reference itself.
- Lower is better, but the value is **not comparable across constraints**: a small feasible
  region concentrates the same unit of probability into less area, lowering NLL for free.
  Use KLD to compare.

> **Historical bug (fixed).** `ConstrainedFlowMatcher.forward` computed the SIREN feature
> under `torch.no_grad()`. Values were unchanged, but $\partial\,\mathrm{SIREN}/\partial x_t$
> was zeroed, so the exact-divergence trace was not the trace of the integrated field. With
> $w_0 = 30$ the dropped Jacobian block is $O(10\text{–}100)\times$ the retained one near the
> boundary. Fixing it moved the functa KLD from 0.3753 to 0.0229 and NLL from 3.2190 to
> 2.8714. Samples are unaffected, so SR/SWD/MMD/JSD did not move.

### 3.6 KLD — *NLL made comparable across constraints*

**Measures:** the excess NLL over the best achievable value, i.e. a Monte Carlo estimate of
$\mathrm{KL}(p_{\text{true}} \Vert p_{\text{model}}) \ge 0$. Subtracting the reference
entropy removes the mass-dependent offset, making the number averageable over a benchmark
spanning very different region sizes.

$$\mathrm{KLD} \;=\; \underbrace{-\frac{1}{N}\sum_k \log p_{\text{model}}(x_k)}_{\mathrm{NLL}} \;-\; \underbrace{\left(-\frac{1}{N}\sum_k \log p_{\text{true}}(x_k)\right)}_{\text{ideal NLL} \;=\; H(p_{\text{true}})}$$

with the truncated reference density

$$\log p_{\text{true}}(x) \;=\; \log p_{\mathrm{gmm}}(x) \;-\; \log(\text{mass})$$

- $\log p_{\mathrm{gmm}}$ — exact, from `MixtureSameFamily.log_prob`.
- $\text{mass}$ — the constraint's GMM probability mass; dividing by it renormalizes the GMM
  to the feasible region. Without this term the reference entropy would be misattributed to
  the model.
- Both terms use **the same $N = 5000$ points and the same finite mask**, so the subtraction
  is exact per point.
- $\ge 0$ in expectation, lower is better. It can dip slightly below zero because
  $\text{mass}$ is itself estimated from a finite pool. Binomial noise on `mass` is small
  (±0.005 nats), but an unresolved audit found MC pool mass and $3000^2$ grid quadrature
  disagree by ~10% median — suspected to be a truncation artifact of the quadrature box
  $[-4.5, 4.5]^2$ cutting the GMM tails.

### 3.7 Which metric answers which question

| Question | Metric |
| --- | --- |
| Are the samples legal? | Success Rate |
| Are they in the right *place*? | SWD |
| Do they match at a ~1-unit length scale? | MMD |
| Is the *shape of the density* right? | JSD |
| Does the model assign real data high probability? | NLL |
| ... corrected for region size, comparable across constraints? | KLD |
| Is mass piled on the constraint surface? | §2.3 profile / wall fraction |

The headline result of the feasibility-vs-fidelity figure is precisely that these decouple:
SR is 100% for GT/ECI/HardFlow and 99.1% for Functa, while JSD is 0.0015 / 0.1450 / 0.0924 /
0.0031. **Feasibility saturates; fidelity does not.**

---

## 4. Grid resolution

| Purpose | Resolution | Defined in |
| --- | --- | --- |
| Encoder heatmap + decoded level set | **600 × 600** (`--resolution`) | `plot_siren_encoder.py` |
| Latent interpolation strips | **600 × 600**, same lattice | `plot_siren_encoder.py` |
| Functa extraction figure | **500 × 500** (`--resolution`) | `plot_functa_extraction.py` |
| Boundary ablation grid | **400 × 400** | `diagnostics.plot_boundary_ablation_grid` |
| Feasibility boundary contour | **400 × 400** (`boundary_resolution`) | `feasibility.FeasibilityStyle` |
| Feasibility density map | **180 × 180 bins** (`bins`) | `feasibility.FeasibilityStyle` |
| Feasibility 1D profile | **200 bins** (`profile_bins`) | `feasibility.FeasibilityStyle` |
| Believed-region fields | **200 × 200** (`iou_grid_size`) | `EvalConfig` |
| Likelihood heatmap | **200 × 200** (`likelihood_grid`) | `EvalConfig` |
| JSD KDE evaluation grid | **100 × 100** (`grid_size`) | `compute_jsd` |
| Legacy scatter overlay | 200² contour, 300² hist2d bins | `scatter.visualize_single_step` |
| GMM reference density | 200 × 200 | `gmm_target.compute_gmm_density` |

All lattices span $[-4.5, 4.5]^2$ except the JSD grid, which spans the joint bounding box of
the two clouds padded by $\pm 0.5$.

**Rendering choices:**

- Heatmaps use `imshow(..., extent=(-scale, scale, -scale, scale), origin="lower")`.
- `interpolation="bilinear"` for encoder fields — smooth gradient, band-free, small PDF.
- `interpolation="nearest"` for feasibility densities — so a one-bin-wide wall renders
  exactly one bin wide. `imshow` is used over `hist2d` so empty bins are masked and the
  boundary line underneath stays visible where no sample landed.
- Feasibility density is **per-point density**, `H / (N * bin_area)`, so panels drawn from
  different sample counts share a colour scale.
- Colour ceiling: `np.quantile(H[H > 0], 0.995)` over **non-zero bins only** (most bins are
  empty and would skew the scale), taken from the reference panel (`vmax_mode="reference"`)
  so every panel is scaled by the ground truth.
- Encoder colour limits come from `field_limits`, symmetric about 0 so the sign of $f_\theta$
  — the inside/outside decision — maps to the colormap's own midpoint. Interpolation strips
  lock one `(vmin, vmax)` across the whole strip so colour means the same thing in every
  panel and the level set is the only thing that moves.

---

## 5. Solver mechanics

`run_evaluation_inference` in [`src/inference/evaluator.py`](src/inference/evaluator.py):

```python
solver = ODESolver(velocity_model=WrappedModel(model))     # flow_matching.solver
samples_chunk = solver.sample(
    time_grid=torch.linspace(0, 1, int(1 / step_size)).to(device),
    x_init=x0_chunk,
    method='midpoint',
    step_size=step_size,                                   # 0.05
    return_intermediates=False,
    **chunk_kwargs)                                        # z / coeffs / bounds
```

| Setting | Value | Note |
| --- | --- | --- |
| Method | `'midpoint'` | **fixed-step** torchdiffeq solver |
| Step size | `EvalConfig.step_size = 0.05` | 20 steps over $t \in [0, 1]$ |
| NFE | ≈ **40** | midpoint costs 2 velocity evaluations per step |
| `atol` / `rtol` | **never passed** | dead knobs: a fixed-grid solver ignores them |
| Batch chunk | `batch_size = 100000` rows | under `torch.inference_mode()` |
| NLL solver | same method + `step_size` | `nll_step_size = ev.step_size` |

Accuracy is controlled **solely by `step_size`**. The step-size ladder in
`audit_likelihood.py` (0.05 / 0.02 / 0.01 / 0.005) confirmed the reported KLD is stable under
refinement, i.e. the measured effects are not discretisation artifacts.

**`time_grid` caveat.** `int(1/0.05) = 20` points spanning $[0, 1]$, so the grid spacing is
$1/19 \approx 0.0526 \ne$ `step_size`. This grid only sets *output* times — torchdiffeq
marches its own uniform 0.05 grid and interpolates onto the requested times. With
`return_intermediates=False` only $t = 1$ survives, so it has no effect on reported samples.

**Conditioning.** The condition tensor is expanded to $(C \cdot N, d)$ and integrated jointly
with `x0` expanded the same way, then reshaped to $(C, N, 2)$.

**ECI / HardFlow do not use this path.** They run explicit Euler with `--steps DEFAULT_STEPS`,
`--correction-loops 1`, `--projection-iters 16`, `--guidance-scale 100.0` and a constraint
margin — see [`src/inference/constrained_samplers.py`](src/inference/constrained_samplers.py).
Because they move state outside the ODE, their samples are not the pushforward of a
probability flow, which is why NLL/KLD are undefined for them (§3.5).

---

## 6. Reproducibility

### 6.1 Global seeding

`set_seed` in [`src/experiment/runtime.py`](src/experiment/runtime.py) sets `random`,
`np.random`, `torch.manual_seed` and `torch.cuda.manual_seed_all`. Called once at the top of
each script: `eval_fm` uses `EvalConfig.seed = 0`, training uses `TrainConfig.seed = 0`, the
figure scripts use `--seed 0`.

### 6.2 Monte Carlo polynomial generation

`sample_valid_polynomials` in [`src/datasets/constraints.py`](src/datasets/constraints.py):

1. Draw `torch.randn(needed * 2, degree+1, degree+1)` candidates (2× oversample to cut loop
   iterations).
2. Evaluate against a **10,000-point GMM proxy set** with
   `einsum('ni,bij,nj->bn', proxy_x_pow, C, proxy_y_pow)`.
3. Accept where the mass ratio $\in$ `[min_area, max_area]` — `[0.05, 0.95]` for training
   pools, `[0.1, 0.9]` for the legacy validation set.
4. Loop until `batch_size` accepted, then normalize each $C$ by its Frobenius norm.

**This function takes no seed argument.** It consumes the global RNG, and the loop count is
data-dependent, so downstream RNG consumption varies run to run. That is why every
reproducible artifact below is **cached rather than re-seeded**.

### 6.3 Base distribution and the frozen validation set

[`src/datasets/validation.py`](src/datasets/validation.py). `get_validation_set` loads
`benchmark/validation_set.pt` if it exists and only regenerates on a cache miss. Contents:

- `polynomials` — 100 constraints, area ratio in `[0.1, 0.9]`
- `x0` — `torch.randn(10000, 2)`, the **source noise**
- `x1` — 100,000 GMM draws
- `bboxes` — 100 axis-aligned boxes (legacy bbox experiments)

Every method and every constraint therefore starts the ODE from the **identical 10,000 base
samples**, `val_set["x0"][:num_x0]`. Differences between methods are model differences, not a
different draw of noise.

The v1k set (`benchmark/validation_set_v1k.pt`) is 1000 constraints stratified into 20
equal-width mass bins over `[0.02, 0.98]`, MC mass from a 1M GMM pool, deterministically
**shuffled** at build time so every shard spans the mass range and costs the same.

### 6.4 Frozen NLL/KLD evaluation points

[`src/metrics/eval_points.py`](src/metrics/eval_points.py):

```python
NLL_EVAL_SEED        = 42
NLL_EVAL_POOL_SIZE   = 100_000       # 1_000_000 for v1k
NLL_EVAL_MAX_POINTS  = 5000
```

A CPU-drawn GMM pool under a forked RNG, rejection-filtered per polynomial, then
`deterministic_subset(valid, 5000, seed + i)` using a dedicated `torch.Generator` — the
per-polynomial seed means adding a polynomial never perturbs the points of the others. The
result is device-independent, atomically written, digest-checked against the polynomial set,
and reproduced bit-for-bit after a cache deletion. `load_nll_eval_set(num_points=k)` returns
a prefix, so a smaller $k$ is still a shared set.

The per-constraint `mass` is cached alongside the points, because KLD subtracts
$\log p_{\mathrm{gmm}} - \log(\text{mass})$; sharing the points but letting each script
re-estimate `mass` from its own pool would leave a per-model offset in the KLD.

v1k needs the 1M pool: at mass 0.02 a 100k pool yields only ~2000 valid points, below
`NLL_EVAL_MAX_POINTS`.

### 6.5 v1k per-index seed streams

[`scripts/eval_val1k.py`](scripts/eval_val1k.py). The benchmark runs as a 20 × 50 job array,
so every stochastic element is keyed to the **global constraint index**, never to shard
position, using `torch.random.fork_rng`:

```python
REFERENCE_POOL_SEED = 20_000   # one fixed reference GMM pool for all shards
GT_SAMPLE_SEED      = 30_000   # + index*100 + attempt, per rejection batch
QUERY_POINT_SEED    = 40_000   # + index, CAVIA query coordinates
METRIC_SEED         = 50_000   # + index, via seed_metric_rng()
```

**`seed_metric_rng(index)` exists specifically because `compute_mmd`'s `np.random.choice`
subsample and `compute_swd`'s random projections are otherwise unseeded.** Without it a
constraint's score would depend on which shard evaluated it — shard invariance is the core
correctness property of the pipeline.

This applies to the v1k path only. The legacy `eval_fm.py` path seeds once at startup, so its
per-constraint MMD/SWD draws depend on evaluation order.

### 6.6 Run identity and provenance

Run id is `<name>-<sha8>` where the fingerprint covers degree, scale, SIREN architecture
**and the checkpoint file digest**, plus the extraction, pool, FM and train blocks. The
`evaluation` block is deliberately **excluded**, so re-evaluating a checkpoint never forks a
new run. `pin_once()` additionally strips `start_idx`/`end_idx` from the v1k fingerprint and
reuses an existing `provenance.json` run id, otherwise 20 array tasks would race to write 20
different run ids.

`artifacts.write_manifest` records the git commit and every array shape alongside the `.npy`
files, and `plot_run.py` / `--plot-only` redraw every figure from those arrays with no
checkpoint and no ODE solve.
