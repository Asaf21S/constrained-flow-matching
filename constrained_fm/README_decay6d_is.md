# Conditional Flow Matching and Importance Sampling for a 6D Decay

This experiment tests whether a flow trained to sample directly inside a box on one particle's
momentum can estimate observables of the other particle more efficiently than rejection sampling.
It also checks that importance sampling (IS) corrects errors in the learned conditional sampler.

## Contents

1. [Problem and target](#problem-and-target)
2. [Evaluation boxes](#evaluation-boxes)
3. [Models and estimators](#models-and-estimators)
4. [Training and evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Interpretation and limitations](#interpretation-and-limitations)
7. [Reproduction and artifacts](#reproduction-and-artifacts)

## Problem and target

The event is a two-body decay with a common random boost. For mass $M$, split fraction $u$,
unit direction $\hat v$, and boost $\vec P$,

$$
\vec p_1 = u(M\hat v + \vec P), \qquad
\vec p_2 = (1-u)(-M\hat v + \vec P).
$$

The parameters are $M\sim\mathcal N(1,0.05^2)$, $\vec P\sim\mathcal N(0,0.1^2 I_3)$,
$u\sim\mathcal U(0.1,0.9)$, and $\hat v$ uniform on the sphere. The six-dimensional target
is the joint distribution of $(\vec p_1,\vec p_2)$. The constraint is an axis-aligned box
$\mathcal B$ on $\vec p_1$; the requested quantity is

$$
\mathbb E[f(\vec p_2)\mid \vec p_1\in\mathcal B].
$$

In the simulator, the direction is sampled in [decay6d.py](src/problems/decay6d.py#L85):
draw a three-component standard-normal vector $g$ and set $\hat v=g/\lVert g\rVert$.
For example, $g=(3,4,0)$ gives $\hat v=(0.6,0.8,0)$, while $g=(1,1,1)$ gives
$\hat v\approx(0.577,0.577,0.577)$. These illustrate the normalization; the actual components
are random Gaussian draws. This construction gives a uniform direction on the sphere's surface.

The three observables are $\lVert\vec p_2\rVert$, $p_{2z}$, and the tail indicator
$\mathbb 1[\lVert\vec p_2\rVert>\tau_{\mathcal B}]$. Here $\tau_{\mathcal B}$ is a cutoff
chosen separately for each box: 95% of true events inside that box have a smaller
$\lVert\vec p_2\rVert$, and about 5% have a larger one. It is not a universal physics constant;
it makes the tail observable similarly rare in all five boxes.

The target density formula contains the hidden split fraction $u$. For a given observed pair
$(\vec p_1,\vec p_2)$, the density is found by adding up the contributions from all possible
values of $u$ between 0.1 and 0.9. For a fixed $u$, the code recovers
$\vec a=M\hat v=\tfrac12(\vec p_1/u-\vec p_2/(1-u))$ and
$\vec P=\tfrac12(\vec p_1/u+\vec p_2/(1-u))$. It evaluates the density of $\vec a$
(whose value at radius $r=\lVert\vec a\rVert$ is
$[\phi_M(r)+\phi_M(-r)]/(4\pi r^2)$), multiplies by the 3D Gaussian density of $\vec P$,
and divides by the change-of-variables Jacobian $8u^3(1-u)^3$. Here $\phi_M$ is the Gaussian
mass density; both signs appear because either signed mass along a uniform direction can produce
the same vector radius. This gives the joint density contribution $p(x\mid u)$ for that split.

The marginal likelihood is the average of those contributions over the uniform split:
$p(x)=\frac{1}{0.8}\int_{0.1}^{0.9}p(x\mid u)\,du$. Numerically, the code uses 4096
equally spaced $u$ values and normalized trapezoid weights. If there are $K=4096$ grid points,
the two endpoints each get weight $1/[2(K-1)]$ and each interior point gets
$1/(K-1)$. The weights sum to one, so the weighted sum approximates the *uniform average*
over $u$, not the unnormalized integral. The code combines the weighted density contributions
in log space. In this report, “exact density” means this physics-based calculation rather than
a learned neural density; the one-dimensional integral is still numerical.

The physical coupling explains why conditioning on $\vec p_1$ changes $\vec p_2$:

![Dataset structure: momentum split, back-to-back components, and opening angle](images/thesis_pool/decay6d_is/dataset/dataset_structure.png)

## Evaluation boxes

Five fixed test boxes target different probabilities and geometries. They were selected by
bisecting a scale against a large simulated pool; their final probabilities were independently
measured from ground-truth samples.

| box | target mass | measured mass | centre of $\vec p_1$ | half-widths | $\tau_{\mathcal B}$ |
| :--- | ---: | ---: | :--- | :--- | ---: |
| small_offcentre | 0.5% | 0.501% | $(0.45,0.45,0.35)$ | $(0.153,0.153,0.153)$ | 0.461 |
| thin_slab | 2% | 1.996% | $(0,0,0)$ | $(0.139,0.139,0.035)$ | 1.055 |
| offcentre_cube | 5% | 4.997% | $(-0.35,0.25,-0.20)$ | $(0.232,0.232,0.232)$ | 0.848 |
| elongated | 10% | 10.018% | $(0.30,0,0)$ | $(0.395,0.132,0.132)$ | 1.028 |
| bulk_cube | 30% | 30.012% | $(0,0,0)$ | $(0.278,0.278,0.278)$ | 0.979 |

The panels below show three projections of the $\vec p_1$ distribution and the five box outlines.
In each panel a box is a two-dimensional projection: points can lie inside the drawn rectangle
but outside the box along the omitted axis.

![The particle-1 dataset and projected test boxes](images/thesis_pool/decay6d_is/dataset/dataset_p1_boxes.png)

The boxes induce quite different $\vec p_2$ distributions. The top row compares
$\lVert\vec p_2\rVert$ for all events and in-box events. All five panels share one vertical
density scale, so their heights can be compared directly. The bottom row shows the in-box
$(p_{2x},p_{2z})$ density.

![Particle-2 distributions induced by each evaluation box](images/thesis_pool/decay6d_is/dataset/dataset_p2_given_box.png)

Ground-truth conditional means are listed below. Tail means vary slightly around
0.05 because the cutoff is estimated from a finite pool.

| box | $\mathbb E[\lVert\vec p_2\rVert]$ | $\mathbb E[p_{2z}]$ | tail probability |
| :--- | ---: | ---: | ---: |
| small_offcentre | 0.29728 | -0.13535 | 0.04950 |
| thin_slab | 0.89792 | 0.00000 | 0.04995 |
| offcentre_cube | 0.58085 | 0.19881 | 0.05011 |
| elongated | 0.82484 | 0.00006 | 0.05016 |
| bulk_cube | 0.78744 | 0.00001 | 0.05012 |

## Models and estimators

The conditional flow $q(\vec p_1,\vec p_2\mid\mathcal B)$ receives the box centre and the
logarithm of each half-width. A half-width is the distance from the box centre to a face; its
logarithm keeps the value positive when converted back and makes narrow and wide boxes easier
to represent on one numerical scale. The model is trained on simulator events conditioned to
lie inside the box. Its samples are not perfectly confined, so its measured leakage is reported. The unconstrained flow
$p_{\mathrm{uncon}}$ models the full six-dimensional target.

For a proposal sample $x\sim q$, the importance weight is

$$
w(x)=\frac{p(x)\,\mathbb 1[x_1\in\mathcal B]}{q(x\mid\mathcal B)}.
$$

The conditional expectation is estimated by the self-normalized ratio
$\sum_i w_i f_i/\sum_i w_i$. The average weight $N^{-1}\sum_i w_i$ independently estimates
$P(\mathcal B)$ and is a normalization check.

The seven reported estimators are:

| label | estimator | plain-language meaning |
| :--- | :--- | :--- |
| (a) | `q_raw` | Average over all $q$ samples, including samples that leaked outside the box. |
| (b) | `q_filtered` | Average over only the $q$ samples inside the box. |
| (c) | `is_learned` | IS using the learned unconstrained density $p_{\mathrm{uncon}}$. |
| (d) | `is_exact` | IS using the exact quadrature density; this is the density-corrected reference. |
| (e1) | `rej_equal_n` | Rejection sampling from $p_{\mathrm{uncon}}$ with the same number of draws. |
| (e2) | `rej_equal_time` | Rejection sampling with the number of draws adjusted to the same measured GPU time. |
| (e3) | `rej_equal_nfe` | Rejection sampling with the number of draws adjusted to the same number of network evaluations. |

For (d), each sample from the box flow gets a weight using the physics-based density calculated
by the $u$-quadrature, rather than the learned unconstrained flow. This corrects the box flow's
sampling errors using the reference density. It is a comparison standard for (c), not a claim
that the numerical quadrature has zero error.

In (e3), a “network evaluation” means one call to the neural network to calculate its velocity
at a solver step. An ODE solver may call the network several times per step, so NFE (“number of
function evaluations”) counts actual calls, not just the number of steps. Equal NFE compares
methods at roughly equal neural-network compute, even when their number of generated samples
differs.

Methods (a) and (b) test the conditional flow without density correction. Methods (c) and (d)
test importance sampling with learned and quadrature densities, respectively. Rejection is
included as a direct baseline and is necessarily wasteful for small boxes.

## Training and evaluation

Both flows use a CondOT flow-matching objective, standard-normal prior, hidden width 1024,
three residual blocks, batch size 4096, Adam with learning rate $10^{-3}$ and cosine decay to
$10^{-5}$. The conditional model trained for 100,000 steps; the unconstrained model trained
for 500,000 steps. Both use global gradient clipping at 1.0 and EMA weights (decay 0.9999)
for evaluation.

| run | run id | steps | diagnostic |
| :--- | :--- | ---: | :--- |
| unconstrained flow | `decay6d_uncon-cb36fa75` | 500,000 | $D_{KL}(p\Vert p_{uncon})=0.00043\pm0.00062$ |
| box-conditioned flow | `decay6d_box-cdea812c` | 100,000 | mean in-box rate 98.74%; range 97.64%–99.52% |
| box benchmark | `decay6d_boxes-519394b0` | — | fixed boxes and ground truth |
| final evaluation | `decay6d_is-f07211bc` | — | 20 repetitions at each $N\in\{10^3,10^4,10^5\}$ |

Here $N$ is the number of samples used to make ONE estimate. For each box, each proposal method
forms 20 estimates from non-overlapping groups of $N$ samples. At $N=100{,}000$, that consumes
2 million samples per box across the 20 estimates; four shards of 500,000 samples provide them.
But each individual estimate still uses 100,000 samples, not 2 million. The smaller-$N$ estimates
use smaller groups from the same stored proposal draws, so results at different $N$ are not
independent of one another.

Rejection uses 80 million candidate draws from the unconstrained model in total, shared across
all five boxes. These are draws before filtering, not 80 million accepted in-box events. For each
rejection comparison, the number of candidates in one repetition is set to $N$ for equal draw
count, or increased to match measured time or NFE. The same candidate stream can be checked
against each box. The estimate is calculated only from candidates that land inside that box.
In the final rerun, all three rejection budgets have 20 finite estimates for every box,
observable, and $N$. Raw $q$ can still
have fewer than 20 finite estimates because it deliberately includes leaked samples, including
rare non-finite fallback outputs. At $N=100{,}000$, raw $q$ has 14--20 valid estimates depending
on box and observable; filtered $q$ and both IS estimators have 20.

The probability-flow ODE uses adaptive dopri5 in float64 with
exact divergence for density evaluation, absolute and relative tolerances $10^{-5}$, and a
max-over-batch error norm. A small number of underflowing trajectories are isolated and retried
with 10,000 fixed RK4 steps; fallback counts are included in the evaluation metrics. The final
run used fallback for 12 of 10 million box-proposal trajectories, 20 backward-density solves
(8 in-box), and 60 of 80 million unconstrained draws.

The simulator and density checks passed before the full run: analytic moment errors were below
$3.7\times10^{-4}$ relative, changing the quadrature resolution four-fold changed its result by
$1.3\times10^{-4}$, the PIT/KS statistic was 0.0056 versus a 0.0096 critical value, and
$\mathbb E_g[p/g]=0.9996\pm0.0006$.

The first 100,000-step unconstrained training was not reliable: a large loss spike occurred
near step 20,000, and its final KL was 0.083. With clipping, EMA, and 500,000 steps, the final
KL fell to 0.00043. The box model also improved, from 96.6% average in-box rate to 98.7%.

## Results

### Accuracy at $N=100{,}000$

For each box, estimator, observable, and $N$, the code first has up to 20 estimates
$\hat\mu_r$, each based on $N$ samples. It computes that box's RMSE across repetitions,
$\sqrt{\mathrm{mean}_r[(\hat\mu_r-\mu_{\rm gt})^2]}$. The table then reports the median of
those five box-level RMSEs, in physical momentum units (tail-probability RMSE is unitless).
It does not pool the samples or calculate an RMSE over $5\times20\times N$ events. The `q_raw`
median conceals an extreme outlier and is qualified below; do not interpret it as uniformly
reliable.

In the detailed [per-box tables](baselines/decay6d_is/eval/tables.md), each pool of $N$ points
produces one estimate, not one RMSE. For each observable, the table reports standard deviation
of absolute error across the valid repetitions and RMSE across those errors. Slash-separated
values are ordered by $N=10^3/10^4/10^5$.

The evaluation also computes the standard deviation of repeated estimates within each box; it
is recorded separately as `std` in `metrics.json`. It is not shown beside the across-box median
RMSE because these are different summaries: RMSE includes both bias and repetition spread,
while `std` measures the spread around the repetition mean. The detailed tables show absolute-
error summaries per box and keep RMSE as a separate column. They also report valid repetition
counts per observable; in this run raw $q$ has some invalid estimates, while filtered $q$, IS,
and rejection have 20.

### Reading the Cost Columns

Each per-box table reports cost means only; draw budgets and density-evaluation counts are fixed
per repetition. `q draws` is the number sampled from the box-conditioned proposal;
`$p_{uncon}$ draws` is the number sampled from the unconstrained model for rejection.
`learned / exact density evals` counts target-density evaluations made by learned- or
exact-density IS. `samples used` is the mean number entering the estimate: all proposals for raw
$q$, in-box proposals for filtered $q$ and IS, and accepted events for rejection. The heading's
`GT events` is the number of simulator-generated benchmark events inside that box used to compute
the ground-truth reference; it is not a per-estimator sample budget. The benchmark generated one
billion target events total.

`NFE` counts calls to the ODE velocity network. Proposal generation tracks density, learned IS
adds a backward learned-density solve, exact IS adds quadrature time but no extra network NFEs,
and rejection uses unconstrained sampling only. Density solves also compute divergence
derivatives that are not extra NFEs, but are included in wall time. The time column reports
seconds for the sampling and density work used by that estimator. Timings and NFEs were recorded
per 10,000-sample ODE chunk; if a repetition uses only part of a recorded chunk, its cost is
prorated by sample count and is an estimate rather than a separately timed solver run.

Equal-time and equal-NFE rejection budgets are approximate matches, not exact per-repetition
equalities. Their draw counts are chosen from the median costs of 10,000-sample chunks and
rounded to 1,000-draw minibatches. The tables show mean realized costs across repetitions;
adaptive solver work and rare fallback trajectories can make those means differ from the
median-calibrated IS target. At $N=100{,}000$, equal-time windows contain 1.4--1.9 million
draws; the observed 60 fallbacks in 80 million draws imply about 1.1--1.5 fallback trajectories
per such window on average. This can raise realized time above the median-chunk target, while
even a few expensive trajectories can substantially raise mean NFE.

| estimator | $\lVert\vec p_2\rVert$ RMSE | $p_{2z}$ RMSE | tail RMSE |
| :--- | ---: | ---: | ---: |
| (a) raw $q$ | 0.00080* | 0.00175* | 0.00082 |
| (b) filtered $q$ | 0.00097 | 0.00161 | 0.00086 |
| (c) learned-$p$ IS | 0.00044 | 0.00102 | 0.00087 |
| (d) exact-$p$ IS | 0.00038 | 0.00083 | 0.00086 |
| (e1) rejection, equal $N$ | 0.00229 | 0.00365 | 0.00360 |
| (e2) rejection, equal time | 0.00052 | 0.00096 | 0.00080 |
| (e3) rejection, equal NFE | 0.00120 | 0.00252 | 0.00210 |

*For raw $q$, the elongated-box RMSE is $6.25\times10^{124}$ for $|\vec p_2|$ and
$5.39\times10^{124}$ for $p_{2z}$. One sample in that box required the RK4 fallback; its
extreme value dominates the ordinary averages. The median shown above is across five boxes and
therefore hides the failure. The filtered estimator removes leaked samples and has ordinary
errors in every box.*

The per-observable error curves show how errors change with sample budget. Exact and learned IS
track one another closely; rejection improves as its draw budget grows, but is much noisier at
equal $N$.

![RMSE and estimates versus sample count for p2 norm](images/thesis_pool/decay6d_is/error_vs_n_p2_norm.png)

![RMSE and estimates versus sample count for p2z](images/thesis_pool/decay6d_is/error_vs_n_p2_z.png)

![RMSE and estimates versus sample count for the tail indicator](images/thesis_pool/decay6d_is/error_vs_n_p2_tail.png)

ESS (“effective sample size”) summarizes how evenly the importance weight is spread. Its
formula is $\mathrm{ESS}=(\sum_i w_i)^2/\sum_i w_i^2$. If all weights are equal, ESS equals the
sample count $N$; if a few samples carry nearly all the weight, ESS is much smaller. The plots
show ESS divided by $N$, so values near 1 are healthy and values near 0 indicate that many draws
contribute little useful information.

In `weight_histograms`, each panel shows the distribution of normalized weights for one box.
The horizontal axis is the weight divided by the average in-box weight, on a log scale: zero
means “average weight,” values to the right are heavier-than-average samples, and values to the
left are lighter. A narrow shape near zero means weights are similar; a long right tail means
some samples dominate. The vertical axis is histogram density, shown on a log scale so the
rare heavy weights remain visible.

At $N=100{,}000$, the learned and exact IS estimates of $P(\mathcal B)$ agree with each other
and the benchmark mass to within about 0.2% relative error in every box. Their effective
sample-size fractions are 0.929–0.979 for learned IS and 0.931–0.981 for exact IS. The high,
nearly identical ESS indicates that the box proposal and unconstrained density now match well
on the tested support.

![ESS fraction and maximum normalized weight versus sample count](images/thesis_pool/decay6d_is/ess_vs_n.png)

![Learned and exact importance-weight distributions](images/thesis_pool/decay6d_is/weight_histograms.png)

Cost ratios use median per-chunk timings so an occasional solver retry does not dominate the
comparison. NFE counts calls to the velocity network made by the ODE solver. During a density
solve, each such call also computes six derivatives to get the exact divergence; this extra work
is not counted as six NFEs. An unconstrained rejection draw only needs a forward velocity call,
without those derivatives. Thus time and NFE are not interchangeable: across boxes, IS costs
14.1--19.4 times as much wall-clock time per proposal sample, but 2.1--2.9 times as many
network calls, as unconstrained rejection draws. At equal
sample count, IS is substantially more accurate than rejection. At equal wall time, rejection
is competitive and is slightly better on the median $|\vec p_2|$ RMSE; therefore the result
does not show a universal wall-clock advantage for IS at these model sizes. However, the
equal-time rejection comparison has very few usable momentum estimates at $N=100{,}000$, so
that apparent tie should be treated as provisional rather than a strong conclusion.

### Marginal distributions

Each panel compares the ground truth, the raw $q$ samples, and the same $q$ samples reweighted
with learned or exact $p$. The density maps make the conditional shape differences visible.
The plotting range is centered on the ground-truth quantiles, so the extreme raw-$q$ fallback
sample is not visible in the elongated marginal; use the RMSE warning above when reading it.

![Particle-2 marginals for small_offcentre](images/thesis_pool/decay6d_is/p2_marginals_box0_small_offcentre.png)

![Particle-2 marginals for thin_slab](images/thesis_pool/decay6d_is/p2_marginals_box1_thin_slab.png)

![Particle-2 marginals for offcentre_cube](images/thesis_pool/decay6d_is/p2_marginals_box2_offcentre_cube.png)

![Particle-2 marginals for elongated](images/thesis_pool/decay6d_is/p2_marginals_box3_elongated.png)

![Particle-2 marginals for bulk_cube](images/thesis_pool/decay6d_is/p2_marginals_box4_bulk_cube.png)

## Interpretation and limitations

1. **Importance sampling is validated.** Exact-density IS recovers the reference observables,
   gets box mass right, and has stable weights. Learned-density IS is nearly indistinguishable
   from exact-density IS after retraining.
2. **The unconstrained flow was the original bottleneck.** The first model had KL 0.083 and
   badly misestimated the thin-slab and elongated box masses. Retraining reduced KL to 0.00043;
   learned IS now gives accurate box masses and observable means.
3. **The conditional proposal is accurate but not numerically flawless.** Leakage is only
   0.22%–1.76%, and filtering it gives stable direct estimates. However, rare adaptive-solver
   underflows require fixed-step fallbacks. One fallback sample creates an astronomically large
   raw-$q$ estimate in the elongated box. That raw estimator and its plotted range should not be
   used as evidence of sampler quality until the fallback trajectory is made stable or its
   contribution is otherwise validated.
4. **The efficiency result depends on the budget.** IS clearly beats rejection at equal $N$;
   at equal wall time, rejection is comparable because an unconstrained sample is much cheaper.
   The current experiment establishes accuracy and sample-efficiency, not a blanket runtime win.

## Reproduction and artifacts

All jobs ran on the DLC2 cluster in the local PyTorch container. Final successful jobs were:

| stage | job |
| :--- | ---: |
| simulator/density validation | 274833 |
| build full evaluation boxes | 274870 |
| train unconstrained flow (500k) | 274973 |
| train box-conditioned flow (100k) | 274981 |
| dataset overview figures | 274972 |
| evaluation retry array | 275140 |
| merge metrics and result figures | 275141 |

The evaluation retry was needed because the first adaptive-solver recovery only handled
backward density solves; forward conditional sampling and unconstrained sampling also
underflowed. The 19 affected tasks were rerun with sample isolation, while successful shards
were reused. A subsequent merge compatibility issue with the older shard format was fixed
before the final successful merge.

From the project root, the main commands are:

```bash
sbatch scripts/run_decay6d_check.sh
sbatch scripts/run_decay6d_boxes.sh
sbatch scripts/run_decay6d_train.sh --mode uncon --iterations 500000
sbatch scripts/run_decay6d_train.sh --mode box
sbatch --dependency=afterok:<boxes>:<uncon>:<box> scripts/run_decay6d_eval.sh
sbatch --dependency=afterok:<eval> scripts/run_decay6d_merge.sh
sbatch scripts/run_decay6d_dataset.sh
```

The final metrics are in [`baselines/decay6d_is/eval/metrics.json`](baselines/decay6d_is/eval/metrics.json),
with run configuration and provenance beside them. Benchmark box definitions and ground-truth
means are in [`baselines/decay6d_is/benchmark/boxes.json`](baselines/decay6d_is/benchmark/boxes.json).
The earlier 100k unconstrained-model comparison is retained in
[`baselines/decay6d_is/eval_v1_uncon100k/metrics.json`](baselines/decay6d_is/eval_v1_uncon100k/metrics.json),
and the corresponding plots are in `images/thesis_pool/decay6d_is/v1_uncon100k/`.

Intermediate per-task shards and model checkpoints are intentionally not part of this report;
the aggregate metrics, run metadata, plotting manifests, and final figures are the committed
results.