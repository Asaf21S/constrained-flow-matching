import os

# Registered dataset/constraint problems, duplicated here so configs validate without torch.
DEFAULT_PROBLEM = "gmm_poly"
PROBLEM_NAMES = ("gmm_poly", "bump2d", "kinematics6d")


GMM_MEANS = [
    [-1.5, -1.5],
    [1.5, 2.0],
    [2.0, -1.5],
    [-0.5, 0.5]
]

GMM_COVS = [
    [[0.7,  0.0], [ 0.0, 0.7]],
    [[1.0, -0.6], [-0.6, 0.8]],
    [[0.3,  0.0], [ 0.0, 1.2]],
    [[1.2,  0.3], [ 0.3, 0.5]]
]

GMM_WEIGHTS = [0.35, 0.25, 0.15, 0.25]

POLYNOMIAL_DEGREE = 3
PLANE_SCALE = 4.5
POLY_MIN_AREA_RATIO = 0.05
POLY_MAX_AREA_RATIO = 0.95

# Fraction of CAVIA query points drawn from the GMM rather than uniformly over the plane.
# Must be identical in meta-training and at extraction time.
# 0.5 measurably degraded reconstruction: a degree-3 zero set is a global object, so
# clustering query points makes the fit ill-conditioned and starves low-mass constraints,
# whose valid regions lie in the GMM tail.
FUNCTA_QUERY_GMM_FRACTION = 0.0
VALIDATION_POLY_MIN_AREA_RATIO = 0.1
VALIDATION_POLY_MAX_AREA_RATIO = 0.9
VALIDATION_BBOX_WIDTH_RANGE = (1.0, 6.5)

VALIDATION_SET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark", "validation_set.pt"))
EVALUATION_RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "experiment_tracking_log.json"))

# --- v1k benchmark: 1000 polynomials stratified uniformly over GMM mass -------------------
# Kept separate from the 100-polynomial set above so previously reported numbers stay
# reproducible; select between them with `resolve_validation_set(name)`.
VAL1K_SET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark", "validation_set_v1k.pt"))
VAL1K_NUM_POLYS = 1000
VAL1K_MIN_MASS = 0.02
VAL1K_MAX_MASS = 0.98
VAL1K_MASS_BINS = 20
VAL1K_MC_POOL_SIZE = 1_000_000
VAL1K_NUM_X0 = 10000
VAL1K_SEED = 1000
VAL1K_VERSION = 1

# --- bench1k: the same stratified protocol, applied to bump2d and kinematics6d ------------
# One frozen file per problem rather than one shared file, because the two constraint
# families have nothing in common to serialise, and because rebuilding one must never
# invalidate the other's already-reported numbers.
BENCH1K_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark"))
BENCH1K_NUM_CONSTRAINTS = 1000
BENCH1K_MASS_BINS = 20
BENCH1K_MC_POOL_SIZE = 1_000_000
BENCH1K_NUM_X0 = 10000
BENCH1K_SEED = 1000
BENCH1K_VERSION = 1
# Held-out constraints the sampler hyperparameters are selected on; a disjoint seed, so no
# benchmark constraint or start point is ever seen during selection.
TUNE_NUM_CONSTRAINTS = 100
TUNE_NUM_X0 = 5000
TUNE_SEED = 7000

# --- bump2d: a 1% Gaussian signal buried in a falling exponential background ---------------
# The signal sits in the x1 bulk but the x2 tail. That split is deliberate: visibility in the
# 1D x1 marginal is set by the background density at mu_1 alone, while purity inside a polygon
# is set by the 2D background density, so placing mu_2 far out hides the bump in projection
# while keeping it recoverable once a constraint cuts on x2.
BUMP_DOMAIN = 10.0
BUMP_BACKGROUND_SCALES = (2.2, 1.5)
BUMP_SIGNAL_MEAN = (2.0, 6.5)
BUMP_SIGNAL_SIGMA = 0.35
BUMP_SIGNAL_WEIGHT = 0.01
BUMP_POLY_MIN_VERTICES = 3
BUMP_POLY_MAX_VERTICES = 7
BUMP_POLY_RADIUS_RANGE = (0.35, 7.0)
BUMP_POLY_MIN_MASS = 0.02
BUMP_POLY_MAX_MASS = 0.98

# The SIREN regresses tanh(C / tau), so tau sets the width of the transition band around the
# boundary in domain units. C has unit normals, so tau is a length. Swept over {0.3, 1.0, 3.0}
# at 120 epochs: mass-IoU p5 came out 0.867 / 0.921 / 0.882. Narrower is not better, because a
# w0=30 sine basis on [-1, 1] coordinates cannot resolve the ramp and overshoots it; wider fits
# to a lower MSE but places the zero level set less precisely, which is what the IoU sees.
# Meta-training and extraction must use the same value, or CAVIA adapts a latent for a
# function it was never trained on.
BUMP_SIREN_TAU = 1.0
# Unlike the GMM, the bump background is concentrated in one corner, so uniform query points
# spend the fixed 15-step budget resolving boundary far from any probability mass.
BUMP_QUERY_TARGET_FRACTION = 0.5
BUMP_SIREN_CHECKPOINT = "constrained_fm/functa_dataset/bump_siren_best.pt"
BUMP_POOL_PATH = "constrained_fm/functa_dataset/pools/bump_pool.pt"

# --- kinematics6d: two massless particles, constrained by their pair invariant mass --------
# Generated in (pT, eta, phi) because that is where the physics factorises, then mapped to the
# Cartesian momenta the flow matcher sees. The map is a diffeomorphism with |J| = pT^2 cosh eta,
# so the Cartesian log-density stays exact and KLD remains well defined.
KIN_PT_RANGE = (10.0, 500.0)
KIN_PT_SCALE = 40.0
KIN_ETA_RANGE = (-3.0, 3.0)
KIN_ETA_SIGMA = 1.5
# M^2 = 2 pT1 pT2 (cosh d_eta - cos d_phi) vanishes for collinear pairs, where d(sqrt)/d(M^2)
# is unbounded; the floor keeps HardFlow's guidance gradient finite there.
KIN_MASS_FLOOR = 1e-6
# Shell probability mass, the analogue of polygon mass: the fraction of pairs inside the
# window. Log-spaced because the interesting regime is the narrow end.
KIN_SHELL_MIN_MASS = 0.01
KIN_SHELL_MAX_MASS = 0.5
KIN_SHELL_MASS_BINS = 20
KIN_MC_POOL_SIZE = 1_000_000
# Median-heuristic RBF bandwidth for the normalised 6D frame, pinned rather than re-estimated
# so MMD stays comparable across samplers. The measured median squared distance is 7.66, well
# under the Gaussian 2 * dim = 12, because the normalised frame stays heavy-tailed along p_z.
# The 2D problems keep the legacy gamma = 1.0, degenerate here at exp(-7.66) ~ 5e-4.
KIN_MMD_GAMMA = 0.13
