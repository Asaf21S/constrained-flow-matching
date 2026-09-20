import os

# Registered dataset/constraint problems, duplicated here so configs validate without torch.
DEFAULT_PROBLEM = "gmm_poly"
PROBLEM_NAMES = ("gmm_poly", "bump2d")


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
