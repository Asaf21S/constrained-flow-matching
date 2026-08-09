import os


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
FUNCTA_QUERY_GMM_FRACTION = 0.5
VALIDATION_POLY_MIN_AREA_RATIO = 0.1
VALIDATION_POLY_MAX_AREA_RATIO = 0.9
VALIDATION_BBOX_WIDTH_RANGE = (1.0, 6.5)

VALIDATION_SET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark", "validation_set.pt"))
EVALUATION_RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "experiment_tracking_log.json"))
