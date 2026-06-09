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
PLANE_SCALE = 4.0

VALIDATION_SET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark", "validation_set.pt"))
