import torch
from torch.distributions import Independent, Normal
from tqdm.auto import tqdm
from flow_matching.solver import ODESolver

from constrained_fm.src.solvers.ode_wrapper import WrappedModel
from constrained_fm.src.datasets.gmm_target import compute_gmm_log_likelihood


def estimate_mass_importance_sampling(model, box, num_points=50000, step_size=0.05, eval_batch_size=4000, device=None):
    """ Estimates the probability mass P(c) using Importance Sampling. """
    # Generate q(x|c)
    x_generated = model.sample(
        num_points=num_points,
        bounds=box,
        step_size=step_size,
        device=device
    )

    # Get likelihood for q(x|c)
    bounds_tensor = torch.tensor([box], dtype=torch.float32, device=device).expand(num_points, 4)
    gaussian_log_density = Independent(Normal(torch.zeros(2, device=device), torch.ones(2, device=device)), 1).log_prob

    log_q_x_list = []

    wrapped_vf = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_vf)

    with torch.no_grad():
        for i in tqdm(range(0, num_points, eval_batch_size), desc="IS Likelihood Computation"):
            x_1_chunk = x_generated[i:i + eval_batch_size]
            bounds_chunk = bounds_tensor[i:i + eval_batch_size]

            _, chunk_log_p = solver.compute_likelihood(
                x_1=x_1_chunk,
                method='midpoint',
                step_size=step_size,
                exact_divergence=True,
                log_p0=gaussian_log_density,
                bounds=bounds_chunk
            )
            log_q_x_list.append(chunk_log_p)

    log_q_x = torch.cat(log_q_x_list, dim=0)

    # Get GMM log-density p(x)
    log_p_x = compute_gmm_log_likelihood(x_generated, device=device)

    # Check if generated points fulfill constraint
    x_min, y_min, x_max, y_max = box
    in_x = (x_generated[:, 0] >= x_min) & (x_generated[:, 0] <= x_max)
    in_y = (x_generated[:, 1] >= y_min) & (x_generated[:, 1] <= y_max)
    indicator_mask = (in_x & in_y).float()

    # Compute IS estimate
    # IS Weight = p(x) / q(x|c)  --> log(Weight) = log_p(x) - log_q(x|c)
    log_weights = log_p_x - log_q_x

    # Exponentiate to get actual weights, and apply mask to kill points outside box
    weights_tensor = torch.exp(log_weights) * indicator_mask

    # The estimated mass is the average of the weights
    estimated_mass = weights_tensor.mean().item() * 100.0

    return estimated_mass
