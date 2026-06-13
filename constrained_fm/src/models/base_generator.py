import torch
import torch.nn as nn
from flow_matching.solver import ODESolver
from torch.distributions import Independent, Normal
from tqdm.auto import tqdm

from constrained_fm.src.models.wrapper import WrappedModel
from constrained_fm.src.utils.polynomials import compute_poly_features, evaluate_poly


class BaseFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, x, **kwargs):
        raise NotImplementedError("Child classes must implement the forward() method!")

    def sample(self, num_points: int, bounds=None, coeffs=None, step_size: float = 0.05, return_intermediates: bool = False, device=None):
        self.eval()

        wrapped_vf = WrappedModel(self)
        solver = ODESolver(velocity_model=wrapped_vf)

        x_init = torch.randn((num_points, 2), dtype=torch.float32, device=device)
        T = torch.linspace(0, 1, 10).to(device)

        kwargs = {}
        if bounds is not None:
            bounds_tensor = torch.tensor([bounds], dtype=torch.float32, device=device)
            kwargs['bounds'] = bounds_tensor.expand(num_points, 4)
        elif coeffs is not None:
            coeffs_flat = coeffs.view(1, -1)
            kwargs['coeffs'] = coeffs_flat.expand(num_points, -1)

        with torch.no_grad():
            samples = solver.sample(
                time_grid=T,
                x_init=x_init,
                method='midpoint',
                step_size=step_size,
                return_intermediates=return_intermediates,
                **kwargs
            )

        if return_intermediates:
            return samples, T
        else:
            print(samples.shape)
            return samples

    def compute_likelihood_grid(self, bounds=None, coeffs=None, degree=3, scale=4.0, grid_size=200, step_size=0.05,
                                eval_batch_size=4000, device=None):
        self.eval()

        wrapped_vf = WrappedModel(self)
        solver = ODESolver(velocity_model=wrapped_vf)

        x_grid = torch.meshgrid(torch.linspace(-4.5, 4.5, grid_size),
                                torch.linspace(-4.5, 4.5, grid_size),
                                indexing='ij')

        x_1 = torch.stack([x_grid[0].flatten(), x_grid[1].flatten()], dim=1).to(device)
        num_points = x_1.shape[0]

        gaussian_log_density = Independent(Normal(torch.zeros(2, device=device),
                                                  torch.ones(2, device=device)), 1).log_prob

        full_bounds_tensor = None
        full_coeffs_tensor = None

        if bounds is not None:
            absolute_bounds = torch.tensor([bounds], dtype=torch.float32, device=device)
            full_bounds_tensor = absolute_bounds.expand(num_points, 4)
        elif coeffs is not None:
            full_coeffs_tensor = coeffs.view(1, -1).expand(num_points, -1)

        exact_log_p_full = torch.full((num_points,), float('-inf'), device=device)
        valid_mask = torch.ones(num_points, dtype=torch.bool, device=device)

        if bounds is not None:
            x_min, y_min, x_max, y_max = bounds
            valid_mask = (x_1[:, 0] >= x_min) & (x_1[:, 0] <= x_max) & \
                         (x_1[:, 1] >= y_min) & (x_1[:, 1] <= y_max)
        elif coeffs is not None:
            x_pow, y_pow = compute_poly_features(x_1, degree=degree, scale=scale)
            C_batch = coeffs.unsqueeze(0).expand(num_points, -1, -1)
            p_vals = evaluate_poly(x_pow, y_pow, C_batch).squeeze()
            valid_mask = p_vals <= 0

        valid_indices = torch.nonzero(valid_mask).squeeze()
        x_1_valid = x_1[valid_mask]
        num_valid_points = x_1_valid.shape[0]

        if full_bounds_tensor is not None:
            full_bounds_tensor = full_bounds_tensor[valid_mask]
        if full_coeffs_tensor is not None:
            full_coeffs_tensor = full_coeffs_tensor[valid_mask]

        exact_log_p_list = []
        print(f"Computing exact divergence for {num_valid_points} valid points in chunks of {eval_batch_size}...")

        with torch.no_grad():
            for i in tqdm(range(0, num_valid_points, eval_batch_size)):
                x_1_chunk = x_1_valid[i:i + eval_batch_size]

                chunk_kwargs = {}
                if full_bounds_tensor is not None:
                    chunk_kwargs['bounds'] = full_bounds_tensor[i:i + eval_batch_size]
                if full_coeffs_tensor is not None:
                    chunk_kwargs['coeffs'] = full_coeffs_tensor[i:i + eval_batch_size]

                _, chunk_log_p = solver.compute_likelihood(
                    x_1=x_1_chunk,
                    method='midpoint',
                    step_size=step_size,
                    exact_divergence=True,
                    log_p0=gaussian_log_density,
                    **chunk_kwargs
                )

                exact_log_p_list.append(chunk_log_p)

        if len(exact_log_p_list) > 0:
            exact_log_p_valid = torch.cat(exact_log_p_list, dim=0)
            exact_log_p_full[valid_indices] = exact_log_p_valid

        likelihood = torch.exp(exact_log_p_full).cpu().reshape(grid_size, grid_size).numpy().T

        return likelihood
