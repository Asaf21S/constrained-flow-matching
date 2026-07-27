from __future__ import annotations

import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

from constrained_fm.src.consts import GMM_MEANS, GMM_COVS, GMM_WEIGHTS, PLANE_SCALE


def get_points(batch_size: int, means: list | torch.Tensor = GMM_MEANS, covs: list | torch.Tensor = GMM_COVS,
                weights: list | torch.Tensor = GMM_WEIGHTS,
                device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.tensor(means, device=device)
    covs = torch.tensor(covs, device=device)
    weights = torch.tensor(weights, device=device)

    num_components = means.shape[0]

    if weights is None:
        weights = torch.ones(num_components, device=device) / num_components

    mix = Categorical(weights)
    comp = MultivariateNormal(means, covs)

    labels = mix.sample((batch_size,))

    all_samples = comp.sample((batch_size,))

    batch_indices = torch.arange(batch_size, device=device)
    data = all_samples[batch_indices, labels]

    return data.float(), labels


def compute_gmm_density(means: list | torch.Tensor = GMM_MEANS, covs: list | torch.Tensor = GMM_COVS,
                         weights: list | torch.Tensor = GMM_WEIGHTS, grid_size: int = 200,
                         device: torch.device | str | None = None) -> torch.Tensor:
    means = torch.as_tensor(means, device=device)
    covs = torch.as_tensor(covs, device=device)
    weights = torch.as_tensor(weights, device=device)

    mix = Categorical(weights)
    comp = MultivariateNormal(means, covs)
    gmm = MixtureSameFamily(mix, comp)

    x_grid = torch.meshgrid(torch.linspace(-PLANE_SCALE, PLANE_SCALE, grid_size, device=device),
                            torch.linspace(-PLANE_SCALE, PLANE_SCALE, grid_size, device=device),
                            indexing='ij')

    grid_points = torch.stack([x_grid[0].flatten(), x_grid[1].flatten()], dim=1)

    with torch.no_grad():
        log_prob = gmm.log_prob(grid_points)
        density = torch.exp(log_prob)

    return density


def compute_gmm_log_likelihood(x: torch.Tensor, means: list | torch.Tensor = GMM_MEANS,
                                covs: list | torch.Tensor = GMM_COVS, weights: list | torch.Tensor = GMM_WEIGHTS,
                                device: torch.device | str | None = None) -> torch.Tensor:
    """
    Computes log p(x) for the target GMM.
    x: [N, 2]
    """
    means_t = torch.as_tensor(means, dtype=torch.float32, device=device)
    covs_t = torch.as_tensor(covs, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

    mix = Categorical(weights_t)
    comp = MultivariateNormal(means_t, covs_t)
    gmm = MixtureSameFamily(mix, comp)

    return gmm.log_prob(x)
