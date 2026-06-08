import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

from constrained_fm.src.consts import GMM_MEANS, GMM_COVS, GMM_WEIGHTS


def get_points(batch_size, means=GMM_MEANS, covs=GMM_COVS, weights=GMM_WEIGHTS, device=None):
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


def compute_gmm_density(means=GMM_MEANS, covs=GMM_COVS, weights=GMM_WEIGHTS, grid_size=200, device=None):
    means = torch.as_tensor(means, device=device)
    covs = torch.as_tensor(covs, device=device)
    weights = torch.as_tensor(weights, device=device)

    mix = Categorical(weights)
    comp = MultivariateNormal(means, covs)
    gmm = MixtureSameFamily(mix, comp)

    x_grid = torch.meshgrid(torch.linspace(-4.5, 4.5, grid_size, device=device),
                            torch.linspace(-4.5, 4.5, grid_size, device=device),
                            indexing='ij')

    grid_points = torch.stack([x_grid[0].flatten(), x_grid[1].flatten()], dim=1)

    with torch.no_grad():
        log_prob = gmm.log_prob(grid_points)
        density = torch.exp(log_prob)

    return density