import torch
from flow_matching.solver import ODESolver
import numpy as np
from tqdm import tqdm

from constrained_fm.src.solvers.ode_wrapper import WrappedModel


def run_evaluation_inference(model, x0, bounds=None, coeffs=None, step_size=0.05, batch_size=100000):
    model.eval()
    device = next(model.parameters()).device

    wrapped_vf = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_vf)

    num_samples = x0.shape[0]

    if bounds is not None:
        cond_t = torch.tensor(bounds, dtype=torch.float32, device=device)
        if cond_t.ndim == 1:
            cond_t = cond_t.unsqueeze(0)

        num_conditions = cond_t.shape[0]
        # Shape: [B, 4] -> [B, N, 4] -> [B * N, 4]
        cond_expanded = cond_t.unsqueeze(1).expand(num_conditions, num_samples, 4).reshape(-1, 4)
        kwarg_name = 'bounds'

    elif coeffs is not None:
        cond_t = torch.tensor(coeffs, dtype=torch.float32, device=device)
        if cond_t.ndim == 1:
            cond_t = cond_t.unsqueeze(0)
        elif cond_t.ndim == 3:  # Flatten [B, 4, 4] to [B, 16] if necessary
            cond_t = cond_t.reshape(cond_t.shape[0], -1)

        num_conditions = cond_t.shape[0]
        # Shape: [C, 16] -> [C, N, 16] -> [C * N, 16]
        cond_expanded = cond_t.unsqueeze(1).expand(num_conditions, num_samples, cond_t.shape[-1]).reshape(
            -1, cond_t.shape[-1])
        kwarg_name = 'coeffs'
    else:
        num_conditions = 1
        cond_expanded = None

    # Expand x0: [N, 2] -> [B, N, 2] -> [B * N, 2]
    x0_expanded = x0.unsqueeze(0).expand(num_conditions, num_samples, x0.shape[-1]).reshape(-1, x0.shape[-1])
    total_evaluations = x0_expanded.shape[0]

    final_samples_list = []

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Data chunk device: {x0_expanded.device}")

    with torch.inference_mode():
        for i in tqdm(range(0, total_evaluations, batch_size), desc="Evaluating Batches"):
            x0_chunk = x0_expanded[i:i + batch_size]

            chunk_kwargs = {}
            if cond_expanded is not None:
                chunk_kwargs[kwarg_name] = cond_expanded[i:i + batch_size]

            samples_chunk = solver.sample(
                time_grid=torch.linspace(0, 1, int(1 / step_size)).to(device),
                x_init=x0_chunk,
                method='midpoint',
                step_size=step_size,
                return_intermediates=False,
                **chunk_kwargs
            )

            # Handle output shape depending on your specific ODESolver implementation
            if isinstance(samples_chunk, torch.Tensor) and samples_chunk.ndim == 3:
                final_samples_list.append(samples_chunk[-1].cpu().numpy())
            else:
                final_samples_list.append(samples_chunk.cpu().numpy())

    final_samples_np = np.concatenate(final_samples_list, axis=0)
    final_samples_reshaped = final_samples_np.reshape(num_conditions, num_samples, -1)

    if num_conditions == 1:
        return final_samples_reshaped[0]

    return final_samples_reshaped
