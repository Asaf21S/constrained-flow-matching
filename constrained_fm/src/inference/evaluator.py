import torch
from flow_matching.solver import ODESolver
import numpy as np
from tqdm import tqdm

from constrained_fm.src.solvers.ode_wrapper import WrappedModel
from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE
from constrained_fm.src.metrics.distributional import compute_swd, compute_mmd, compute_jsd
from constrained_fm.src.metrics.success_rates import compute_success_rate_bbox, compute_success_rate_polynomial
from constrained_fm.src.metrics.distributional import filter_true_samples
from constrained_fm.src.geometry.polynomials import compute_poly_features


def run_evaluation_inference(model, x0, bounds=None, coeffs=None, step_size=0.05, batch_size=100000, device=None):
    model.eval()

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


def evaluate_single_configuration(samples, x_true_pool, bounds=None, coeffs=None, disjoint_boxes=None,
                                  degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, x_pow_true=None, y_pow_true=None,
                                  device=None):
    metrics = {
        "swd": float('inf'),
        "mmd": float('inf'),
        "jsd": float('inf')
    }

    if disjoint_boxes is not None:
        sample_success_mask = torch.zeros(samples.shape[0], dtype=torch.bool, device=device)
        true_pool_mask = torch.zeros(x_true_pool.shape[0], dtype=torch.bool, device=device)

        for i in range(disjoint_boxes.shape[0]):
            x_min, y_min, x_max, y_max = disjoint_boxes[i].tolist()

            # Mask Generated Samples
            s_in = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max) & (samples[:, 1] >= y_min) & (
                        samples[:, 1] <= y_max)
            sample_success_mask |= s_in

            # Mask True Ground Truth Pool
            t_in = (x_true_pool[:, 0] >= x_min) & (x_true_pool[:, 0] <= x_max) & (x_true_pool[:, 1] >= y_min) & (
                        x_true_pool[:, 1] <= y_max)
            true_pool_mask |= t_in

        metrics["success_rate"] = sample_success_mask.float().mean().item() * 100.0
        x_true_filtered = x_true_pool[true_pool_mask]

    else:
        x_true_filtered = filter_true_samples(x_true_pool, bounds=bounds, coeffs=coeffs, x_pow=x_pow_true,
                                              y_pow=y_pow_true)

        if bounds is not None:
            metrics["success_rate"] = compute_success_rate_bbox(samples, bounds)
        elif coeffs is not None:
            metrics["success_rate"] = compute_success_rate_polynomial(samples, coeffs, degree, scale, device)

    if len(x_true_filtered) > 10 and len(samples) > 10:
        metrics["swd"] = compute_swd(samples, x_true_filtered)
        metrics["mmd"] = compute_mmd(samples, x_true_filtered)
        metrics["jsd"] = compute_jsd(samples, x_true_filtered)
    else:
        print("Warning: Not enough valid points in true pool or generated samples to compute SWD/MMD/JSD.")

    return metrics


def evaluate_validation_set_metrics(val_samples_batched, x_true_pool, bounds=None, coeffs=None,
                                    degree=POLYNOMIAL_DEGREE, scale=PLANE_SCALE, device=None):
    return_single = False
    if isinstance(val_samples_batched, np.ndarray):
        val_samples_batched = torch.tensor(val_samples_batched, dtype=torch.float32, device=x_true_pool.device)

    if val_samples_batched.ndim == 2:
        return_single = True
        val_samples_batched = val_samples_batched.unsqueeze(0)

    C_dim = val_samples_batched.shape[0]

    if coeffs is not None:
        x_pow_true, y_pow_true = compute_poly_features(x_true_pool, degree=degree, scale=scale)
    else:
        x_pow_true, y_pow_true = None, None

    results = {
        "swd": [],
        "mmd": [],
        "jsd": [],
        "success_rate": []
    }

    for i in tqdm(range(C_dim), desc="Validation Set Evaluation"):
        samples_gen_single = val_samples_batched[i]
        current_bounds = bounds[i] if bounds is not None else None
        current_coeffs = coeffs[i] if coeffs is not None else None

        metrics_i = evaluate_single_configuration(
            samples=samples_gen_single,
            x_true_pool=x_true_pool,
            bounds=current_bounds,
            coeffs=current_coeffs,
            degree=degree,
            scale=scale,
            x_pow_true=x_pow_true,
            y_pow_true=y_pow_true,
            device=device
        )

        results["swd"].append(metrics_i["swd"])
        results["mmd"].append(metrics_i["mmd"])
        results["jsd"].append(metrics_i["jsd"])

        if "success_rate" in metrics_i:
            results["success_rate"].append(metrics_i["success_rate"])

    if not results["success_rate"]:
        del results["success_rate"]

    if return_single:
        return {k: v[0] for k, v in results.items()}

    return results
