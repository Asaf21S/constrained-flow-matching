import torch
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime


from constrained_fm.src.geometry.polynomials import compute_poly_features
from constrained_fm.src.metrics.distributional import compute_swd, compute_mmd, compute_jsd, filter_true_samples
from constrained_fm.src.consts import EVALUATION_RESULTS_PATH


def evaluate_distributional_metrics_batched(samples_gen_batched, x_true_pool, bounds=None, coeffs=None):
    """
    Evaluates SWD, MMD, and JSD for batched constraints.

    Parameters:
    samples_gen_batched: Tensor of shape (C, N, 2)
    x_true_pool: Tensor of shape (M, 2) containing all unconstrained ground truth points
    bounds: List of bounds [[x1,y1,x2,y2], ...] of length C
    coeffs: Tensor of shape (C, D+1, D+1)
    """
    return_single = False
    if samples_gen_batched.ndim == 2:
        return_single = True
        samples_gen_batched = samples_gen_batched.unsqueeze(0)

    C_dim = samples_gen_batched.shape[0]

    results = {
        "swd": [],
        "mmd": [],
        "jsd": []
    }

    if coeffs is not None:
        x_pow_true, y_pow_true = compute_poly_features(x_true_pool)
    else:
        x_pow_true, y_pow_true = None, None

    for i in tqdm(range(C_dim), desc="Distributional Metrics Evaluation"):
        samples_gen_single = samples_gen_batched[i]
        current_bounds = bounds[i] if bounds is not None else None
        current_coeffs = coeffs[i] if coeffs is not None else None
        x_true_filtered = filter_true_samples(x_true_pool, bounds=current_bounds, coeffs=current_coeffs,
                                              x_pow=x_pow_true, y_pow=y_pow_true)

        results["swd"].append(compute_swd(samples_gen_single, x_true_filtered))
        results["mmd"].append(compute_mmd(samples_gen_single, x_true_filtered))
        results["jsd"].append(compute_jsd(samples_gen_single, x_true_filtered))

    if return_single:
        return {k: v[0] for k, v in results.items()}

    return results


def log_evaluation_metrics(metrics_dict: dict, note: str, eval_type: str = "unconstrained",
                           path: str = EVALUATION_RESULTS_PATH):
    """
    Appends evaluation metrics to a JSON log file with a timestamp and tracking note.

    Parameters:
    -----------
    metrics_dict: dict
        The dictionary containing your metrics (e.g., {'success_rate': [...], 'swd': [...]})
    note: str
        A short description of what changed in this run (e.g., "Increased ResBlocks to 5")
    eval_type: str
        The type of evaluation (e.g., "unconstrained", "bbox", "polynomial")
    path: str
        Path to the JSON log file.
    """
    summary_stats = {}
    raw_data = {}

    for key, val in metrics_dict.items():
        if isinstance(val, (np.ndarray, torch.Tensor)):
            val = val.tolist()

        raw_data[key] = val

        if isinstance(val, list) and len(val) > 0:
            summary_stats[f"{key}_median"] = float(np.median(val))
            summary_stats[f"{key}_mean"] = float(np.mean(val))
        elif isinstance(val, (int, float)):
            summary_stats[key] = float(val)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_type": eval_type,
        "note": note,
        "summary": summary_stats,
        "raw_metrics": raw_data
    }

    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.append(log_entry)

    with open(path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Logged {eval_type} metrics to '{path}'")
    print(f"   Note: {note}")
    for k, v in summary_stats.items():
        if "median" in k or not any(x in k for x in ["median", "mean"]):
            print(f"   - {k}: {v:.4f}")


def load_logged_metrics(path: str = EVALUATION_RESULTS_PATH, entry_index: int = -1):
    """
    Loads a specific run from the JSON log.

    Parameters:
    -----------
    path: str
        Path to the JSON log file.
    entry_index: int
        Which log entry to load. Defaults to -1 (the most recent run).
    """
    if not os.path.exists(path):
        print(f"Error: Log file not found at '{path}'")
        return None

    with open(path, "r") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print("Error: Could not decode JSON. File might be empty or corrupted.")
            return None

    if len(history) == 0:
        print("Log file is empty.")
        return None

    try:
        entry = history[entry_index]
    except IndexError:
        print(f"Error: Index {entry_index} out of bounds. The log only has {len(history)} entries.")
        return None

    print(f"Loaded Run: {entry.get('timestamp', 'Unknown Time')}")
    print(f"Evaluation Type: {entry.get('eval_type', 'N/A')}")
    print(f"Note: {entry.get('note', 'No note provided')}")

    metrics_dict = entry.get("raw_metrics", {})

    if not metrics_dict:
        print("Error: No raw metrics found in this log entry.")
        return None

    return metrics_dict


def print_readme_metrics_table(metrics_dict: dict):
    """
    Computes summary statistics from a metrics dictionary and prints
    a formatted Markdown table ready for a README file.
    Handles both batched lists (constrained) and single floats (unconstrained).
    """
    lines = [
        "| Metric | Median / Value | Mean | Worst 5% | Target |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    def clean_data(data_list):
        arr = np.array(data_list, dtype=float)
        return arr[~np.isinf(arr)]

    if 'success_rate' in metrics_dict:
        sr_data = metrics_dict['success_rate']

        if isinstance(sr_data, (list, np.ndarray)) and len(sr_data) > 0:
            clean_sr = clean_data(sr_data)
            if len(clean_sr) > 0:
                lines.append(
                    f"| **Success Rate (%)** | {np.median(clean_sr):.2f} | {np.mean(clean_sr):.2f} | {np.percentile(clean_sr, 5):.2f} | *Higher is better* |")

        elif isinstance(sr_data, (float, int)):
            lines.append(f"| **Success Rate (%)** | {float(sr_data):.2f} | - | - | *Higher is better* |")

    dist_metrics = [
        ('swd', 'Sliced Wasserstein (SWD)'),
        ('mmd', 'Mean Discrepancy (MMD)'),
        ('jsd', 'Jensen-Shannon (JSD)')
    ]

    for key, name in dist_metrics:
        if key in metrics_dict:
            raw_data = metrics_dict[key]

            if isinstance(raw_data, (list, np.ndarray)) and len(raw_data) > 0:
                clean_arr = clean_data(raw_data)
                if len(clean_arr) > 0:
                    lines.append(
                        f"| **{name}** | {np.median(clean_arr):.4f} | {np.mean(clean_arr):.4f} | {np.percentile(clean_arr, 95):.4f} | *Lower is better* |")

            elif isinstance(raw_data, (float, int)):
                if not np.isinf(raw_data):
                    lines.append(f"| **{name}** | {float(raw_data):.4f} | - | - | *Lower is better* |")

    markdown_table = "\n".join(lines)
    print(markdown_table)
