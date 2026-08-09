import json
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np

from constrained_fm.src.consts import POLYNOMIAL_DEGREE, PLANE_SCALE, FUNCTA_QUERY_GMM_FRACTION
from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.datasets.functa_conditioning import sample_query_points
from constrained_fm.src.geometry.polynomials import compute_poly_features, compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.models.functa_siren import build_modulated_siren
from constrained_fm.src.datasets.gmm_target import get_points


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")

repo_root = Path(__file__).resolve().parents[1]
base_dir = repo_root / "functa_dataset"
checkpoint_dir = base_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

epochs = 3000
steps_per_epoch = 400  # 400 steps * 16 batch size = 6,400 shapes per epoch
batch_size = 16
points_per_shape = 1000

# Meta-Learning (CAVIA) Hyperparameters
outer_lr = 1e-4
inner_lr = 1e-2
inner_steps = 15  # matches the test-time extraction budget so the outer loop optimizes for it
lambda_z = 1e-4  # L2 penalty on the context vector

latent_dim = 512
hidden_dim = 512
n_layers = 4
w0 = 30.0
poly_degree = POLYNOMIAL_DEGREE
plane_scale = PLANE_SCALE
query_gmm_fraction = FUNCTA_QUERY_GMM_FRACTION  # must match every extraction call site

save_every = 50
patience = 250
min_delta = 1e-4


print("Precomputing proxy features for rejection sampling...")
proxy_x, _ = get_points(batch_size=10000, device=device)
proxy_x = proxy_x.to(device)
global_proxy_x_pow, global_proxy_y_pow = compute_poly_features(proxy_x, degree=poly_degree, scale=plane_scale)
global_proxy_x_pow = global_proxy_x_pow.to(device)
global_proxy_y_pow = global_proxy_y_pow.to(device)

def generate_batch(batch_size: int, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates an on-the-fly batch of polynomial constraints entirely in VRAM.

    Args:
        batch_size: number of independent polynomial constraint "shapes" (tasks).
        num_points: number of query points sampled per shape.

    Returns:
        X: (batch_size, num_points, 2) query coordinates in the canonical
            SIREN input domain [-1, 1]^2.
        Y: (batch_size, num_points) regression targets tanh(P(x, y)) in (-1, 1).
    """
    C = sample_valid_polynomials(
        batch_size=batch_size,
        degree=poly_degree,
        scale=plane_scale,
        proxy_x_pow=global_proxy_x_pow,
        proxy_y_pow=global_proxy_y_pow,
        device=device
    )

    X_raw = sample_query_points(batch_size, num_points, scale=plane_scale,
                                gmm_fraction=query_gmm_fraction, device=device)

    x_pow, y_pow = compute_poly_features_batched(X_raw, degree=poly_degree, scale=plane_scale)
    P_vals = evaluate_poly_batched(x_pow, y_pow, C)

    Y = torch.tanh(P_vals)
    return X_raw / plane_scale, Y


print("Generating fixed holdout set of 100 polynomials for validation...")
val_X, val_Y = generate_batch(batch_size=100, num_points=points_per_shape)

siren = build_modulated_siren(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, w0=w0).to(device)
optimizer = torch.optim.Adam(siren.parameters(), lr=outer_lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
mse_loss = nn.MSELoss()

loss_history = []
val_loss_history = []
best_val_loss = float('inf')
patience_counter = 0

for epoch in tqdm(range(1, epochs + 1), desc="Training CAVIA Functa"):
    siren.train()
    epoch_loss = 0.0

    for step in range(steps_per_epoch):
        X_batch, Y_batch = generate_batch(batch_size, points_per_shape)

        # Initialize zeroed z vectors. Requires gradients for autograd tracking.
        z = torch.zeros(batch_size, latent_dim, device=device, requires_grad=True)

        for _ in range(inner_steps):
            preds = siren(X_batch, z).squeeze(-1)
            loss_inner = mse_loss(preds, Y_batch)

            # Compute gradients of z with create_graph=True to allow outer loop backprop
            grad_z = torch.autograd.grad(loss_inner, z, create_graph=True)[0]

            # Manual SGD step for z
            z = z - inner_lr * grad_z

        optimizer.zero_grad(set_to_none=True)

        # Evaluate the adapted z on the task
        preds_adapted = siren(X_batch, z).squeeze(-1)

        # Compute final loss (MSE + L2 Regularization on z to prevent extreme modulation)
        loss_outer = mse_loss(preds_adapted, Y_batch) + lambda_z * (z ** 2).mean()

        # Backpropagate through the inner loop graph into the SIREN base weights
        loss_outer.backward()
        torch.nn.utils.clip_grad_norm_(siren.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss_outer.item()

    avg_loss = epoch_loss / steps_per_epoch
    loss_history.append(avg_loss)
    scheduler.step(avg_loss)

    if epoch % 10 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d} - Outer MSE+L2 Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

    if epoch % save_every == 0:
        print(f"\n--- Running Validation Inference (Epoch {epoch}) ---")
        siren.eval()

        # Initialize validation z vectors
        z_val = torch.zeros(val_X.shape[0], latent_dim, device=device, requires_grad=True)

        # SGD steps on z (inner_steps, matching training), mean-reduced loss to match inner_lr scale.
        for _ in range(inner_steps):
            preds_val = siren(val_X, z_val).squeeze(-1)
            loss_val = mse_loss(preds_val, val_Y)

            # Pure SGD step
            grad_z = torch.autograd.grad(loss_val, z_val)[0]
            z_val = z_val - inner_lr * grad_z

        with torch.no_grad():
            preds_final = siren(val_X, z_val).squeeze(-1)
            avg_val_loss = mse_loss(preds_final, val_Y).item()

        val_loss_history.append((epoch, avg_val_loss))
        print(f"Validation Extraction Loss (MSE): {avg_val_loss:.6f}\n")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(siren.state_dict(), base_dir / "siren_best.pt")
            with open(base_dir / "siren_best_meta.json", "w") as f:
                json.dump({"epoch": epoch, "val_mse": best_val_loss}, f)
            print(f"New best validation loss {best_val_loss:.6f} -> saved siren_best.pt\n")
        else:
            patience_counter += save_every
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch}: no improvement > {min_delta} "
                    f"in validation loss for {patience_counter} epochs."
                )
                torch.cuda.empty_cache()
                break

        torch.cuda.empty_cache()

torch.save(siren.state_dict(), base_dir / "siren_final.pt")
with open(base_dir / "siren_final_meta.json", "w") as f:
    json.dump({"epoch": epoch, "val_mse": best_val_loss}, f)
np.save(base_dir / "loss_history.npy", np.array(loss_history))
np.save(base_dir / "val_loss_history.npy", np.array(val_loss_history))

print(f"Training complete. Final model saved to {base_dir / 'siren_final.pt'}, "
      f"best model (val MSE={best_val_loss:.6f}) saved to {base_dir / 'siren_best.pt'}")
