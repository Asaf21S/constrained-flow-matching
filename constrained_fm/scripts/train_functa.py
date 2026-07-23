import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np

from constrained_fm.src.datasets.constraints import sample_valid_polynomials
from constrained_fm.src.geometry.polynomials import compute_poly_features_batched, evaluate_poly_batched
from constrained_fm.src.models.functa_siren import build_modulated_siren

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")

base_dir = Path("/workspace/constrained_fm/functa_dataset/")
checkpoint_dir = base_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

epochs = 3000
steps_per_epoch = 400  # 400 steps * 256 batch size = ~102,400 shapes per epoch
batch_size = 256
points_per_shape = 1500

# Meta-Learning (CAVIA) Hyperparameters
outer_lr = 1e-4
inner_lr = 1e-2
inner_steps = 3  # Fast adaptation steps
lambda_z = 1e-4  # L2 penalty on the context vector

latent_dim = 256
hidden_dim = 256
n_layers = 4
w0 = 10.0
poly_degree = 3
plane_scale = 4.5

save_every = 50
patience = 250
min_delta = 1e-4


def generate_batch(batch_size, num_points):
    """Generates an on-the-fly batch of polynomial constraints entirely in VRAM."""
    C = sample_valid_polynomials(batch_size=batch_size, degree=poly_degree, scale=plane_scale, device=device)

    # Sample points in [-4.5, 4.5]
    X = (torch.rand(batch_size, num_points, 2, device=device) * 9.0) - 4.5
    X_scaled = X / plane_scale

    x_pow, y_pow = compute_poly_features_batched(X_scaled, degree=poly_degree)
    P_vals = evaluate_poly_batched(x_pow, y_pow, C)

    # Binary mask: 1 if P(x, y) <= 0, else 0
    Y = (P_vals <= 0).to(torch.float32)
    return X, Y


print("Generating fixed holdout set of 100 polynomials for validation...")
val_X, val_Y = generate_batch(batch_size=100, num_points=points_per_shape)

siren = build_modulated_siren(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, w0=w0).to(device)
optimizer = torch.optim.Adam(siren.parameters(), lr=outer_lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
bce_loss = nn.BCELoss()

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
            loss_inner = bce_loss(preds, Y_batch)

            # Compute gradients of z with create_graph=True to allow outer loop backprop
            grad_z = torch.autograd.grad(loss_inner, z, create_graph=True)[0]

            # Manual SGD step for z
            z = z - inner_lr * grad_z

        optimizer.zero_grad(set_to_none=True)

        # Evaluate the adapted z on the task
        preds_adapted = siren(X_batch, z).squeeze(-1)

        # Compute final loss (BCE + L2 Regularization on z to prevent extreme modulation)
        loss_outer = bce_loss(preds_adapted, Y_batch) + lambda_z * (z ** 2).mean()

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
        print(f"Epoch {epoch:04d} - Outer BCE+L2 Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

    if epoch % save_every == 0:
        print(f"\n--- Running Validation Inference (Epoch {epoch}) ---")
        siren.eval()

        # Initialize validation z vectors
        z_val = torch.zeros(val_X.shape[0], latent_dim, device=device, requires_grad=True)

        # Rigorous inference optimization identical to test-time
        val_opt = torch.optim.Adam([z_val], lr=0.01)
        val_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(val_opt, T_max=300, eta_min=1e-5)

        for _ in range(300):
            val_opt.zero_grad(set_to_none=True)
            preds_val = siren(val_X, z_val).squeeze(-1)

            # Per-shape BCE loss + penalty
            bce = nn.BCELoss(reduction='none')(preds_val, val_Y).mean(dim=1)
            penalty = lambda_z * (z_val ** 2).mean(dim=1)
            loss_val = (bce + penalty).mean()

            loss_val.backward()
            val_opt.step()
            val_scheduler.step()

        with torch.no_grad():
            preds_final = siren(val_X, z_val).squeeze(-1)
            avg_val_loss = bce_loss(preds_final, val_Y).item()

        val_loss_history.append((epoch, avg_val_loss))
        print(f"Validation Extraction Loss: {avg_val_loss:.6f}\n")

        torch.cuda.empty_cache()

        # Save checkpoints
        torch.save(siren.state_dict(), checkpoint_dir / f"siren_ep{epoch:04d}.pt")

        # Early Stopping
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(siren.state_dict(), base_dir / "siren_best.pt")
        else:
            patience_counter += save_every

        if patience_counter >= patience:
            print(f"\n[Early Stopping] Triggered at Epoch {epoch}!")
            break

torch.save(siren.state_dict(), base_dir / "siren_final.pt")
np.save(base_dir / "loss_history.npy", np.array(loss_history))
np.save(base_dir / "val_loss_history.npy", np.array(val_loss_history))

print(f"Training complete. Best model saved to {base_dir}")
