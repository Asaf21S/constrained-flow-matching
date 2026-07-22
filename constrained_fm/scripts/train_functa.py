import torch
from torch.func import vmap
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np

from constrained_fm.functa_dataset.generate_dataset import generate_dataset
from constrained_fm.src.models.functa_siren import build_modulated_siren

# --- 1. Headless Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")

# --- 2. Project Paths ---
# Pointing to your cluster home directory
base_dir = Path("/workspace/constrained_fm/functa_dataset/")
checkpoint_dir = base_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# --- 3. Hyperparameters ---
epochs = 3000
batch_size = 256
lr = 1e-4
lambda_z = 1e-4
latent_dim = 256
hidden_dim = 256
n_layers = 4
w0 = 10.0
num_samples = 1500

save_every = 50          
patience = 50            
min_delta = 1e-4         

DOMAIN_MIN = -4.5
DOMAIN_MAX = 4.5
DOMAIN_SIZE = DOMAIN_MAX - DOMAIN_MIN
EPSILON = 0.05

data_path = base_dir / "test_spatial_dataset.pt"

if not data_path.exists():
    print("Dataset not found. Generating...")
    generate_dataset(
        num_shapes=10000,
        points_per_shape=5000,
        output_path=data_path
    )
else:
    print(f"Loading existing dataset from {data_path}")

data = torch.load(data_path, weights_only=True)
X = data["X"]
Y = data["Y"]
meta = data["meta"]

N, M, _ = X.shape
indices_tensor = torch.arange(N, dtype=torch.long)
dataset = TensorDataset(indices_tensor, X, Y)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Keep at 0 to avoid shared memory crashes on the cluster
    pin_memory=(device.type == "cuda"),
)

siren = build_modulated_siren(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, w0=w0).to(device)
embed = nn.Embedding(N, latent_dim).to(device)
nn.init.normal_(embed.weight, mean=0.0, std=0.01)

optimizer = torch.optim.Adam(list(siren.parameters()) + list(embed.parameters()), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
bce_loss = nn.BCELoss()

siren.train()
embed.train()

# Optional: Comment this out if it throws compilation errors on the cluster's CUDA drivers
# siren = torch.compile(siren)

loss_history = []
best_loss = float('inf')
patience_counter = 0

for epoch in tqdm(range(1, epochs + 1), desc="Training Auto-Decoder"):
    epoch_loss = 0.0
    # Unpack idx, X, Y from the DataLoader (indices are already shuffled correctly)
    for idx_batch, X_batch, Y_batch in loader:
        B = X_batch.shape[0]
        perm = torch.randperm(X_batch.shape[1])[:num_samples]
        
        idx_batch = idx_batch.to(device)
        X_batch = X_batch[:, perm, :].to(device)
        Y_batch = Y_batch[:, perm].to(device, dtype=torch.float32)

        # Retrieve the corresponding latent vectors using the true shuffled indices
        z_batch = embed(idx_batch)

        # Explicit in_dims to avoid silent broadcasting issues
        preds = vmap(siren, in_dims=(0, 0))(X_batch, z_batch)  # (B, M, 1)
        preds = preds.squeeze(-1)  # (B, M)

        loss = bce_loss(preds, Y_batch) + lambda_z * (z_batch ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(siren.parameters()) + list(embed.parameters()), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * B

    # Average loss per shape (BCELoss already averages over points)
    avg_loss = epoch_loss / N
    loss_history.append(avg_loss)

    scheduler.step(avg_loss)

    if epoch % 10 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d} – avg BCE+L2 loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

    if epoch % save_every == 0:
        torch.save(siren.state_dict(), checkpoint_dir / f"siren_ep{epoch:04d}.pt")
        torch.save(embed.state_dict(), checkpoint_dir / f"embed_ep{epoch:04d}.pt")
        
    if best_loss - avg_loss > min_delta:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(siren.state_dict(), base_dir / "siren_best.pt")
        torch.save(embed.state_dict(), base_dir / "embed_best.pt")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"\n[Early Stopping] Triggered at Epoch {epoch}! Loss hasn't improved by {min_delta} for {patience} epochs.")
        break

torch.save(siren.state_dict(), base_dir / "siren_final.pt")
torch.save(embed.state_dict(), base_dir / "embed_final.pt")
np.save(base_dir / "loss_history.npy", np.array(loss_history))
print(f"Training complete. Best models saved to {base_dir}")
