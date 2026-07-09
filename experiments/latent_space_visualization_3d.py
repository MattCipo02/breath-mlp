"""
latent_space_visualization_3d.py

Fits a 3D t-SNE projection of the latent space representations
comparing Breath MLP bottlenecks vs Deep MLP layers on MNIST, and plots them in 3D.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
from torch.utils.data import TensorDataset, DataLoader

from breath_mlp import generate_breath_architecture, generate_deep_architecture, BreathMLP, DeepMLP

# ======================================================================
# CONFIG
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 12
BATCH_SIZE = 128
LR = 0.001
START_WIDTH = 512
OUTPUT_DIM = 10
TSNE_SAMPLES = 1500
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*60)
print(f"3D COMPARATIVE LATENT SPACE VISUALIZATION")
print(f"Device: {device.type.upper()}")
print("="*60)

# ======================================================================
# DATA
# ======================================================================
(X_tr, y_tr), (X_te, y_te) = mnist.load_data()
X_tr_t = torch.tensor(X_tr.reshape(-1, 784).astype("float32") / 255.0)
y_tr_t = torch.tensor(y_tr.flatten().astype("int64"))
X_te_t = torch.tensor(X_te.reshape(-1, 784).astype("float32") / 255.0)
y_te_t = torch.tensor(y_te.flatten().astype("int64"))

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

# ======================================================================
# ARCHITECTURES
# ======================================================================
breath_layers = generate_breath_architecture(
    START_WIDTH, compression_factor=0.25, expansion_factor=2.0,
    min_width=OUTPUT_DIM, output_dim=OUTPUT_DIM
)
deep_layers = generate_deep_architecture(START_WIDTH, decay_factor=0.5, min_width=OUTPUT_DIM)

class BreathMLPViz3D(BreathMLP):
    def forward(self, x):
        self._bottlenecks = {}
        bn_count = 0
        if self.use_norm:
            x = self.input_norm(x)
        x = self.act(self.linears[0](x))
        compression_tensors = {}
        linear_idx = 1
        i = 1
        last_comp_idx = None
        while i < len(self.hidden_layers):
            comp_tensor = self.linears[linear_idx](x)
            linear_idx += 1
            if self.use_skips and last_comp_idx is not None:
                proj_key = f"proj_{last_comp_idx}_to_{i}"
                prev_comp_tensor = compression_tensors[last_comp_idx]
                proj = self.projections[proj_key](prev_comp_tensor)
                comp_tensor = comp_tensor + proj
            comp_tensor = self.act(comp_tensor)
            if self.use_norm:
                comp_tensor = self.norms[f"norm_{i}"](comp_tensor)
            w = self.hidden_layers[i]
            key = f"BN{bn_count+1} (d={w})"
            self._bottlenecks[key] = comp_tensor.detach()
            bn_count += 1
            compression_tensors[i] = comp_tensor
            x = comp_tensor
            last_comp_idx = i
            if i + 1 < len(self.hidden_layers):
                x = self.act(self.linears[linear_idx](x))
                linear_idx += 1
                i += 2
            else:
                i += 1
        return self.output_layer(x)

# ======================================================================
# TRAINING & EXTRACTION
# ======================================================================
breath_model = BreathMLPViz3D(784, breath_layers, OUTPUT_DIM, use_skips=True,
                              activation="relu", use_norm=False).to(device)
deep_model = DeepMLP(784, deep_layers, OUTPUT_DIM, use_skips=True).to(device)

opt_b = torch.optim.Adam(breath_model.parameters(), lr=LR)
opt_d = torch.optim.Adam(deep_model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()

print(f"Training Breath MLP...")
breath_model.train()
for ep in range(EPOCHS):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_b.zero_grad()
        crit(breath_model(xb), yb).backward()
        opt_b.step()

print(f"Training Deep MLP...")
deep_model.train()
for ep in range(EPOCHS):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_d.zero_grad()
        crit(deep_model(xb), yb).backward()
        opt_d.step()

# Extract samples
idx = torch.randperm(len(X_te_t))[:TSNE_SAMPLES]
x_sub = X_te_t[idx].to(device)
labels = y_te_t[idx].numpy()

# Extract Breath BN3
breath_model.eval()
with torch.no_grad():
    _ = breath_model(x_sub)
breath_rep = breath_model._bottlenecks[list(breath_model._bottlenecks.keys())[-1]].cpu().numpy()

# Extract Deep DL6 (last layer before output)
deep_model.eval()
deep_reps = []
def hook(m, inp, out):
    deep_reps.append(out.detach().cpu())
h = deep_model.linears[-1].register_forward_hook(hook)
with torch.no_grad():
    _ = deep_model(x_sub)
h.remove()
deep_rep = deep_reps[0].numpy()

# ======================================================================
# 3D t-SNE FITS
# ======================================================================
print(f"Fitting 3D t-SNE for Breath MLP BN3 (d=16)...")
t0 = time.time()
b_emb_3d = TSNE(n_components=3, perplexity=30, random_state=SEED).fit_transform(breath_rep)
print(f"  Completed in {time.time()-t0:.1f}s")

print(f"Fitting 3D t-SNE for Deep MLP DL6 (d=16)...")
t0 = time.time()
d_emb_3d = TSNE(n_components=3, perplexity=30, random_state=SEED).fit_transform(deep_rep)
print(f"  Completed in {time.time()-t0:.1f}s")

# ======================================================================
# 3D COMPARATIVE PLOT
# ======================================================================
PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#a9a9a9"
]
BG = "#0d0d1a"

fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor(BG)

# We will plot 2 rows (Row 1: Breath, Row 2: Deep) and 2 angles per row
angles = [(20, 45), (45, 120)]

# Row 1: Breath MLP
for idx_angle, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(2, 2, idx_angle + 1, projection="3d")
    ax.set_facecolor(BG)
    for c in range(10):
        mask = labels == c
        ax.scatter(b_emb_3d[mask, 0], b_emb_3d[mask, 1], b_emb_3d[mask, 2],
                   c=PALETTE[c], s=10, alpha=0.8, edgecolors="none")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")
    ax.grid(color="#333355", linestyle="--", linewidth=0.5)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.set_title(f"Breath MLP BN3 (d=16) | Elev={elev}°, Azim={azim}°", color="white", fontsize=10, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)

# Row 2: Deep MLP
for idx_angle, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(2, 2, idx_angle + 3, projection="3d")
    ax.set_facecolor(BG)
    for c in range(10):
        mask = labels == c
        ax.scatter(d_emb_3d[mask, 0], d_emb_3d[mask, 1], d_emb_3d[mask, 2],
                   c=PALETTE[c], s=10, alpha=0.8, edgecolors="none")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")
    ax.grid(color="#333355", linestyle="--", linewidth=0.5)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.set_title(f"Deep MLP DL6 (d=16) | Elev={elev}°, Azim={azim}°", color="white", fontsize=10, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)

# Legend
handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=PALETTE[c], markersize=8, label=str(c)) for c in range(10)]
fig.legend(handles=handles, title="Digit", title_fontsize=10, fontsize=9,
           loc="center right", bbox_to_anchor=(0.98, 0.5),
           labelcolor="white", facecolor="#1a1a2e", edgecolor="#444466")

fig.suptitle("Latent Space 3D Comparison --- Breath MLP (BN3) vs Deep MLP (DL6) (MNIST)",
             fontsize=14, fontweight="bold", color="white", y=0.97)

plt.tight_layout()
out_name = "latent_comparison_3d.png"
plt.savefig(out_name, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
