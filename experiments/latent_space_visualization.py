"""
latent_space_visualization.py

Trains Breath MLP and Deep MLP on MNIST, then visualizes the latent
representations at each bottleneck/layer using t-SNE.

Goal: Show whether each Breath MLP bottleneck captures a different
level of abstraction (evidence for hierarchical multi-scale compression).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
TSNE_SAMPLES = 2000
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*60)
print(f"LATENT SPACE VISUALIZATION --- Breath vs Deep MLP (MNIST)")
print(f"Device: {device.type.upper()}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*60)

# ======================================================================
# DATA
# ======================================================================
print("\nCaricamento MNIST...")
(X_tr, y_tr), (X_te, y_te) = mnist.load_data()
X_tr_t = torch.tensor(X_tr.reshape(-1, 784).astype("float32") / 255.0)
y_tr_t = torch.tensor(y_tr.flatten().astype("int64"))
X_te_t = torch.tensor(X_te.reshape(-1, 784).astype("float32") / 255.0)
y_te_t = torch.tensor(y_te.flatten().astype("int64"))

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t),  batch_size=BATCH_SIZE, shuffle=False)

# ======================================================================
# ARCHITECTURES
# ======================================================================
breath_layers = generate_breath_architecture(
    START_WIDTH, compression_factor=0.25, expansion_factor=2.0,
    min_width=OUTPUT_DIM, output_dim=OUTPUT_DIM
)
deep_layers = generate_deep_architecture(START_WIDTH, decay_factor=0.5, min_width=OUTPUT_DIM)

print(f"\nBreath MLP layers: {breath_layers}")
print(f"Deep MLP layers:   {deep_layers}")

# ======================================================================
# VISUALIZATION SUBCLASS
# ======================================================================
class BreathMLPViz(BreathMLP):
    """Extends BreathMLP to cache post-activation bottleneck representations."""
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
            key = f"BN{bn_count+1}  (d={w})"
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
# TRAINING
# ======================================================================
def train_model(model, name):
    model = model.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Training {name}  ({n_params:,} params)...")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        if (ep + 1) % 4 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS}...")
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(1)
            correct += (preds == yb.to(device)).sum().item()
    acc = correct / len(y_te_t)
    elapsed = time.time() - t0
    print(f"  -> Acc={acc:.4f}  |  Time={elapsed:.1f}s  |  Params={n_params:,}")
    return model, acc, n_params

# ======================================================================
# EXTRACT LATENTS
# ======================================================================
def get_breath_bottlenecks(model, X, n=TSNE_SAMPLES):
    idx = torch.randperm(len(X))[:n]
    x_sub = X[idx].to(device)
    model.eval()
    with torch.no_grad():
        _ = model(x_sub)
    labels = y_te_t[idx].numpy()
    reps = {k: v.cpu().numpy() for k, v in model._bottlenecks.items()}
    return reps, labels, idx

def get_deep_layers(model, X, idx, n=TSNE_SAMPLES):
    x_sub = X[idx].to(device)
    labels = y_te_t[idx].numpy()
    n_lay = len(deep_layers)
    sel = sorted(set([0, n_lay // 2, n_lay - 1]))
    reps = {}
    hooks = []
    for si in sel:
        name = f"DL{si+1}  (d={deep_layers[si]})"
        reps[name] = []
        lin = model.linears[si]
        def make_hook(k):
            def hook(m, inp, out): reps[k].append(out.detach().cpu())
            return hook
        hooks.append(lin.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        _ = model(x_sub)
    for h in hooks:
        h.remove()
    reps = {k: torch.cat(v, 0).numpy() for k, v in reps.items()}
    return reps, labels

# ======================================================================
# t-SNE
# ======================================================================
def run_tsne_dict(reps_dict, tag=""):
    result = {}
    for name, rep in reps_dict.items():
        print(f"    t-SNE [{tag}] {name} shape={rep.shape}...", end=" ", flush=True)
        t = time.time()
        emb = TSNE(n_components=2, perplexity=40, random_state=SEED).fit_transform(rep)
        print(f"{time.time()-t:.1f}s")
        result[name] = emb
    return result

# ======================================================================
# PLOT HELPERS
# ======================================================================
PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#a9a9a9"
]
BG = "#0d0d1a"

def scatter(ax, emb, labels, title, note=""):
    for c in range(10):
        m = labels == c
        ax.scatter(emb[m,0], emb[m,1], c=PALETTE[c], s=5, alpha=0.7, linewidths=0, label=str(c))
    ax.set_title(title, fontsize=9.5, fontweight="bold", color="white", pad=5)
    if note:
        ax.set_xlabel(note, fontsize=7.5, color="#aaaaaa")
    ax.set_facecolor(BG)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

def make_figure(tsne_dict, labels, raw_emb, title, fname, acc, n_params):
    n = len(tsne_dict) + 1
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 5.2))
    fig.patch.set_facecolor(BG)
    scatter(axes[0], raw_emb, labels, "Raw Input\n(784d -> 2D t-SNE)", "pixel space")
    for ax, (name, emb) in zip(axes[1:], tsne_dict.items()):
        scatter(ax, emb, labels, name)
    handles = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=PALETTE[c],markersize=8,label=str(c)) for c in range(10)]
    axes[-1].legend(handles=handles, title="Digit", title_fontsize=9, fontsize=8.5,
                    loc="lower right", labelcolor="white", facecolor="#1a1a2e", edgecolor="#444466")
    fig.suptitle(f"{title}\nAcc={acc:.4f}  |  Params={n_params:,}",
                 fontsize=12, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"  -> Salvato: {fname}")
    plt.close()

def make_comparison(b_tsne, d_tsne, b_labels, d_labels, raw_emb, b_acc, b_n, d_acc, d_n, fname):
    n_cols = max(len(b_tsne), len(d_tsne)) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(4.8*n_cols, 10.5))
    fig.patch.set_facecolor(BG)
    rows = [
        ("Breath MLP -- Bottlenecks", b_tsne, b_labels, b_acc, b_n),
        ("Deep MLP -- Hidden Layers",  d_tsne, d_labels, d_acc, d_n),
    ]
    for row, (row_title, tsne_dict, labels, acc, n_p) in enumerate(rows):
        scatter(axes[row,0], raw_emb, labels, "Raw Input\n(784d -> 2D)", "pixel space")
        for col, (name, emb) in enumerate(tsne_dict.items(), start=1):
            scatter(axes[row,col], emb, labels, name)
        for col in range(len(tsne_dict)+1, n_cols):
            axes[row,col].set_visible(False)
        axes[row,0].set_ylabel(f"{row_title}\nAcc={acc:.4f} | Params={n_p:,}",
                               fontsize=10, fontweight="bold", color="white", labelpad=8)
    handles = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=PALETTE[c],markersize=9,label=str(c)) for c in range(10)]
    fig.legend(handles=handles, title="Digit", title_fontsize=10, fontsize=9,
               loc="center right", bbox_to_anchor=(1.0,0.5),
               labelcolor="white", facecolor="#1a1a2e", edgecolor="#444466")
    fig.suptitle("Latent Space -- Breath MLP Bottlenecks vs Deep MLP Layers (MNIST)",
                 fontsize=13, fontweight="bold", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"  -> Salvato: {fname}")
    plt.close()

# ======================================================================
# MAIN
# ======================================================================
breath_model = BreathMLPViz(784, breath_layers, OUTPUT_DIM, use_skips=True,
                            activation="relu", use_norm=False)
deep_model   = DeepMLP(784, deep_layers, OUTPUT_DIM, use_skips=True)

breath_model, breath_acc, breath_n = train_model(breath_model, "Breath MLP")
deep_model,   deep_acc,   deep_n   = train_model(deep_model,   "Deep MLP")

print(f"\n  Estraendo rappresentazioni (n={TSNE_SAMPLES})...")
breath_reps, b_labels, idx_sub = get_breath_bottlenecks(breath_model, X_te_t)
deep_reps,   d_labels          = get_deep_layers(deep_model, X_te_t, idx_sub)

print("  Running t-SNE sul pixel space raw...")
t0 = time.time()
raw_emb = TSNE(n_components=2, perplexity=40, random_state=SEED).fit_transform(X_te_t[idx_sub].numpy())
print(f"    {time.time()-t0:.1f}s")

print("\n  Running t-SNE sui bottleneck di Breath MLP:")
b_tsne = run_tsne_dict(breath_reps, tag="Breath")
print("  Running t-SNE sui layer di Deep MLP:")
d_tsne = run_tsne_dict(deep_reps, tag="Deep")

print("\n  Generazione grafici...")
make_figure(b_tsne, b_labels, raw_emb,
            "Breath MLP -- Latent Space at Each Bottleneck (MNIST)",
            "latent_breath_bottlenecks.png", breath_acc, breath_n)
make_figure(d_tsne, d_labels, raw_emb,
            "Deep MLP -- Latent Space at Selected Layers (MNIST)",
            "latent_deep_layers.png", deep_acc, deep_n)
make_comparison(b_tsne, d_tsne, b_labels, d_labels, raw_emb,
                breath_acc, breath_n, deep_acc, deep_n,
                "latent_space_comparison.png")

print("\n" + "="*60)
print("COMPLETATO!")
print(f"  Breath MLP -- Acc={breath_acc:.4f}  |  Params={breath_n:,}")
print(f"  Deep MLP   -- Acc={deep_acc:.4f}  |  Params={deep_n:,}")
print("Output: latent_breath_bottlenecks.png, latent_deep_layers.png, latent_space_comparison.png")
print("="*60)
