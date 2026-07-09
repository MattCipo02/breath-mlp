import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np
import sys
import pandas as pd
import argparse

# Import Keras to load CIFAR-10 from local cache
from tensorflow.keras.datasets import cifar10
from torch.utils.data import TensorDataset, DataLoader

# Import custom Breath MLP
from breath_mlp import generate_breath_architecture, BreathMLP, BreathMLPPool

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
args_cli = parser.parse_args()

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(f"DISPOSITIVO ATTIVO PER IL CANONICAL 4X Vision Transformer (ViT): {device.type.upper()}")
if device.type == "cuda":
    print(f" -> GPU: {torch.cuda.get_device_name(0)}")
print("="*60)

# Hyperparameters
batch_size = 128
epochs = args_cli.epochs
learning_rate = 1e-3
d_model = 192
n_head = 4
n_layer = 4
patch_size = 4
num_patches = (32 // patch_size) ** 2
patch_dim = 3 * patch_size * patch_size

# Load CIFAR-10
print("\nCaricamento dati CIFAR-10 dalla cache...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()

# Reshape images to (B, C, H, W) for patch projection
X_train_t = torch.tensor(X_train_raw.transpose(0, 3, 1, 2).astype("float32") / 255.0)
y_train_t = torch.tensor(y_train_raw.flatten().astype("int64"))
X_test_t = torch.tensor(X_test_raw.transpose(0, 3, 1, 2).astype("float32") / 255.0)
y_test_t = torch.tensor(y_test_raw.flatten().astype("int64"))

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =====================================================================
# --- ViT MODULES ---
# =====================================================================
class BidirectionalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_model = d_model
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head) ** 0.5)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# Canonical 4x FFN
class StandardFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    def forward(self, x):
        return self.net(x)

# Canonical 4x Breath FFN (Linear)
class BreathFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = [4 * d_model] # [768]
        self.net = BreathMLP(
            input_dim=d_model,
            hidden_layers=hidden_layers,
            output_dim=d_model,
            use_skips=True,
            activation="gelu",
            use_norm=True
        )
    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(B * T, C)
        out_flat = self.net(x_flat)
        return out_flat.view(B, T, C)

# Canonical 4x BreathPool Max
class BreathPoolMaxFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = [4 * d_model] # [768]
        self.net = BreathMLPPool(
            input_dim=d_model,
            hidden_layers=hidden_layers,
            output_dim=d_model,
            use_skips=True,
            pool_type="max",
            activation="gelu",
            use_norm=True
        )
    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(B * T, C)
        out_flat = self.net(x_flat)
        return out_flat.view(B, T, C)

# Canonical 4x BreathPool Avg
class BreathPoolAvgFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = [4 * d_model] # [768]
        self.net = BreathMLPPool(
            input_dim=d_model,
            hidden_layers=hidden_layers,
            output_dim=d_model,
            use_skips=True,
            pool_type="avg",
            activation="gelu",
            use_norm=True
        )
    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(B * T, C)
        out_flat = self.net(x_flat)
        return out_flat.view(B, T, C)

# Canonical 4x BreathPool Hybrid
class BreathPoolHybridFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = [4 * d_model] # [768]
        self.net = BreathMLPPool(
            input_dim=d_model,
            hidden_layers=hidden_layers,
            output_dim=d_model,
            use_skips=True,
            pool_type="hybrid",
            activation="gelu",
            use_norm=True
        )
    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(B * T, C)
        out_flat = self.net(x_flat)
        return out_flat.view(B, T, C)

class ViTBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_type):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        if ffn_type == "standard":
            self.ffn = StandardFFN(d_model)
        elif ffn_type == "breath":
            self.ffn = BreathFFN(d_model)
        elif ffn_type == "breath_pool_max":
            self.ffn = BreathPoolMaxFFN(d_model)
        elif ffn_type == "breath_pool_avg":
            self.ffn = BreathPoolAvgFFN(d_model)
        elif ffn_type == "breath_pool_hybrid":
            self.ffn = BreathPoolHybridFFN(d_model)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, num_classes=10, ffn_type="standard"):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch Projection
        self.patch_to_embedding = nn.Linear(patch_dim, d_model)
        
        # Positional Embedding & CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            ViTBlock(d_model, n_head, ffn_type) for _ in range(n_layer)
        ])
        
        # Classification Head
        self.ln_f = nn.LayerNorm(d_model)
        self.mlp_head = nn.Linear(d_model, num_classes)
        
    def forward(self, img):
        B, C, H, W = img.size()
        
        # Create patches
        # img shape: [B, C, H, W] -> patches: [B, num_patches, patch_dim]
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, num_patches, -1)
        
        # Project patches
        x = self.patch_to_embedding(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Transformer layers
        for block in self.blocks:
            x = block(x)
            
        # CLS token output representation
        x = self.ln_f(x[:, 0])
        return self.mlp_head(x)

# =====================================================================
# --- TRAINING FUNCTION ---
# =====================================================================
def train_vit(ffn_type, seed):
    print("\n" + "-"*50)
    print(f" TRAINING ViT (CIFAR-10) | FFN: {ffn_type.upper()} | SEED: {seed}")
    print("-"*50)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    model = SimpleViT(num_classes=10, ffn_type=ffn_type).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Parametri Addestrabili Totali: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs_batch, targets_batch in train_loader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Eval Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs_batch, targets_batch in test_loader:
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
                outputs = model(inputs_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += targets_batch.size(0)
                correct += (predicted == targets_batch).sum().item()
        
        accuracy = correct / total
        print(f"    * Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Test Acc: {accuracy:.4f}")
        
    elapsed = time.time() - start_time
    print(f" -> Addestramento ViT completato in {elapsed:.1f} secondi.")
    
    return trainable_params, elapsed, accuracy

if __name__ == '__main__':
    SEEDS = [42, 137, 2026]
    results_raw = []
    
    # List of models to run
    configs = ["standard", "breath", "breath_pool_max", "breath_pool_avg", "breath_pool_hybrid"]
    
    for seed in SEEDS:
        for config in configs:
            p, t, acc = train_vit(config, seed)
            results_raw.append({"Config": config, "Seed": seed, "Params": p, "Time": t, "Accuracy": acc})
        
    df_raw = pd.DataFrame(results_raw)
    df_raw.to_csv("vit_canonical_4x_raw_multi_seed_results.csv", index=False)
    
    # Compute aggregates
    summary = []
    for config in ["standard", "breath", "breath_pool_max", "breath_pool_avg", "breath_pool_hybrid"]:
        sub = df_raw[df_raw["Config"] == config]
        if len(sub) == 0:
            continue
        summary.append({
            "Config": config,
            "Params": sub["Params"].iloc[0],
            "Time_mean": sub["Time"].mean(),
            "Time_std": sub["Time"].std(),
            "Accuracy_mean": sub["Accuracy"].mean(),
            "Accuracy_std": sub["Accuracy"].std()
        })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv("vit_canonical_4x_results.csv", index=False)
    
    # Print Report
    print("\n" + "="*90)
    print("             REPORT COMPARATIVO MULTI-SEED CANONICAL 4X VISION TRANSFORMER (CIFAR-10)")
    print("="*90)
    print(f"Configurazione FFN        | Parametri | Tempo (medio) | Accuracy Media (5 epoche)")
    print(f"--------------------------|-----------|---------------|--------------------------")
    for row in summary:
        print(f"{row['Config']:26s} | {row['Params']:9,} | {row['Time_mean']:11.1f}s | {row['Accuracy_mean']:.4f} +/- {row['Accuracy_std']:.4f}")
    print("="*90)
