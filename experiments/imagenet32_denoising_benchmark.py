import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import time
import os
import argparse
import pickle
import math
import sys
import urllib.request

# Local import of breath_mlp classes
from breath_mlp import (
    generate_breath_architecture,
    generate_deep_architecture,
    BreathMLP,
    DeepMLP
)

# =====================================================================
# --- ARCHITECTURE DEFINITIONS ---
# =====================================================================
class HourglassBlock(nn.Module):
    def __init__(self, dz, dh):
        super().__init__()
        self.norm = nn.LayerNorm(dz)
        self.fc1 = nn.Linear(dz, dh)
        self.fc2 = nn.Linear(dh, dz)
        self.relu = nn.ReLU()
        
    def forward(self, z):
        return z + self.fc2(self.relu(self.fc1(self.norm(z))))

class HourglassMLP(nn.Module):
    def __init__(self, input_dim=3072, dz=3546, dh=270, L=5, fixed_projection=True):
        super().__init__()
        self.fixed_projection = fixed_projection
        self.input_dim = input_dim
        self.dz = dz
        
        if fixed_projection:
            Win_tensor = torch.empty(dz, input_dim)
            nn.init.normal_(Win_tensor, std=1.0 / math.sqrt(input_dim))
            self.register_buffer('Win', Win_tensor)
        else:
            self.Win = nn.Parameter(torch.empty(dz, input_dim))
            nn.init.normal_(self.Win, std=1.0 / math.sqrt(input_dim))
            
        self.blocks = nn.ModuleList([HourglassBlock(dz, dh) for _ in range(L)])
        self.Wout = nn.Linear(dz, input_dim)
        
    def forward(self, x):
        z = torch.matmul(x, self.Win.t())
        for block in self.blocks:
            z = block(z)
        out = self.Wout(z)
        return out

class ConventionalBlock(nn.Module):
    def __init__(self, dx, dh):
        super().__init__()
        self.norm = nn.LayerNorm(dx)
        self.fc1 = nn.Linear(dx, dh)
        self.fc2 = nn.Linear(dh, dx)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return x + self.fc2(self.relu(self.fc1(self.norm(x))))

class ConventionalMLP(nn.Module):
    def __init__(self, input_dim=3072, dh=3075, L=2):
        super().__init__()
        self.blocks = nn.ModuleList([ConventionalBlock(input_dim, dh) for _ in range(L)])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# =====================================================================
# --- PUSH NOTIFICATION UTILITY (ntfy.sh) ---
# =====================================================================
def send_push_notification(topic, title, message):
    if not topic:
        return
    url = f"https://ntfy.sh/{topic}"
    try:
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "default"}
        )
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Errore durante l'invio della notifica: {e}")

# =====================================================================
# --- DATASET UTILITIES (Optimized to uint8 to save RAM) ---
# =====================================================================
def load_cifar10():
    print("Loading cached CIFAR-10 dataset...")
    try:
        from tensorflow.keras.datasets import cifar10
        (X_train_raw, _), (X_test_raw, _) = cifar10.load_data()
        X_train = X_train_raw.reshape(-1, 3072)
        X_test = X_test_raw.reshape(-1, 3072)
    except ImportError:
        train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        X_train = train_ds.data.reshape(-1, 3072)
        X_test = test_ds.data.reshape(-1, 3072)
        
    return torch.tensor(X_train, dtype=torch.uint8), torch.tensor(X_test, dtype=torch.uint8)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_imagenet32(data_dir):
    print(f"Loading ImageNet-32 dataset from: {data_dir}")
    train_x = []
    
    for i in range(1, 11):
        batch_file = os.path.join(data_dir, f'train_data_batch_{i}')
        if not os.path.exists(batch_file):
            print(f"Warning: file {batch_file} not found. Cannot load full ImageNet-32.")
            return None, None
        d = unpickle(batch_file)
        x = d['data']
        train_x.append(x)
        
    train_x = np.concatenate(train_x, axis=0)
    
    val_file = os.path.join(data_dir, 'val_data')
    if not os.path.exists(val_file):
        print(f"Warning: validation file {val_file} not found.")
        return None, None
    d_val = unpickle(val_file)
    test_x = d_val['data']
    
    return torch.tensor(train_x, dtype=torch.uint8), torch.tensor(test_x, dtype=torch.uint8)

# =====================================================================
# --- DATA AUGMENTATION (4x) ---
# =====================================================================
def augment_batch(batch_flat):
    B = batch_flat.size(0)
    batch_img = batch_flat.view(B, 3, 32, 32)
    
    orig = batch_img
    h_flip = torch.flip(batch_img, dims=[3])
    v_flip = torch.flip(batch_img, dims=[2])
    hv_flip = torch.flip(batch_img, dims=[2, 3])
    
    augmented = torch.cat([orig, h_flip, v_flip, hv_flip], dim=0)
    return augmented.view(-1, 3072)

# =====================================================================
# --- NOISE GENERATION ---
# =====================================================================
def add_noise(images, noise_std=0.25):
    noise = torch.randn_like(images) * noise_std
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)

# =====================================================================
# --- PSNR CALCULATION ---
# =====================================================================
def compute_psnr(clean, denoised):
    mse = torch.mean((clean - denoised) ** 2)
    if mse == 0:
        return float('inf')
    psnr = -10.0 * torch.log10(mse)
    return psnr.item()

# =====================================================================
# --- MAIN PIPELINE ---
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Image Denoising Benchmark (Breath MLP vs Hourglass vs Conventional)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet32"],
                        help="Dataset to benchmark on")
    parser.add_argument("--data_dir", type=str, default="./imagenet32",
                        help="Directory containing ImageNet-32 batch files")
    parser.add_argument("--model", type=str, default="breath", choices=["breath", "hourglass", "conventional"],
                        help="Model architecture to run")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--dz", type=int, default=4096,
                        help="Width parameter dz (starting width for Breath, latent dim for Hourglass)")
    parser.add_argument("--dh", type=int, default=270,
                        help="Bottleneck width dh (Hourglass / Conventional only)")
    parser.add_argument("--L", type=int, default=5,
                        help="Number of residual blocks L (Hourglass / Conventional only)")
    parser.add_argument("--fixed_proj", type=str, default="true", choices=["true", "false"],
                        help="Freeze the input projection layer at random init (Hourglass only)")
    parser.add_argument("--min_width", type=int, default=16,
                        help="Minimum width for Breath MLP layers")
    parser.add_argument("--activation", type=str, default="relu", choices=["gelu", "silu", "relu"],
                        help="Activation function for BreathMLP (gelu, silu, relu)")
    parser.add_argument("--use_norm", type=str, default="false", choices=["true", "false"],
                        help="Use LayerNorm inside BreathMLP (true, false)")
    parser.add_argument("--ntfy_topic", type=str, default="",
                        help="ntfy topic to send push notifications to your phone")
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "imagenet32":
        X_train, X_test = load_imagenet32(args.data_dir)
        if X_train is None:
            print("Failed to load ImageNet-32. Falling back to CIFAR-10.")
            X_train, X_test = load_cifar10()
            args.dataset = "cifar10"
    else:
        X_train, X_test = load_cifar10()
        
    print(f"Loaded {X_train.size(0):,} training images and {X_test.size(0):,} test images.")
    
    # Define DataLoader
    train_loader = DataLoader(TensorDataset(X_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=256, shuffle=False)
    
    # Build Model
    fixed_proj_bool = args.fixed_proj == "true"
    use_norm_bool = args.use_norm == "true"
    if args.model == "hourglass":
        model = HourglassMLP(input_dim=3072, dz=args.dz, dh=args.dh, L=args.L, fixed_projection=fixed_proj_bool)
    elif args.model == "breath":
        breath_layers = generate_breath_architecture(args.dz, min_width=args.min_width)
        print(f"Generating Breath MLP layers: {breath_layers}")
        model = BreathMLP(input_dim=3072, hidden_layers=breath_layers, output_dim=3072, use_skips=True, activation=args.activation, use_norm=use_norm_bool)
    else:
        model = ConventionalMLP(input_dim=3072, dh=args.dh, L=args.L)
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("\n" + "="*60)
    print(f"Model: {args.model.upper()} | Trainable Params: {trainable_params:,} | Total Params: {total_params:,}")
    print(f"Device: {device.type.upper()}")
    print("="*60 + "\n")
    
    # Send start notification
    if args.ntfy_topic:
        send_push_notification(
            args.ntfy_topic, 
            "Training Started", 
            f"Model: {args.model.upper()} | Params: {trainable_params:,}\nDataset: {args.dataset.upper()} | Epochs: {args.epochs}"
        )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (clean_batch,) in enumerate(train_loader):
            # Apply 4x Data Augmentation
            clean_batch_aug = augment_batch(clean_batch)
            
            # Move to device and convert to float32 [0, 1] on the fly
            clean_batch_aug = clean_batch_aug.to(device).float() / 255.0
            noisy_batch_aug = add_noise(clean_batch_aug, noise_std=0.25)
            
            optimizer.zero_grad()
            denoised_batch = model(noisy_batch_aug)
            loss = criterion(denoised_batch, clean_batch_aug)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                step_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{args.epochs} | Step {batch_idx+1}/{len(train_loader)} | Loss: {step_loss:.5f}")
                
        elapsed = time.time() - start_time
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed in {elapsed:.1f}s | Avg Loss: {epoch_loss:.5f}")
        
        # Intermediate validation
        model.eval()
        test_psnr = 0.0
        with torch.no_grad():
            for test_batch, in test_loader:
                # Convert to float32 [0, 1] on the fly
                test_batch = test_batch.to(device).float() / 255.0
                noisy_test_batch = add_noise(test_batch, noise_std=0.25)
                denoised_test = model(noisy_test_batch)
                test_psnr += compute_psnr(test_batch, denoised_test)
                
        avg_test_psnr = test_psnr / len(test_loader)
        print(f"Validation PSNR: {avg_test_psnr:.4f} dB\n")
        
        # Send epoch completed notification
        if args.ntfy_topic:
            send_push_notification(
                args.ntfy_topic,
                f"Epoch {epoch+1}/{args.epochs} Completed ({args.model.upper()})",
                f"Loss: {epoch_loss:.5f} | Val PSNR: {avg_test_psnr:.4f} dB\nTempo: {elapsed:.1f}s"
            )
        
    # Final evaluation
    model.eval()
    final_psnr = 0.0
    final_mse = 0.0
    with torch.no_grad():
        for test_batch, in test_loader:
            test_batch = test_batch.to(device).float() / 255.0
            noisy_test_batch = add_noise(test_batch, noise_std=0.25)
            denoised_test = model(noisy_test_batch)
            final_psnr += compute_psnr(test_batch, denoised_test)
            final_mse += torch.mean((test_batch - denoised_test) ** 2).item()
            
    avg_final_psnr = final_psnr / len(test_loader)
    avg_final_mse = final_mse / len(test_loader)
    print("="*60)
    print(f"FINAL BENCHMARK RESULT ({args.dataset.upper()})")
    print(f" -> Test MSE:  {avg_final_mse:.5f}")
    print(f" -> Test PSNR: {avg_final_psnr:.4f} dB")
    print("="*60)
    
    # Send final notification
    if args.ntfy_topic:
        send_push_notification(
            args.ntfy_topic,
            f"Benchmark Finished ({args.model.upper()})",
            f"Dataset: {args.dataset.upper()}\nFinal MSE: {avg_final_mse:.5f}\nFinal PSNR: {avg_final_psnr:.4f} dB"
        )

if __name__ == "__main__":
    main()
