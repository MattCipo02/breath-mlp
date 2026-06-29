import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
import urllib.request
from scipy.io import loadmat
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Import our pure PyTorch model definitions
from breath_mlp import (
    generate_breath_architecture,
    generate_deep_architecture,
    BreathMLP,
    DeepMLP
)

# =====================================================================
# --- BENCHMARK CONFIGURATION ---
# =====================================================================
DATASET = "sarcos"          # Options: "sarcos", "california", "mnist"
START_WIDTH = 512           # Width of the first hidden layer (e.g., 512, 1024)
COMPRESSION_FACTOR = 0.25   # Factor to compress layers
EXPANSION_FACTOR = 2.0      # Factor to expand layers
DECAY_FACTOR = 0.5          # Decay factor for Deep MLP
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Automatically use GPU for PyTorch if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print(f"Running PyTorch benchmark on: {DATASET.upper()}")
print(f"Device detected:             {device.type.upper()}")
if device.type == "cuda":
    print(f"GPU Name:                    {torch.cuda.get_device_name(0)}")
print("="*60)

# =====================================================================
# --- DATASET LOADING ---
# =====================================================================
print("Loading and preprocessing dataset...")
if DATASET == "sarcos":
    train_url = "http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat"
    test_url = "http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat"
    train_path = "sarcos_inv.mat"
    test_path = "sarcos_inv_test.mat"

    if not os.path.exists(train_path):
        print("Downloading SARCOS train data (13MB)...")
        urllib.request.urlretrieve(train_url, train_path)
    if not os.path.exists(test_path):
        print("Downloading SARCOS test data (1.3MB)...")
        urllib.request.urlretrieve(test_url, test_path)

    train_data = loadmat(train_path)['sarcos_inv']
    test_data = loadmat(test_path)['sarcos_inv_test']

    # 21 inputs (positions, velocities, accelerations) -> 1 target torque
    X_train, y_train = train_data[:, :21], train_data[:, 21]
    X_test, y_test = test_data[:, :21], test_data[:, 21]
    num_classes = 1
    task_type = "regression"

elif DATASET == "california":
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_classes = 1
    task_type = "regression"

elif DATASET == "mnist":
    from torchvision import datasets, transforms
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Flatten and normalize
    X_train = mnist_train.data.view(-1, 784).float() / 255.0
    y_train = mnist_train.targets
    X_test = mnist_test.data.view(-1, 784).float() / 255.0
    y_test = mnist_test.targets
    
    num_classes = 10
    task_type = "classification"

# Normalize inputs for regression datasets
if DATASET != "mnist":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

# =====================================================================
# --- ARCHITECTURE GENERATION ---
# =====================================================================
breath_layers = generate_breath_architecture(START_WIDTH, COMPRESSION_FACTOR, EXPANSION_FACTOR, min_width=16)
deep_layers = generate_deep_architecture(START_WIDTH, DECAY_FACTOR, min_width=16)

print(f"\n -> Breath Architecture layers: {breath_layers}")
print(f" -> Deep Architecture layers:   {deep_layers}\n")

# =====================================================================
# --- TRAINING PIPELINES ---
# =====================================================================
configs = [
    {"name": "Deep Standard", "layers": deep_layers, "use_skips": False, "is_breath": False},
    {"name": "Deep + Skips", "layers": deep_layers, "use_skips": True, "is_breath": False},
    {"name": "Breath Standard", "layers": breath_layers, "use_skips": False, "is_breath": True},
    {"name": "Breath + Skips", "layers": breath_layers, "use_skips": True, "is_breath": True},
]

results = []

# Prepare tensors
if isinstance(X_train, np.ndarray):
    X_train_t = torch.tensor(X_train)
    X_test_t = torch.tensor(X_test)
if isinstance(y_train, np.ndarray):
    y_train_t = torch.tensor(y_train)
    y_test_t = torch.tensor(y_test)
else:
    X_train_t, X_test_t, y_train_t, y_test_t = X_train, X_test, y_train, y_test

# Convert targets to correct dtype
y_train_t = y_train_t.long() if task_type == "classification" else y_train_t.float()
y_test_t = y_test_t.long() if task_type == "classification" else y_test_t.float()

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)

for config in configs:
    print(f"Training {config['name']}...")
    
    # Build PyTorch Model
    if config['is_breath']:
        model = BreathMLP(
            input_dim=X_train_t.shape[1],
            hidden_layers=config['layers'],
            output_dim=num_classes,
            use_skips=config['use_skips']
        )
    else:
        model = DeepMLP(
            input_dim=X_train_t.shape[1],
            hidden_layers=config['layers'],
            output_dim=num_classes,
            use_skips=config['use_skips']
        )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for inputs_batch, targets_batch in train_loader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            
            if task_type == "regression":
                outputs = outputs.squeeze(-1)
                
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start
    
    # Evaluation
    model.eval()
    X_test_gpu = X_test_t.to(device)
    with torch.no_grad():
        preds_t = model(X_test_gpu)
        if task_type == "regression":
            preds = preds_t.cpu().numpy().flatten()
        else:
            _, predicted = torch.max(preds_t.data, 1)
            preds = predicted.cpu().numpy()

    # Calculate metrics
    y_test_np = y_test_t.numpy()
    if task_type == "regression":
        mse = mean_squared_error(y_test_np, preds)
        r2 = r2_score(y_test_np, preds)
        nmse = (mse / np.var(y_test_np)) * 100
        metric_str = f"MSE: {mse:.4f} | R2: {r2:.4f} | nMSE: {nmse:.2f}%"
        metric_val = f"R2: {r2:.4f}"
    else:
        acc = np.mean(y_test_np == preds)
        metric_str = f"Accuracy: {acc:.4f}"
        metric_val = f"Acc: {acc:.4f}"

    print(f" -> Finished in {elapsed:.1f}s | Params: {trainable_params:,} | {metric_str}")
    
    results.append({
        "Model": config['name'],
        "Parameters": trainable_params,
        "Time (s)": round(elapsed, 1),
        "Result": metric_val
    })

# =====================================================================
# --- PRINT & SAVE RESULTS ---
# =====================================================================
df = pd.DataFrame(results)
print("\n" + "="*80)
print(f"                           BENCHMARK RESULTS ({DATASET.upper()})")
print("="*80)
print(df.to_string(index=False))
print("="*80)

df.to_csv("benchmark_results.csv", index=False)
print("Results exported to 'benchmark_results.csv'.")
