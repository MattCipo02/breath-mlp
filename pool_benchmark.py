"""
pool_benchmark.py

K-Fold Cross-Validation benchmark comparing BreathMLP (linear compression)
against BreathMLPPool (Max Pooling and Avg Pooling) on SARCOS, California, and MNIST.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import argparse
import urllib.request

from scipy.io import loadmat
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from breath_mlp import (
    generate_breath_architecture,
    generate_deep_architecture,
    BreathMLP,
    BreathMLPPool,
    BreathMLPPurePool,
    DeepMLP
)

# ======================================================================
# ARGS
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="sarcos",
                    choices=["sarcos", "california", "mnist"])
parser.add_argument("--folds",          type=int,   default=5)
parser.add_argument("--start_width",    type=int,   default=512)
parser.add_argument("--epochs",         type=int,   default=40)
parser.add_argument("--batch_size",     type=int,   default=64)
parser.add_argument("--lr",             type=float, default=0.001)
parser.add_argument("--compression",    type=float, default=0.25)
parser.add_argument("--expansion",      type=float, default=2.0)
parser.add_argument("--decay",          type=float, default=0.5)
parser.add_argument("--activation",     type=str,   default="relu", choices=["relu", "gelu", "silu"])
parser.add_argument("--use_norm",       action="store_true", help="Use LayerNorm inside networks")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*80)
print(f"BREATH POOLING BENCHMARK  |  dataset={args.dataset.upper()}  |  folds={args.folds}")
print(f"Device: {device.type.upper()}", end="")
if device.type == "cuda":
    print(f"  |  GPU: {torch.cuda.get_device_name(0)}", end="")
print()
print(f"Activation: {args.activation.upper()}  |  LayerNorm: {args.use_norm}")
print("="*80)

# ======================================================================
# DATASET AND ARCHITECTURES
# ======================================================================
print(f"\nLoading {args.dataset}...")
is_regression = True
output_dim = 1

if args.dataset == "sarcos":
    train_path, test_path = "sarcos_inv.mat", "sarcos_inv_test.mat"
    if not os.path.exists(train_path):
        urllib.request.urlretrieve("http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat", train_path)
    if not os.path.exists(test_path):
        urllib.request.urlretrieve("http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat", test_path)
    train_data = loadmat(train_path)["sarcos_inv"]
    test_data  = loadmat(test_path)["sarcos_inv_test"]
    full_data  = np.vstack([train_data, test_data])
    X_all = full_data[:, :21].astype("float32")
    y_all = full_data[:, 21 ].astype("float32")
    
    breath_layers = generate_breath_architecture(
        args.start_width, args.compression, args.expansion, min_width=16
    )
    deep_layers = generate_deep_architecture(args.start_width, args.decay, min_width=16)

elif args.dataset == "california":
    housing = fetch_california_housing()
    X_all   = housing.data.astype("float32")
    y_all   = housing.target.astype("float32")
    
    breath_layers = generate_breath_architecture(
        args.start_width, args.compression, args.expansion, min_width=16
    )
    deep_layers = generate_deep_architecture(args.start_width, args.decay, min_width=16)

elif args.dataset == "mnist":
    from tensorflow.keras.datasets import mnist as keras_mnist
    (X_tr, y_tr), _ = keras_mnist.load_data()
    X_all = X_tr.reshape(-1, 784).astype("float32") / 255.0
    y_all = y_tr.flatten().astype("int64")
    is_regression = False
    output_dim = 10
    
    # Use wider hidden layers for MNIST as in main benchmark
    breath_layers = generate_breath_architecture(
        args.start_width * 2, args.compression, args.expansion, min_width=10, output_dim=10
    )
    deep_layers = generate_deep_architecture(args.start_width * 2, args.decay, min_width=10)

input_dim = X_all.shape[1]
print(f"Dataset: {X_all.shape[0]:,} samples x {input_dim} features")
print(f"Breath layers: {breath_layers}")
print(f"Deep layers:   {deep_layers}")

# ======================================================================
# MODEL FACTORY
# ======================================================================
def make_model(config):
    if config["type"] == "deep":
        return DeepMLP(
            input_dim=input_dim, hidden_layers=config["layers"],
            output_dim=output_dim, use_skips=config["use_skips"],
            activation=args.activation, use_norm=args.use_norm
        )
    elif config["type"] == "breath_linear":
        return BreathMLP(
            input_dim=input_dim, hidden_layers=config["layers"],
            output_dim=output_dim, use_skips=config["use_skips"],
            activation=args.activation, use_norm=args.use_norm
        )
    elif config["type"] == "breath_pool_max":
        return BreathMLPPool(
            input_dim=input_dim, hidden_layers=config["layers"],
            output_dim=output_dim, use_skips=config["use_skips"],
            pool_type="max", activation=args.activation, use_norm=args.use_norm
        )
    elif config["type"] == "breath_pool_avg":
        return BreathMLPPool(
            input_dim=input_dim, hidden_layers=config["layers"],
            output_dim=output_dim, use_skips=config["use_skips"],
            pool_type="avg", activation=args.activation, use_norm=args.use_norm
        )
    elif config["type"] == "breath_pure_pool_max":
        return BreathMLPPurePool(
            input_dim=input_dim, hidden_layers=config["layers"],
            output_dim=output_dim, use_skips=config["use_skips"],
            pool_type="max", activation=args.activation, use_norm=args.use_norm
        )
    elif config["type"] == "breath_pure_pool_avg":
        return BreathMLPPurePool(
            input_dim=input_dim, hidden_layers=config["layers"],
            output_dim=output_dim, use_skips=config["use_skips"],
            pool_type="avg", activation=args.activation, use_norm=args.use_norm
        )

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

configs = [
    {"name": "Deep Standard",         "type": "deep",             "layers": deep_layers,   "use_skips": False},
    {"name": "Breath + Skips (Lin)",  "type": "breath_linear",    "layers": breath_layers, "use_skips": True},
    {"name": "BreathPool Max + Skips","type": "breath_pool_max",   "layers": breath_layers, "use_skips": True},
    {"name": "BreathPool Avg + Skips","type": "breath_pool_avg",   "layers": breath_layers, "use_skips": True},
    {"name": "PurePool Max + Skips",  "type": "breath_pure_pool_max", "layers": breath_layers, "use_skips": True},
    {"name": "PurePool Avg + Skips",  "type": "breath_pure_pool_avg", "layers": breath_layers, "use_skips": True},
]

# ======================================================================
# TRAIN & EVAL
# ======================================================================
def train_and_eval(model, X_tr_np, y_tr_np, X_va_np, y_va_np):
    if args.dataset == "mnist":
        X_tr_s = X_tr_np
        X_va_s = X_va_np
    else:
        scaler_X = StandardScaler()
        X_tr_s = scaler_X.fit_transform(X_tr_np)
        X_va_s = scaler_X.transform(X_va_np)

    if is_regression:
        scaler_y = StandardScaler()
        y_tr_s = scaler_y.fit_transform(y_tr_np.reshape(-1, 1)).ravel()
        y_tr_t = torch.tensor(y_tr_s, dtype=torch.float32).unsqueeze(1)
        criterion = nn.MSELoss()
    else:
        y_tr_t = torch.tensor(y_tr_np, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    X_va_t = torch.tensor(X_va_s, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=args.batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        preds = model(X_va_t.to(device)).cpu().numpy()

    if is_regression:
        preds_orig = scaler_y.inverse_transform(preds)
        return {
            "R2":   float(r2_score(y_va_np, preds_orig)),
            "MSE":  float(mean_squared_error(y_va_np, preds_orig)),
            "Time": float(elapsed)
        }
    else:
        preds_cls = preds.argmax(axis=1)
        return {
            "Accuracy": float((preds_cls == y_va_np).mean()),
            "Time": float(elapsed)
        }

# ======================================================================
# K-FOLD LOOP
# ======================================================================
kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
fold_results = {c["name"]: [] for c in configs}

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\n{'='*80}")
    print(f"  FOLD {fold_idx+1}/{args.folds}  |  train={len(train_idx):,}  val={len(val_idx):,}")
    print(f"{'='*80}")

    seed = fold_idx * 137 + 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr_f, X_va_f = X_all[train_idx], X_all[val_idx]
    y_tr_f, y_va_f = y_all[train_idx], y_all[val_idx]

    for config in configs:
        model = make_model(config)
        n_p   = count_params(model)
        print(f"  {config['name']:25s} ({n_p:,} params) ... ", end="", flush=True)
        m = train_and_eval(model, X_tr_f, y_tr_f, X_va_f, y_va_f)
        fold_results[config["name"]].append(m)
        key = "R2" if is_regression else "Accuracy"
        print(f"{key}={m[key]:.4f}  t={m['Time']:.0f}s")

# ======================================================================
# AGGREGATE & REPORT
# ======================================================================
print("\n" + "="*80)
print(f"  {args.folds}-FOLD CV RESULTS  ({args.dataset.upper()})")
print("="*80)

rows = []
for config in configs:
    fm = fold_results[config["name"]]
    model_tmp = make_model(config)
    n_p = count_params(model_tmp)
    del model_tmp

    if is_regression:
        r2s   = [m["R2"]   for m in fm]
        mses  = [m["MSE"]  for m in fm]
        times = [m["Time"] for m in fm]
        row = {
            "Model": config["name"], "Params": n_p,
            "R2_mean": float(np.mean(r2s)),  "R2_std":  float(np.std(r2s)),
            "MSE_mean": float(np.mean(mses)), "MSE_std": float(np.std(mses)),
            "Time_mean": float(np.mean(times))
        }
        print(f"  {config['name']:25s} | {n_p:9,} params | "
              f"R2 = {row['R2_mean']:.4f} +/- {row['R2_std']:.4f} | "
              f"MSE = {row['MSE_mean']:.4f} +/- {row['MSE_std']:.4f} | "
              f"t = {row['Time_mean']:.0f}s")
    else:
        accs  = [m["Accuracy"] for m in fm]
        times = [m["Time"]     for m in fm]
        row = {
            "Model": config["name"], "Params": n_p,
            "Acc_mean": float(np.mean(accs)), "Acc_std": float(np.std(accs)),
            "Time_mean": float(np.mean(times))
        }
        print(f"  {config['name']:25s} | {n_p:9,} params | "
              f"Acc = {row['Acc_mean']:.4f} +/- {row['Acc_std']:.4f} | "
              f"t = {row['Time_mean']:.0f}s")
    rows.append(row)

df = pd.DataFrame(rows)
out_file = f"pool_results_{args.dataset}_{args.folds}fold.csv"
df.to_csv(out_file, index=False)
print(f"\nSaved results to: {out_file}")
print("="*80)
