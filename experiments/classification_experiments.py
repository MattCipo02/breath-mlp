import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import sys
import os

# Import Keras to load CIFAR-10 from local cache
from tensorflow.keras.datasets import cifar10
from torch.utils.data import TensorDataset, DataLoader

# Import Breath MLP custom classes
from breath_mlp import generate_breath_architecture, BreathMLP

# Check CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(f"DISPOSITIVO RILEVATO DA PYTORCH: {device.type.upper()}")
if device.type == "cuda":
    print(f" -> GPU: {torch.cuda.get_device_name(0)}")
print("="*60)

# Configuration
COMPRESSION_FACTOR = 0.25
EXPANSION_FACTOR = 2.0
EPOCHS = 12
BATCH_SIZE = 128

# Load CIFAR-10
print("\nCaricamento dati CIFAR-10 dalla cache...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()

X_train_t = torch.tensor(X_train_raw.reshape(-1, 3072).astype("float32") / 255.0)
y_train_t = torch.tensor(y_train_raw.flatten().astype("int64"))
X_test_t = torch.tensor(X_test_raw.reshape(-1, 3072).astype("float32") / 255.0)
y_test_t = torch.tensor(y_test_raw.flatten().astype("int64"))

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def run_experiment(start_width):
    print("\n" + "#"*60)
    print(f" AVVIO ESPERIMENTI CLASSIFICAZIONE PER LARGHEZZA: {start_width}")
    print("#"*60)
    
    breath_layers = generate_breath_architecture(start_width, COMPRESSION_FACTOR, EXPANSION_FACTOR, min_width=16)
    print(f" -> Struttura Breath MLP generata: {breath_layers}")

    configs = [
        {"name": "Breath (ReLU, No Norm)", "act": "relu", "use_norm": False},
        {"name": "Breath (GELU, No Norm)", "act": "gelu", "use_norm": False},
        {"name": "Breath (GELU + LayerNorm)", "act": "gelu", "use_norm": True},
        {"name": "Breath (SiLU + LayerNorm)", "act": "silu", "use_norm": True},
    ]

    results = []

    for config in configs:
        model = BreathMLP(
            input_dim=3072,
            hidden_layers=breath_layers,
            output_dim=10,
            use_skips=True,
            activation=config["act"],
            use_norm=config["use_norm"]
        )
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n -> Configurazione: {config['name']} ({trainable_params:,} parametri)")
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training Loop
        start_time = time.time()
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for inputs_batch, targets_batch in train_loader:
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs_batch)
                loss = criterion(outputs, targets_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
        elapsed = time.time() - start_time
        
        # Evaluation Loop
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
        print(f"    * Finito in {elapsed:.1f}s | Acc: {accuracy:.4f} | Parametri: {trainable_params:,}")
        
        results.append({
            "Width": start_width,
            "Configurazione": config["name"],
            "Parametri": trainable_params,
            "Tempo_Sec": round(elapsed, 1),
            "Test_Accuracy": accuracy
        })
        
    return results

# Run experiments for both widths
results_4096 = run_experiment(4096)
results_8192 = run_experiment(8192)

# Compare and print results
all_results = results_4096 + results_8192
df = pd.DataFrame(all_results)

print("\n" + "="*80)
print("             RISULTATI ESPERIMENTI DI CLASSIFICAZIONE (CIFAR-10)")
print("="*80)
print(df.to_string(index=False))
print("="*80)

df.to_csv("classification_new_results.csv", index=False)
print("\nRisultati salvati in 'classification_new_results.csv'.")
