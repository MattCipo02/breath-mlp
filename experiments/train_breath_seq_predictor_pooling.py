import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from breath_sequence_pool import BreathSeqPredictorPool

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*80)
print(f"DISPOSITIVO ATTIVO PER ADDESTRAMENTO POOLING: {device.type.upper()}")
if device.type == "cuda":
    print(f" -> GPU: {torch.cuda.get_device_name(0)}")
print("="*80)

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Hyperparameters
batch_size = 64
block_size = 256       # Input sequence length (T)
embedding_dim = 512    # Embedding size (C)
max_iters = 30000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 50

# Load Tiny Shakespeare
dataset_path = "tinyshakespeare.txt"
if not os.path.exists(dataset_path):
    import urllib.request
    dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Download di Tiny Shakespeare in corso...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Download completato.")

with open(dataset_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Character mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Split train/val datasets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch_seq(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size - 1, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+block_size] for i in ix]) # Target is a single scalar token
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_seq(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Training function for a single pool_type
def train_pooling_model(pool_type):
    print("\n" + "="*70)
    print(f" ADDESTRAMENTO BREATH-SEQ-PREDICTOR-POOL CON POOL_TYPE: {pool_type.upper()}")
    print("="*70)
    
    # Reset seed for fair comparison
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
         torch.cuda.manual_seed_all(seed)
         
    model = BreathSeqPredictorPool(
        vocab_size=vocab_size,
        sequence_length=block_size,
        embedding_dim=embedding_dim,
        ffn_start_mult_seq=4,
        ffn_start_mult_vocab=16,
        pool_type=pool_type,
        activation="gelu",
        use_norm=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Parametri Addestrabili Totali: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    losses = []
    start_time = time.time()
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            eval_loss = estimate_loss(model)
            print(f"    * Step {iter:4d} | Train Loss: {eval_loss['train']:.4f} | Val Loss: {eval_loss['val']:.4f}")
            losses.append((iter, eval_loss['train'], eval_loss['val']))
            
        xb, yb = get_batch_seq('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    elapsed = time.time() - start_time
    print(f" -> Addestramento completato in {elapsed:.1f} secondi.")
    
    # Generate text sample
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=250)
    generated_text = decode(generated_ids[0].tolist()[block_size:])
    
    return total_params, elapsed, losses, generated_text

if __name__ == "__main__":
    results = {}
    
    # 1. Run Max Pooling
    p_max, t_max, losses_max, text_max = train_pooling_model("max")
    results["max"] = (p_max, t_max, losses_max, text_max)
    
    # 2. Run Avg Pooling
    p_avg, t_avg, losses_avg, text_avg = train_pooling_model("avg")
    results["avg"] = (p_avg, t_avg, losses_avg, text_avg)
    
    # 3. Run Hybrid Pooling
    p_hyb, t_hyb, losses_hyb, text_hyb = train_pooling_model("hybrid")
    results["hybrid"] = (p_hyb, t_hyb, losses_hyb, text_hyb)
    
    # Save loss curves to CSV
    steps = [x[0] for x in losses_max]
    df_curves = pd.DataFrame({
        "Step": steps,
        "Max_Train": [x[1] for x in losses_max],
        "Max_Val": [x[2] for x in losses_max],
        "Avg_Train": [x[1] for x in losses_avg],
        "Avg_Val": [x[2] for x in losses_avg],
        "Hybrid_Train": [x[1] for x in losses_hyb],
        "Hybrid_Val": [x[2] for x in losses_hyb]
    })
    df_curves.to_csv("breath_seq_predictor_pooling_loss_curves_30k_asimm.csv", index=False)
    print("\n-> Curve di loss salvate in 'breath_seq_predictor_pooling_loss_curves_10k_8x.csv'")
    
    # Print Comparison Report
    print("\n" + "="*90)
    print("     REPORT COMPARATIVO BREATH-SEQ-PREDICTOR-POOL (30000 ITER, 4x-16x EXPANSION)")
    print("="*90)
    print("Configurazione Pooling    | Parametri | Tempo (sec) | Val Loss Finale (iter 30000)")
    print("--------------------------|-----------|-------------|----------------------------")
    for pt in ["max", "avg", "hybrid"]:
        p, t, l, _ = results[pt]
        print(f"BreathPool {pt.capitalize():13s} | {p:9,} | {t:10.1f}s | {l[-1][2]:.4f}")
    print("="*90)
    
    for pt in ["max", "avg", "hybrid"]:
        _, _, _, txt = results[pt]
        print(f"\n" + "-"*30 + f" TESTO GENERATO DA POOL_TYPE: {pt.upper()} " + "-"*30)
        print(txt)
        print("-"*80)
