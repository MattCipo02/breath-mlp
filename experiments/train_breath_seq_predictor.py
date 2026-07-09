from breath_sequence_pool import BreathSeqPredictorPool
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from breath_sequence_pool import BreathSeqPredictor

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(f"DISPOSITIVO ATTIVO PER ADDESTRAMENTO: {device.type.upper()}")
if device.type == "cuda":
    print(f" -> GPU: {torch.cuda.get_device_name(0)}")
print("="*60)

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
    # We need context of size block_size and target at block_size+1
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

# Initialize model
model = BreathSeqPredictorPool(
    vocab_size=vocab_size,
    sequence_length=block_size,
    embedding_dim=embedding_dim,
    ffn_start_mult_seq=4,
    ffn_start_mult_vocab=16,
    activation="gelu",
    use_norm=True
).to(device)



total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Modello BreathSeqPredictor Inizializzato:")
print(f" -> Parametri Addestrabili Totali: {total_params:,}")
print(f" -> Hidden Layers Seq Pool: {model.seq_pool.hidden_layers}")
print(f" -> Hidden Layers Vocab Map: {model.vocab_hidden_layers}")
print("="*60)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
start_time = time.time()
history = []

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        eval_loss = estimate_loss(model)
        print(f"Step {iter:4d} | Train Loss: {eval_loss['train']:.4f} | Val Loss: {eval_loss['val']:.4f}")
        history.append({
            "Step": iter,
            "Train_Loss": eval_loss['train'],
            "Val_Loss": eval_loss['val']
        })
        
    xb, yb = get_batch_seq('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

elapsed_time = time.time() - start_time
print("="*60)
print(f"Addestramento completato in {elapsed_time:.1f} secondi.")
print("="*60)

# Save history
df_history = pd.DataFrame(history)
df_history.to_csv("breath_seq_predictor_results_block512.csv", index=False)
print("Risultati salvati in 'breath_seq_predictor_results_block512.csv'")

# Generate sample text
print("\n" + "-"*30 + " TESTO GENERATO DAL MODELLO " + "-"*30)
# Start with a context of 256 space characters
context = torch.zeros((1, block_size), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=250)
generated_text = decode(generated_ids[0].tolist()[block_size:])
print(generated_text)
print("-"*80)
