import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import time
import os
import urllib.request
import pandas as pd
import argparse
import tiktoken

# Import custom Breath MLP
from breath_mlp import generate_breath_architecture, BreathMLPPool

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(f"DISPOSITIVO ATTIVO PER IL TEST DOUBLE BREATHPOOL: {device.type.upper()}")
if device.type == "cuda":
    print(f" -> GPU: {torch.cuda.get_device_name(0)}")
print("="*60)

# Command-line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--max_iters", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ffn_start_mult", type=int, default=4)
args_cli = parser.parse_args()

# Hyperparameters (Identical to Point C clean run)
batch_size = 32
block_size = 128
max_iters = args_cli.max_iters
eval_interval = 250
learning_rate = 1e-3
eval_iters = 50
d_model = 256
n_head = 8
n_layer = 6
FFN_START_MULT = args_cli.ffn_start_mult

# Load Tiny Shakespeare
dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
dataset_path = "tinyshakespeare.txt"

if not os.path.exists(dataset_path):
    print("Download di Tiny Shakespeare in corso...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Download completato.")

with open(dataset_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenization using tiktoken (GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")
encoded_text = enc.encode(text)
vocab_size = enc.n_vocab

print(f"[Dataset BPE] Vocab Size (GPT-2): {vocab_size}")
print(f"[Dataset BPE] Total Tokens: {len(encoded_text):,}")

# Split train/val datasets
data = torch.tensor(encoded_text, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f" -> Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# =====================================================================
# --- TRANSFORMER MODULES ---
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_model = d_model
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# BreathPoolDouble FFN
class BreathPoolDoubleFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = generate_breath_architecture(
            start_width=FFN_START_MULT * d_model,
            compression_factor=0.25,
            expansion_factor=2.0,
            min_width=d_model,
            output_dim=d_model
        )
        self.net1 = BreathMLPPool(
            input_dim=d_model,
            hidden_layers=hidden_layers,
            output_dim=d_model,
            use_skips=True,
            pool_type="max",
            activation="gelu",
            use_norm=True
        )
        self.net2 = BreathMLPPool(
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
        # Sequence stacking: net1 -> net2
        out_flat = self.net2(self.net1(x_flat))
        return out_flat.view(B, T, C)

class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = BreathPoolDoubleFFN(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, d_model),
            wpe = nn.Embedding(block_size, d_model),
            h = nn.ModuleList([Block(d_model, n_head, block_size) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================================================
# --- RUN BENCHMARK ---
# =====================================================================
def train_gpt(seed):
    print("\n" + "="*70)
    print(f" ADDESTRAMENTO GPT BPE DOUBLE BREATHPOOL | SEED: {seed}")
    print("="*70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    model = GPT(vocab_size, d_model, n_head, n_layer, block_size)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ffn_params = sum(p.numel() for p in model.transformer.h[0].ffn.parameters())
    print(f" -> Parametri Addestrabili Totali (con Weight Tying): {trainable_params:,}")
    print(f" -> Parametri FFN per singolo Blocco: {ffn_params:,}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    losses = []
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            eval_loss = estimate_loss(model)
            print(f"    * Step {iter:4d} | Train Loss: {eval_loss['train']:.4f} | Val Loss: {eval_loss['val']:.4f}")
            losses.append((iter, eval_loss['train'], eval_loss['val']))
            
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    elapsed = time.time() - start_time
    print(f" -> Addestramento completato in {elapsed:.1f} secondi.")
    
    # Generate BPE sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=100)
    generated_text = enc.decode(generated_ids[0].tolist())
    
    return trainable_params, elapsed, losses, generated_text

if __name__ == '__main__':
    seed = args_cli.seed
    p, t, l, txt = train_gpt(seed)
    
    # Save curves to CSV
    steps = [x[0] for x in l]
    df_curves = pd.DataFrame({
        "Step": steps,
        "double_Train": [x[1] for x in l],
        "double_Val": [x[2] for x in l]
    })
    df_curves.to_csv("loss_curves_bpe_double.csv", index=False)
    print("\n-> Curve di loss salvate con successo in 'loss_curves_bpe_double.csv'")
    
    # Save Summary Report
    summary = [{
        "Config": "breath_pool_double",
        "Params": p,
        "Time": t,
        "Min_Val_Loss": min([x[2] for x in l]),
        "Final_Val_Loss": l[-1][2]
    }]
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv("transformer_bpe_double_results.csv", index=False)
    
    # Print Report
    print("\n" + "="*90)
    print(f"     REPORT NANO-GPT BPE DOUBLE BREATHPOOL | SEED: {seed}")
    print("="*90)
    print(f"Configurazione FFN        | Parametri | Tempo (sec) | Min Val Loss | Val Loss Finale")
    print(f"--------------------------|-----------|-------------|--------------|----------------")
    for row in summary:
        print(f"{row['Config']:26s} | {row['Params']:9,} | {row['Time']:10.1f}s | {row['Min_Val_Loss']:.4f}       | {row['Final_Val_Loss']:.4f}")
    print("="*90)
    
    # Print generated sample
    print(f"\n" + "-"*35 + f" TESTO GENERATO DA GPT DOUBLE BREATHPOOL " + "-"*35)
    print(txt)
    print("-"*100)
