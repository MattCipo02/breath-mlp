import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import time
import os
import pandas as pd
import argparse
import tiktoken

# Import custom Breath MLP
from breath_mlp import generate_breath_architecture, BreathMLP, BreathMLPPool

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(f"DISPOSITIVO ATTIVO PER IL TRANSFORMER HIGH DATA: {device.type.upper()}")
if device.type == "cuda":
    print(f" -> GPU: {torch.cuda.get_device_name(0)}")
print("="*60)

# Command-line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--max_iters", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ffn_start_mult", type=int, default=4)
args_cli = parser.parse_args()

# Hyperparameters
batch_size = 32
block_size = 128
max_iters = args_cli.max_iters
eval_interval = 500
learning_rate = 1e-3
eval_iters = 100
d_model = 256
n_head = 8
n_layer = 6
FFN_START_MULT = args_cli.ffn_start_mult

# Load WikiText-2 Dataset
train_path = "data/wikitext2/train.txt"
val_path = "data/wikitext2/valid.txt"

if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError("I file di WikiText-2 train.txt / valid.txt non sono stati trovati in data/wikitext2/")

with open(train_path, 'r', encoding='utf-8') as f:
    train_text = f.read()
with open(val_path, 'r', encoding='utf-8') as f:
    val_text = f.read()

# Tokenization using tiktoken (GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")
train_encoded = enc.encode(train_text)
val_encoded = enc.encode(val_text)
vocab_size = enc.n_vocab

print(f"[Dataset WikiText-2 BPE] Vocab Size (GPT-2): {vocab_size}")
print(f"[Dataset WikiText-2 BPE] Train Tokens: {len(train_encoded):,}")
print(f"[Dataset WikiText-2 BPE] Valid Tokens: {len(val_encoded):,}")

# Split train/val datasets
train_data = torch.tensor(train_encoded, dtype=torch.long)
val_data = torch.tensor(val_encoded, dtype=torch.long)

# Auto-compute parameter-matched StandardFFN multiplier
_tmp_hidden = generate_breath_architecture(
    FFN_START_MULT * d_model, 0.25, 2.0, min_width=d_model, output_dim=d_model
)
_tmp_breath = BreathMLP(d_model, _tmp_hidden, d_model, use_skips=True, activation="gelu", use_norm=True)
_breath_ffn_params = sum(p.numel() for p in _tmp_breath.parameters())
STD_MULT = max(4, round(_breath_ffn_params / (2 * d_model * d_model)))
del _tmp_breath, _tmp_hidden
print(f"[Config] FFN start_mult={FFN_START_MULT}x | BreathMLP FFN/block~{_breath_ffn_params:,} | StandardFFN will use {STD_MULT}x d_model")

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

class StandardFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, STD_MULT * d_model),
            nn.GELU(),
            nn.Linear(STD_MULT * d_model, d_model)
        )
    def forward(self, x):
        return self.net(x)

class BreathFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = generate_breath_architecture(
            start_width=FFN_START_MULT * d_model,
            compression_factor=0.25,
            expansion_factor=2.0,
            min_width=d_model,
            output_dim=d_model
        )
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

class BreathPoolMaxFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = generate_breath_architecture(
            start_width=FFN_START_MULT * d_model,
            compression_factor=0.25,
            expansion_factor=2.0,
            min_width=d_model,
            output_dim=d_model
        )
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

class BreathPoolAvgFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = generate_breath_architecture(
            start_width=FFN_START_MULT * d_model,
            compression_factor=0.25,
            expansion_factor=2.0,
            min_width=d_model,
            output_dim=d_model
        )
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

class BreathPoolHybridFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_layers = generate_breath_architecture(
            start_width=FFN_START_MULT * d_model,
            compression_factor=0.25,
            expansion_factor=2.0,
            min_width=d_model,
            output_dim=d_model
        )
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

class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size, ffn_type):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
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
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size, ffn_type):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, d_model),
            wpe = nn.Embedding(block_size, d_model),
            h = nn.ModuleList([Block(d_model, n_head, block_size, ffn_type) for _ in range(n_layer)]),
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
# --- RUN BENCHMARKS ---
# =====================================================================
def train_gpt(ffn_type, seed):
    print("\n" + "="*70)
    print(f" ADDESTRAMENTO GPT BPE HIGH DATA: {ffn_type.upper()} | SEED: {seed}")
    print("="*70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    model = GPT(vocab_size, d_model, n_head, n_layer, block_size, ffn_type)
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
    configs = ["standard", "breath", "breath_pool_max", "breath_pool_avg", "breath_pool_hybrid"]
    
    results = {}
    for cfg in configs:
        p, t, l, txt = train_gpt(cfg, seed)
        results[cfg] = {
            "params": p,
            "time": t,
            "losses": l,
            "text": txt
        }
        
    # Save curves to CSV
    steps = [x[0] for x in results["standard"]["losses"]]
    df_curves = pd.DataFrame({"Step": steps})
    for cfg in configs:
        df_curves[f"{cfg}_Train"] = [x[1] for x in results[cfg]["losses"]]
        df_curves[f"{cfg}_Val"] = [x[2] for x in results[cfg]["losses"]]
        
    df_curves.to_csv("loss_curves_bpe_high_data.csv", index=False)
    print("\n-> Curve di loss salvate con successo in 'loss_curves_bpe_high_data.csv'")
    
    # Save Summary Report
    summary = []
    for cfg in configs:
        summary.append({
            "Config": cfg,
            "Params": results[cfg]["params"],
            "Time": results[cfg]["time"],
            "Min_Val_Loss": min([x[2] for x in results[cfg]["losses"]]),
            "Final_Val_Loss": results[cfg]["losses"][-1][2]
        })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv("transformer_bpe_high_data_results.csv", index=False)
    print("-> Sommario dei risultati salvato in 'transformer_bpe_high_data_results.csv'")
    
    # Print Report
    print("\n" + "="*90)
    print(f"     REPORT COMPARATIVO NANO-GPT WIKITEXT-2 (BPE) | SEED: {seed}")
    print("="*90)
    print(f"Configurazione FFN        | Parametri | Tempo (sec) | Min Val Loss | Val Loss Finale")
    print(f"--------------------------|-----------|-------------|--------------|----------------")
    for row in summary:
        print(f"{row['Config']:26s} | {row['Params']:9,} | {row['Time']:10.1f}s | {row['Min_Val_Loss']:.4f}       | {row['Final_Val_Loss']:.4f}")
    print("="*90)
