import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys

# Import custom Breath MLP from Experiments folder
sys.path.append("c:/Users/matte/Desktop/Matteo/università/AIRO/Machine Learning/Exercises/Try/Experiments")
from breath_mlp import generate_breath_architecture, BreathMLP, BreathMLPPool

# Replicate the model definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 50257
d_model = 256
n_head = 8
n_layer = 6
block_size = 128
FFN_START_MULT = 4

_tmp_hidden = generate_breath_architecture(
    FFN_START_MULT * d_model, 0.25, 2.0, min_width=d_model, output_dim=d_model
)
_tmp_breath = BreathMLP(d_model, _tmp_hidden, d_model, use_skips=True, activation="gelu", use_norm=True)
_breath_ffn_params = sum(p.numel() for p in _tmp_breath.parameters())
STD_MULT = max(4, round(_breath_ffn_params / (2 * d_model * d_model)))
del _tmp_breath, _tmp_hidden

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

import math

def profile(ffn_type):
    model = GPT(vocab_size, d_model, n_head, n_layer, block_size, ffn_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Generate dummy batch
    xb = torch.randint(0, vocab_size, (64, block_size), device=device)
    yb = torch.randint(0, vocab_size, (64, block_size), device=device)
    
    # Warmup
    for _ in range(10):
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # Timed steps
    torch.cuda.synchronize()
    t0 = time.time()
    steps = 50
    for _ in range(steps):
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    
    avg_step_ms = (t1 - t0) / steps * 1000
    print(f"| {ffn_type:20s} | {avg_step_ms:10.2f} ms/step |")
    return avg_step_ms

print("Profiling steps...")
print("| FFN Type             | Time per Step (ms) |")
print("|----------------------|--------------------|")
configs = ["standard", "breath", "breath_pool_max", "breath_pool_avg", "breath_pool_hybrid"]
times = {}
for cfg in configs:
    times[cfg] = profile(cfg)
    
total_time_sec = sum(times[cfg] * 3000 / 1000 for cfg in configs)
print(f"Expected Total Time: {total_time_sec / 60:.2f} minutes")
