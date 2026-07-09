import sys
import os
import torch
import torch.nn as nn

# Append parent directory to sys.path to import breath_mlp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from breath_mlp import BreathMLP, DeepMLP, BreathMLPPool, FeaturePooling

def get_model_flops(model, input_size):
    flops = 0
    hooks = []
    
    def register_hook(module):
        def hook_fn(module, input_val, output_val):
            nonlocal flops
            # Check for Linear layers
            if isinstance(module, nn.Linear):
                batch_size = input_val[0].shape[0]
                # FLOPs = 2 * in_features * out_features * batch_size
                flops += 2 * module.in_features * module.out_features * batch_size
            
            # Check for LayerNorm layers
            elif isinstance(module, nn.LayerNorm):
                batch_size = input_val[0].shape[0]
                flops += 5 * input_val[0].numel()
                
            # Check for FeaturePooling custom layers
            elif module.__class__.__name__ == "FeaturePooling":
                batch_size = input_val[0].shape[0]
                in_features = input_val[0].shape[1]
                flops += in_features * batch_size
                
        # Register on leaf modules only (no children)
        if len(list(module.children())) == 0 or module.__class__.__name__ == "FeaturePooling":
            hooks.append(module.register_forward_hook(hook_fn))
            
    model.apply(register_hook)
    
    # Run dummy forward pass
    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        model(dummy_input)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    return flops

# Define toy wrapper modules for Transformer FFN block profiling
class StandardFFNBlock(nn.Module):
    def __init__(self, d_model, multiplier=4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, multiplier * d_model)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(multiplier * d_model, d_model)
    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))

class BreathPoolFFNBlock(nn.Module):
    def __init__(self, d_model, multiplier=4, pool_type="max"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, multiplier * d_model)
        self.act = nn.GELU()
        self.pool = FeaturePooling(d_model, pool_type=pool_type)
    def forward(self, x):
        return self.pool(self.act(self.linear1(x)))

def main():
    print("=" * 80)
    print("TRANSFORMER FFN BLOCK FLOPs PROFILE (Batch Size: 1, Sequence/Token Count: 1)")
    print("=" * 80)
    
    # 1. GPT FFN Blocks (d_model = 256, FFN = 4x = 1024)
    print("1. GPT Language Model (d_model = 256, FFN = 1024)")
    gpt_std = StandardFFNBlock(d_model=256, multiplier=4)
    gpt_pool = BreathPoolFFNBlock(d_model=256, multiplier=4, pool_type="max")
    
    gpt_std_flops = get_model_flops(gpt_std, (1, 256))
    gpt_pool_flops = get_model_flops(gpt_pool, (1, 256))
    
    print(f"  Standard FFN Block [256 -> 1024 -> 256]:")
    print(f"    Params: {sum(p.numel() for p in gpt_std.parameters()):,}")
    print(f"    FLOPs per token: {gpt_std_flops:,}")
    print(f"  BreathPool FFN Block [256 -> 1024 -> pool -> 256]:")
    print(f"    Params: {sum(p.numel() for p in gpt_pool.parameters()):,}")
    print(f"    FLOPs per token: {gpt_pool_flops:,}")
    print(f"    FLOPs Block Reduction: {((gpt_pool_flops - gpt_std_flops)/gpt_std_flops)*100:+.2f}%")
    
    # 2. Vision Transformer FFN Blocks (d_model = 192, FFN = 4x = 768)
    print("\n2. Vision Transformer (ViT) (d_model = 192, FFN = 768)")
    vit_std = StandardFFNBlock(d_model=192, multiplier=4)
    vit_pool = BreathPoolFFNBlock(d_model=192, multiplier=4, pool_type="max")
    
    vit_std_flops = get_model_flops(vit_std, (1, 192))
    vit_pool_flops = get_model_flops(vit_pool, (1, 192))
    
    print(f"  Standard FFN Block [192 -> 768 -> 192]:")
    print(f"    Params: {sum(p.numel() for p in vit_std.parameters()):,}")
    print(f"    FLOPs per token: {vit_std_flops:,}")
    print(f"  BreathPool FFN Block [192 -> 768 -> pool -> 192]:")
    print(f"    Params: {sum(p.numel() for p in vit_pool.parameters()):,}")
    print(f"    FLOPs per token: {vit_pool_flops:,}")
    print(f"    FLOPs Block Reduction: {((vit_pool_flops - vit_std_flops)/vit_std_flops)*100:+.2f}%")
    
    print("\n" + "=" * 80)
    print("TABULAR COMPARISON FLOPs PROFILE (Batch Size: 1)")
    print("=" * 80)
    
    input_dim = 8
    output_dim = 1
    
    std_model = DeepMLP(input_dim=input_dim, hidden_layers=[128, 64], output_dim=output_dim, use_skips=False)
    std_flops = get_model_flops(std_model, (1, input_dim))
    print(f"  Standard FFN [8, 128, 64, 1]:")
    print(f"    FLOPs per sample: {std_flops:,}")
    
    breath_layers = [128, 32, 64, 16, 32, 8, 16]
    pool_model = BreathMLPPool(input_dim=input_dim, hidden_layers=breath_layers, output_dim=output_dim, use_skips=True, pool_output=True)
    pool_flops = get_model_flops(pool_model, (1, input_dim))
    print(f"  BreathPool (Max/Avg/Hybrid) [8, 128, 32, 64, 16, 32, 8, 16, 1]:")
    print(f"    FLOPs per sample: {pool_flops:,}")
    print(f"    FLOPs vs Standard: {((pool_flops - std_flops)/std_flops)*100:+.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
