import torch
import torch.nn as nn

# =====================================================================
# --- GENERAL ARCHITECTURE GENERATORS ---
# =====================================================================
def generate_breath_architecture(start_width, compression_factor=0.25, expansion_factor=2.0, min_width=16, output_dim=None):
    """
    Generates an oscillating (Breath) hidden layer size sequence strictly using
    the provided compression and expansion factors.

    All intermediate layers are guaranteed to be STRICTLY GREATER than output_dim.
    This ensures the output_layer is always a meaningful compression, never a
    square (identity-like) projection.
    """
    limit = min_width
    if output_dim is not None:
        limit = max(limit, output_dim)
        
    layers_list = [start_width]
    current = start_width
    
    while True:
        # 1. Compress
        compressed = int(current * compression_factor)
        if compressed < limit:
            break
        # All intermediate layers must be STRICTLY greater than output_dim.
        # If compressed == output_dim, the output_layer itself will handle
        # that final mapping — no need for a redundant bottleneck at output_dim.
        if output_dim is not None and compressed <= output_dim:
            break
        layers_list.append(compressed)
        
        # 2. Expand
        expanded = int(compressed * expansion_factor)
        if expanded < limit:
            break
        layers_list.append(expanded)
        current = expanded
        
    return layers_list

def generate_deep_architecture(start_width, decay_factor=0.5, min_width=16):
    """
    Generates a monotonically decaying (Deep) hidden layer size sequence.
    Example: 512, 0.5 -> [512, 256, 128, 64, 32, 16]
    """
    layers_list = []
    current = start_width
    while current >= min_width:
        layers_list.append(current)
        current = int(current * decay_factor)
    return layers_list
# =====================================================================
# --- PYTORCH IMPLEMENTATION ---
# =====================================================================
class BreathMLP(nn.Module):
    """
    Breath MLP implementation in PyTorch.
    Alternates between compression and expansion, with projected skip connections between bottlenecks.
    """
    def __init__(self, input_dim, hidden_layers, output_dim=1, use_skips=True, activation="relu", use_norm=False):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.use_skips = use_skips
        self.activation = activation
        self.use_norm = use_norm
        
        # Optional Input LayerNorm
        if use_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        
        # Selectable Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "elu":
            self.act = nn.ELU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.act = nn.ReLU()
            
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_layers[0]))
        self.projections = nn.ModuleDict()
        
        if use_norm:
            self.norms = nn.ModuleDict()
        
        i = 1
        last_comp_idx = None
        while i < len(hidden_layers):
            comp_units = hidden_layers[i]
            prev_units = hidden_layers[i-1]
            self.linears.append(nn.Linear(prev_units, comp_units))
            
            # Optional Bottleneck LayerNorm
            if use_norm:
                self.norms[f"norm_{i}"] = nn.LayerNorm(comp_units)
            
            if use_skips and last_comp_idx is not None:
                prev_comp_units = hidden_layers[last_comp_idx]
                proj_key = f"proj_{last_comp_idx}_to_{i}"
                self.projections[proj_key] = nn.Linear(prev_comp_units, comp_units)
                
            last_comp_idx = i
            
            if i + 1 < len(hidden_layers):
                exp_units = hidden_layers[i+1]
                self.linears.append(nn.Linear(comp_units, exp_units))
                i += 2
            else:
                i += 1
                
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
    def forward(self, x):
        # 1. Normalize input if enabled
        if self.use_norm:
            x = self.input_norm(x)
        
        # 2. First projection & activation
        x = self.act(self.linears[0](x))
        compression_tensors = {}
        
        linear_idx = 1
        i = 1
        last_comp_idx = None
        while i < len(self.hidden_layers):
            comp_tensor = self.linears[linear_idx](x)
            linear_idx += 1
            
            if self.use_skips and last_comp_idx is not None:
                proj_key = f"proj_{last_comp_idx}_to_{i}"
                prev_comp_tensor = compression_tensors[last_comp_idx]
                proj = self.projections[proj_key](prev_comp_tensor)
                comp_tensor = comp_tensor + proj
                
            # Apply activation
            comp_tensor = self.act(comp_tensor)
            
            # Apply LayerNorm if enabled
            if self.use_norm:
                comp_tensor = self.norms[f"norm_{i}"](comp_tensor)
            
            compression_tensors[i] = comp_tensor
            x = comp_tensor
            last_comp_idx = i
            
            if i + 1 < len(self.hidden_layers):
                x = self.act(self.linears[linear_idx](x))
                linear_idx += 1
                i += 2
            else:
                i += 1
                
        out = self.output_layer(x)
        return out


class DeepMLP(nn.Module):
    """
    Standard Deep MLP implementation in PyTorch with projected skip connections every 2 layers.
    Supports customizable activation and LayerNorm.
    """
    def __init__(self, input_dim, hidden_layers, output_dim=1, use_skips=True, activation="relu", use_norm=False):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.use_skips = use_skips
        self.activation = activation
        self.use_norm = use_norm
        
        # Optional Input LayerNorm
        if use_norm:
            self.input_norm = nn.LayerNorm(input_dim)
            
        # Selectable Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "elu":
            self.act = nn.ELU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.act = nn.ReLU()
            
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_layers[0]))
        self.projections = nn.ModuleDict()
        
        if use_norm:
            self.norms = nn.ModuleDict()
            
        for idx in range(1, len(hidden_layers)):
            self.linears.append(nn.Linear(hidden_layers[idx-1], hidden_layers[idx]))
            
            # Optional LayerNorm per layer
            if use_norm:
                self.norms[f"norm_{idx}"] = nn.LayerNorm(hidden_layers[idx])
                
            if use_skips and idx % 2 == 1 and idx >= 3:
                proj_key = f"proj_{idx-2}_to_{idx}"
                self.projections[proj_key] = nn.Linear(hidden_layers[idx-2], hidden_layers[idx])
                
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
    def forward(self, x):
        if self.use_norm:
            x = self.input_norm(x)
            
        tensors = []
        x = self.act(self.linears[0](x))
        tensors.append(x)
        
        for idx in range(1, len(self.hidden_layers)):
            current = self.linears[idx](x)
            
            if self.use_skips and idx % 2 == 1 and idx >= 3:
                proj_key = f"proj_{idx-2}_to_{idx}"
                proj = self.projections[proj_key](tensors[idx-2])
                current = current + proj
                
            current = self.act(current)
            if self.use_norm:
                current = self.norms[f"norm_{idx}"](current)
                
            tensors.append(current)
            x = current
            
        out = self.output_layer(x)
        return out


class FeaturePooling(nn.Module):
    """
    Feature-dimension 1D pooling module.
    Downsamples a 1D feature vector of shape [batch_size, in_features]
    to [batch_size, out_features] using Adaptive Max/Avg/Hybrid Pooling.
    """
    def __init__(self, out_features, pool_type="max"):
        super().__init__()
        self.out_features = out_features
        self.pool_type = pool_type
        
        if pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(out_features)
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(out_features)
        elif pool_type == "hybrid":
            self.max_pool = nn.AdaptiveMaxPool1d(out_features)
            self.avg_pool = nn.AdaptiveAvgPool1d(out_features)
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}. Choose 'max', 'avg', or 'hybrid'.")
            
    def forward(self, x):
        # Input shape: [batch_size, in_features]
        # Unsqueeze to add a channel dimension: [batch_size, 1, in_features]
        x_unsqueezed = x.unsqueeze(1)
        
        if self.pool_type == "hybrid":
            max_val = self.max_pool(x_unsqueezed)
            avg_val = self.avg_pool(x_unsqueezed)
            weight = torch.sigmoid(self.alpha)
            out = weight * max_val + (1 - weight) * avg_val
        else:
            out = self.pool(x_unsqueezed)
            
        # Squeeze to return to: [batch_size, out_features]
        return out.squeeze(1)



class BreathMLPPool(nn.Module):
    """
    100% Parameter-Free Compression and Skip connections (BreathMLPPool).
    Uses AdaptivePool1d (Max, Avg, or learnable Hybrid) for ALL compression steps:
      - Main path compressions
      - Skip connection resizing between bottlenecks
      - Final output compression (output_layer)
    Only the input projection and expansion layers contain learnable weights.
    """
    def __init__(self, input_dim, hidden_layers, output_dim=1, use_skips=True, 
                 pool_type="avg", activation="relu", use_norm=False, pool_output=True):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.use_skips = use_skips
        self.pool_type = pool_type
        self.activation = activation
        self.use_norm = use_norm
        
        # Optional Input LayerNorm
        if use_norm:
            self.input_norm = nn.LayerNorm(input_dim)
            
        # Selectable Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "elu":
            self.act = nn.ELU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.act = nn.ReLU()
            
        self.layers = nn.ModuleDict()
        if use_norm:
            self.norms = nn.ModuleDict()
            
        # Input projection
        self.layers["input_proj"] = nn.Linear(input_dim, hidden_layers[0])
        
        i = 1
        last_comp_idx = None
        while i < len(hidden_layers):
            comp_units = hidden_layers[i]
            
            # Main path compression: parameter-free pooling
            self.layers[f"comp_{i}"] = FeaturePooling(comp_units, pool_type=pool_type)
            
            # Optional Bottleneck LayerNorm
            if use_norm:
                self.norms[f"norm_{i}"] = nn.LayerNorm(comp_units)
                
            # Skip connection downsampler: also parameter-free pooling!
            if use_skips and last_comp_idx is not None:
                # We dynamically downsample using FeaturePooling from last bottleneck size to current bottleneck size
                self.layers[f"skip_pool_{last_comp_idx}_to_{i}"] = FeaturePooling(comp_units, pool_type=pool_type)
                
            last_comp_idx = i
            
            if i + 1 < len(hidden_layers):
                exp_units = hidden_layers[i+1]
                # Expansion: standard linear layer
                self.layers[f"exp_{i+1}"] = nn.Linear(comp_units, exp_units)
                i += 2
            else:
                i += 1
                
        # Output layer: use parameter-free pooling if pool_output is True,
        # otherwise use a standard Linear projection layer (essential for mapping features to label/target space).
        if pool_output:
            self.output_layer = FeaturePooling(output_dim, pool_type=pool_type)
        else:
            self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
    def forward(self, x):
        # 1. Normalize input if enabled
        if self.use_norm:
            x = self.input_norm(x)
            
        # 2. First projection & activation
        x = self.act(self.layers["input_proj"](x))
        compression_tensors = {}
        
        i = 1
        last_comp_idx = None
        while i < len(self.hidden_layers):
            # Apply pooling compression
            comp_tensor = self.layers[f"comp_{i}"](x)
            
            # Apply parameter-free skip downsampling
            if self.use_skips and last_comp_idx is not None:
                prev_comp_tensor = compression_tensors[last_comp_idx]
                proj = self.layers[f"skip_pool_{last_comp_idx}_to_{i}"](prev_comp_tensor)
                comp_tensor = comp_tensor + proj
                
            # Apply activation
            comp_tensor = self.act(comp_tensor)
            
            # Apply LayerNorm if enabled
            if self.use_norm:
                comp_tensor = self.norms[f"norm_{i}"](comp_tensor)
                
            compression_tensors[i] = comp_tensor
            x = comp_tensor
            last_comp_idx = i
            
            if i + 1 < len(self.hidden_layers):
                # Apply linear expansion
                x = self.act(self.layers[f"exp_{i+1}"](x))
                i += 2
            else:
                i += 1
                
        out = self.output_layer(x)
        return out





