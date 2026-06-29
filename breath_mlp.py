import torch
import torch.nn as nn

# =====================================================================
# --- GENERAL ARCHITECTURE GENERATORS ---
# =====================================================================
def generate_breath_architecture(start_width, compression_factor=0.25, expansion_factor=2.0, min_width=16):
    """
    Generates a decaying-oscillating (Breath) hidden layer size sequence.
    Example: 512, 0.25, 2.0 -> [512, 128, 256, 32, 64]
    """
    layers_list = [start_width]
    current = start_width
    
    while True:
        # 1. Compress
        compressed = int(current * compression_factor)
        if compressed < min_width:
            break
        layers_list.append(compressed)
        
        # 2. Expand with decay constraint
        expanded = int(compressed * expansion_factor)
        if len(layers_list) >= 3:
            previous_expanded = layers_list[-3]
            # Ensure the new expansion is smaller than the previous one to maintain decay
            if expanded >= previous_expanded:
                expanded = int(previous_expanded * 0.5)
        
        if expanded < min_width:
            break
        layers_list.append(expanded)
        current = compressed
        
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
    def __init__(self, input_dim, hidden_layers, output_dim=1, use_skips=True):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.use_skips = use_skips
        
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_layers[0]))
        self.projections = nn.ModuleDict()
        
        i = 1
        last_comp_idx = None
        while i < len(hidden_layers):
            comp_units = hidden_layers[i]
            prev_units = hidden_layers[i-1]
            self.linears.append(nn.Linear(prev_units, comp_units))
            
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
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linears[0](x))
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
                
            comp_tensor = self.relu(comp_tensor)
            compression_tensors[i] = comp_tensor
            x = comp_tensor
            
            if i + 1 < len(self.hidden_layers):
                x = self.relu(self.linears[linear_idx](x))
                linear_idx += 1
                i += 2
            else:
                i += 1
                
        out = self.output_layer(x)
        return out

class DeepMLP(nn.Module):
    """
    Standard Deep MLP implementation in PyTorch with projected skip connections every 2 layers.
    """
    def __init__(self, input_dim, hidden_layers, output_dim=1, use_skips=True):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.use_skips = use_skips
        
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_layers[0]))
        self.projections = nn.ModuleDict()
        
        for idx in range(1, len(hidden_layers)):
            self.linears.append(nn.Linear(hidden_layers[idx-1], hidden_layers[idx]))
            
            if use_skips and idx % 2 == 1 and idx >= 3:
                proj_key = f"proj_{idx-2}_to_{idx}"
                self.projections[proj_key] = nn.Linear(hidden_layers[idx-2], hidden_layers[idx])
                
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        tensors = []
        x = self.relu(self.linears[0](x))
        tensors.append(x)
        
        for idx in range(1, len(self.hidden_layers)):
            current = self.linears[idx](x)
            
            if self.use_skips and idx % 2 == 1 and idx >= 3:
                proj_key = f"proj_{idx-2}_to_{idx}"
                proj = self.projections[proj_key](tensors[idx-2])
                current = current + proj
                
            current = self.relu(current)
            tensors.append(current)
            x = current
            
        out = self.output_layer(x)
        return out
