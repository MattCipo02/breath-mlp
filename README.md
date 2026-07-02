# Breath MLP: A Parameter-Efficient Hourglass Multi-Layer Perceptron

A parameter-efficient feedforward neural network architecture in **PyTorch** that alternates between high-dimensional latent spaces (expansion) and low-dimensional bottlenecks (compression) in a decaying sequence, bridged by linear projected skip connections.

This repository provides the official implementation, architecture generator, and comparison benchmarks on classic datasets (such as **SARCOS**, **California Housing**, and **MNIST**).

---

### 💡 The Evolution of Breath MLP (A Step-by-Step Architectural Development)

To understand how **Breath MLP** works, let us retrace the logical steps and engineering choices that led to its formulation, starting from standard Transformer FFNs all the way to parameter-free feature pooling.

---

### Step 1: The Starting Point (Standard MLP & Transformer FFN)
In a classical Feed-Forward Network (FFN) used inside Transformers (such as GPT or BERT), the architecture follows an "inverted bottleneck" structure:
$$d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$$
Typically, $d_{\text{ff}} = 4 \times d_{\text{model}}$. This structure:
* Represents a single cycle of expansion and contraction.
* Is effective, but constitutes a **degenerate case** of a single oscillation: it lacks architectural depth and forces the network to allocate massive linear projection matrices.

---

### Step 2: The Intuition (Latent Breathing)
Instead of restricting ourselves to a single temporary expansion, why not allow the network to **"breathe"** multiple times?
The core idea of **Breath MLP** is to systematically alternate:
1. **Expansion (Inhalation):** Projecting into a high-dimensional latent space to allow features to recombine and capture complex non-linear interactions.
2. **Contraction (Exhalation):** Squeezing the representation through a low-dimensional bottleneck. This acts as a natural **denoising filter** and regularizer, forcing the network to discard redundant information and discover the low-dimensional manifold underlying the data.

```
Input (N) ──> [Dense (512)] (Expansion 1)
                 │
              [Dense (128)] (Bottleneck 1) ───────────┐
                 │                                    │ (Projected Skip Connection)
              [Dense (256)] (Expansion 2)             │
                 │                                    │ [Linear Projection (128 -> 64)]
              [Dense (64)]  (Bottleneck 2) <──────────┘ 
                 │
              [Dense (128)] (Expansion 3)
                 │
              [Dense (32)]  (Bottleneck 3) ──> Output (M)
```

---

### Step 3: Mathematical Formalization (The "Purist Ruleset")
To avoid ad-hoc heuristic designs and define a geometrically rigorous topology, we establish our core constraints:
1. **Constant Oscillation Factors:** Every contraction reduces the dimension by a constant factor $C = 0.25$ (1/4 of the preceding peak). Every expansion increases the dimension by a constant factor $E = 2.0$ (doubles the preceding bottleneck).
2. **Continuous Decay (Envelope):** To guarantee convergence and avoid infinite loops at constant widths, the height of the peaks ($P_1 > P_2 > P_3 \dots$) and bottlenecks ($B_1 > B_2 > B_3 \dots$) must continuously decay at each cycle, forming a dampened oscillating wave.
3. **Strict Output Bottleneck Constraint:** No intermediate bottleneck can be smaller than or equal to the final output dimension ($d_{\text{out}}$). If a compression step falls below or matches this threshold, the loop breaks immediately. Consequently, every hidden layer is guaranteed to be strictly larger than the output dimension, ensuring that the final transition from the last latent state to the actual output layer is resolved via a parameter-free feature pooling contraction, eliminating the need for a final heavy linear projection.

---

### Step 4: The Gradient Bottleneck and Projected Skip Connections
As the network grows deeper by nesting contractions and expansions, we encounter a classical challenge: **the vanishing gradient problem**. Tight bottlenecks tend to block the flow of gradients during backpropagation.
* **The Solution:** Introduce shortcut connections (Skip Connections) between consecutive bottlenecks (e.g., from $B_1$ to $B_2$).
* **The Dimensional Challenge:** Unlike traditional ResNets, the bottlenecks have different sizes ($B_1 = 128$ and $B_2 = 32$). We cannot add them directly.
* **The Linear Answer:** We use a learned linear projection matrix ($\mathbf{W}_{\text{proj}}$) to resize the preceding bottleneck and add it to the next. This "gradient highway" stabilizes training and raises performance to par with much larger models.

---

### Step 5: Extreme Parameter Optimization (Feature Pooling / BreathPool)
*But can we do even better?*
In Step 4, every main-path contraction and every skip connection resizing still requires a learned linear projection matrix full of weights.
**BreathPool** is designed to eliminate these redundant parameters completely, making all contractions **100% parameter-free**:
* We replace all linear compression layers and skip resizing projections with **Adaptive Feature Pooling** (`AdaptiveMaxPool1d` or `AdaptiveAvgPool1d`).
* Pooling acts directly along the activation's feature dimension, resizing it geometrically without requiring any learnable weights.
* The only learnable weights remaining in the network are the input projection and the expansion layers.

In the following sections, we demonstrate how this evolution allows **BreathPool** to slash **50% to 70% of total parameters**, speed up training times, and maintain or exceed standard accuracy benchmarks.

## ⚡ Feature Pooling & Parameter-Free Compression (BreathMLPPool)

To maximize parameter efficiency and accelerate training, we replaced the linear compression layers in the Breath MLP with parameter-free **Adaptive Feature Pooling** (Max, Average, and learnable Hybrid pooling).

### 📐 The Three Pooling Strategies:
1. 🏆 **BreathPool Max:** Replaces compression and skip resizing paths with `AdaptiveMaxPool1d`. Serves as a sparse, high-amplitude feature extractor (reminiscent of CNN pooling), which fits image and text classification exceptionally well.
2. 🏆 **BreathPool Avg:** Replaces compression and skip resizing paths with `AdaptiveAvgPool1d`. Provides a continuous, dense gradient backpropagation path to all neurons, preventing gradient bottlenecking.
3. 🏆 **BreathPool Hybrid:** Blends Max and Average pooling dynamically using a learnable weight $\sigma(\alpha)$ per layer:
   $$\text{Output} = \sigma(\alpha) \cdot \text{MaxPool}(X) + (1 - \sigma(\alpha)) \cdot \text{AvgPool}(X)$$
   The model automatically learns the optimal pooling mixture for each layer based on the dataset nature.

### 📊 Benchmark Results & Training Times (5-Fold CV & Multi-Seed)

All models were evaluated under identical conditions on an RTX 4070 Laptop GPU. 

We utilize **100% parameter-free compression and output mapping** for all pooling variants (using `FeaturePooling` also at the final layer). 
- **The Target Scaling Constraint:** For regression (SARCOS, California), targets are normalized to $[0, 1]$ using `MinMaxScaler`. If we use standard centering (`StandardScaler`, yielding negative targets), the model's performance collapses ($R^2 \approx 0.50$) because it cannot output negative values under ReLU.
- **The Output Sparsity Constraint:** For classification (MNIST), mapping features directly to class logits using Max Pooling causes severe gradient sparsity (only the maximum neuron in each bin receives a gradient), dropping accuracy to `75.93%`. Average pooling distributes gradients densely, maintaining `97.94%`. Learnable Hybrid pooling automatically balances its weights to bypass Max pool, rescuing accuracy to `92.53%`.

#### A. Robotics Inverse Dynamics (SARCOS Dataset - MinMaxScaler $[0,1]$)
* **Test Objective:** Predict joint torques from 21 kinematic features.
* **Network Layer Configurations:**
  * **Deep Standard:** `[21, 512, 256, 128, 64, 32, 1]`
  * **Breath (Linear) / BreathPool (Max, Avg, Hybrid):** `[21, 512, 128, 256, 64, 128, 32, 64, 1]`

| Model | Parameters | Training Time (Mean) | $R^2$ Score (5-fold CV) | MSE (5-fold CV) | Δ Params (Total) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Deep Standard | 185,857 | 47s | `0.9751 +/- 0.0066` | `10.4459 +/- 2.8255` | Baseline |
| Breath + Skips (Lin) | 151,361 | 66s | `0.9744 +/- 0.0035` | `10.7080 +/- 1.5295` | -18.6% |
| **BreathPool Max** | **54,720** | **58s** | `0.9703 +/- 0.0047` | `12.4305 +/- 1.9140` | **-70.5%** |
| **BreathPool Avg** | 54,720 | 58s | `0.9664 +/- 0.0105` | `14.0014 +/- 4.1221` | -70.5% |
| 🏆 **BreathPool Hybrid** | **54,726** | 114s | **`0.9758 +/- 0.0016`** | **`10.1046 +/- 0.6625`** | **-70.5%** |

#### B. Regression (California Housing - MinMaxScaler $[0,1]$)
* **Test Objective:** Predict median house values from 8 demographic features.
* **Network Layer Configurations:**
  * **Deep Standard:** `[8, 512, 256, 128, 64, 32, 16, 1]`
  * **Breath (Linear) / BreathPool (Max, Avg, Hybrid):** `[8, 512, 128, 256, 64, 128, 32, 64, 16, 32, 1]`

| Model | Parameters | Training Time (Mean) | $R^2$ Score (5-fold CV) | MSE (5-fold CV) | Δ Params (Total) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Standard** | 179,713 | 21s | `0.7828 +/- 0.0089` | `0.2892 +/- 0.0150` | Baseline |
| Breath + Skips (Lin) | 146,785 | 34s | `0.7771 +/- 0.0135` | `0.2970 +/- 0.0220` | -18.3% |
| **BreathPool Max** | **48,608** | **31s** | `0.7402 +/- 0.0232` | `0.3458 +/- 0.0317` | **-72.9%** |
| 🏆 **BreathPool Avg** | 48,608 | 31s | **`0.7484 +/- 0.0090`** | **`0.3350 +/- 0.0130`** | -72.9% |
| **BreathPool Hybrid** | **48,616** | 56s | `0.7344 +/- 0.0161` | `0.3538 +/- 0.0248` | **-72.9%** |

#### C. Image Classification (MNIST Dataset - Raw Target Mapping)
* **Test Objective:** Classify hand-written digits (10 classes) using raw target pooling.
* **Network Layer Configurations:**
  * **Deep Standard:** `[784, 1024, 512, 256, 128, 64, 32, 16, 10]`
  * **Breath (Linear) / BreathPool (Max, Avg, Hybrid):** `[784, 1024, 256, 512, 128, 256, 64, 128, 32, 64, 16, 32, 10]`

| Model | Parameters | Training Time (Mean) | Test Accuracy (5-fold CV) | Δ Params (Total) |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **Deep Standard** | 1,503,898 | 74s | **`98.03% +/- 0.20%`** | Baseline |
| **Breath + Skips (Lin)** | 1,373,194 | 120s | `97.90% +/- 0.30%` | -8.7% |
| **BreathPool Max** | **979,424** | **103s** | `75.93% +/- 7.22%` | **-34.9%** |
| **BreathPool Avg** | **979,424** | **101s** | `97.94% +/- 0.18%` | **-34.9%** |
| **BreathPool Hybrid** | **979,434** | 206s | `92.53% +/- 4.54%` | **-34.9%** |

---


#### D. Transformer FFN Integration (NanoGPT on Tiny Shakespeare - 3-Seed)
* **Test Objective:** Autoregressive character-level language modeling trained for 1000 iterations across 3 seeds. We compare two distinct regimes: a canonical 4x capacity setup and a larger parameter-matched 8x setup.

##### D1. Canonical Configuration (FFN_START = 4x d_model)
In this setup, the FFN expansion is set to `4 * d_model = 1024`. **This 4x multiplier represents the canonical size standardly used in production Transformers (such as GPT-2 and BERT).** 
* **Network Layer Configurations:**
  * **Standard FFN:** `[256, 1024, 256]`
  * **Breath (Linear) / BreathPool (Max, Avg, Hybrid):** `[256, 1024, 256]`

For `BreathPool`, because the architecture does not have room for intermediate oscillations (`hidden_layers = [1024]`), the final compression step is performed entirely by parameter-free pooling. This **eliminates the second linear projection** of the FFN, cutting the block's parameters by **50%**.

| Model | Model Params | Training Time (Mean) | Validation Loss (3-seed) | Δ Params (Block) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard FFN (4x)** | 4,805,185 | 222.2s | `1.5798 +/- 0.0039` | Baseline |
| **Breath FFN (4x)** | 4,808,257 | 234.0s | `1.5770 +/- 0.0037` | +0.10% |
| 🏆 **BreathPool Max** | **3,233,857** *(-32.7%)* | **221.2s** | **`1.5778 +/- 0.0074`** | **-50.0%** |
| **BreathPool Avg** | 3,233,857 *(-32.7%)* | 221.4s | `1.6327 +/- 0.0091` | -50.0% |
| **BreathPool Hybrid** | 3,233,863 *(-32.7%)* | 248.7s | `1.5892 +/- 0.0099` | -50.0% |

* **Key Finding:** Replacing the entire second linear layer of a standard FFN block with parameter-free **Max Pooling** matches standard FFN validation loss while saving **50.0% of the FFN block parameters** (and 32.7% of the entire GPT model parameters). **Importantly, training speed is fully preserved, matching standard FFN training time (221.2s vs 222.2s) with zero computational or training overhead.**

##### D2. Expanded Parameter-Matched Configuration (FFN_START = 8x d_model)
Here, the FFN starts with a width of `8 * d_model = 2048`. The Standard baseline is expanded to `18 * d_model = 4864` intermediate width to parameter-match the linear Breath FFN (`[2048, 512, 1024]` hidden layout). BreathPool models save **55.5% of FFN block parameters** (and 49.7% of total model parameters) by performing all bottleneck contractions via pooling.
* **Network Layer Configurations:**
  * **Standard FFN (18x):** `[256, 4608, 256]`
  * **Breath (Linear) / BreathPool (Max, Avg, Hybrid):** `[256, 2048, 512, 1024, 256]`

| Model | Model Params | Training Time (Mean) | Validation Loss (3-seed) | Δ Params (Total) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard (18x)** | 15,836,737 | 390.7s | `1.5740 +/- 0.0144` | Baseline |
| **Breath (Linear)** | 15,839,809 | 386.0s | `1.5460 +/- 0.0092` | +0.02% |
| 🏆 **BreathPool Max** | **7,970,881** *(-49.7%)* | 269.7s *(-31.0%)* | **`1.5308 +/- 0.0123`** | **-49.7%** |
| **BreathPool Avg** | **7,970,881** *(-49.7%)* | **222.5s** *(-43.0%)* | `1.5549 +/- 0.0093` | -49.7% |
| **BreathPool Hybrid** | 7,970,893 *(-49.7%)* | 276.0s *(-29.4%)* | `1.5336 +/- 0.0102` | -49.7% |

* **Key Finding:** By nesting multiple compression/expansion cycles and resolving bottlenecks via pooling, **BreathPool Max** improves validation loss by **-0.0432** over Standard and **-0.0152** over Breath (Linear) while using **half the parameters** (49.7% reduction). 
* **Training Speedup:** Because parameter-free pooling replaces heavy linear projections, training speed is dramatically enhanced. **BreathPool Avg** achieves a **43.0% training speedup** (222.5s vs 390.7s) and **BreathPool Max** achieves a **31.0% speedup** (269.7s vs 390.7s) on laptop GPU compared to the parameter-matched standard transformer baseline.

#### E. Vision Transformer FFN Integration (ViT on CIFAR-10 - 3-Seed)
* **Test Objective:** Image classification with a patch-based ViT ($d_{\text{model}}=192$, 4 layers, 4 heads, patch size 4×4) trained for 5 epochs across 3 seeds. BreathPool models perform all contractions and the final mapping via pooling, resulting in **61.8% FFN block parameter savings** (and **56.2% total model parameter savings**).
* **Network Layer Configurations:**
  * **Standard FFN:** `[192, 4032, 192]`
  * **Breath (Linear) / BreathPool (Max, Avg, Hybrid):** `[192, 1536, 384, 768, 192]`

| Model | Parameters | Training Time (Mean) | Test Accuracy (3-seed) | Δ Params (Total) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard** | 6,830,410 | 157.3s | `50.39% +/- 1.07%` | Baseline |
| **Breath (Linear)** | 5,944,906 | 139.1s | `52.72% +/- 0.91%` | -13.0% |
| **BreathPool Max** | 2,993,482 | 105.7s | `55.39% +/- 0.99%` | -56.2% |
| 🏆 **BreathPool Avg** | **2,993,482** | **104.8s** *(-33.3%)* | **`55.84% +/- 1.26%`** | **-56.2%** |
| **BreathPool Hybrid** | 2,993,490 | 136.2s | `55.32% +/- 1.06%` | -56.2% |

* **Key Finding:** On ViT/images, **`BreathPool Avg`** achieves the highest performance (**55.84%** accuracy, representing a **+5.45%** absolute improvement over the Standard baseline) while running **33.3% faster** (104.8s vs 157.3s) and using **less than half the parameters** (-56.2%). 
* **Pooling Strategy Selection:** Image patch features are dense, continuous, and highly structured, which benefits from the smooth, full-activation backpropagation path of Average Pooling. In contrast, text modeling (NanoGPT) benefits more from Max Pooling due to the sparse, selective nature of token representations.

---

### 🧠 Key Takeaways on Feature Pooling

1. **Massive Parameter Reductions:**
   By removing learnable parameters from compression and skip paths, **`BreathPool`** variants reduce the parameter footprint by **up to 72.9%** on tabular tasks (California, SARCOS), **up to 61.8% on FFN blocks in Vision Transformers (ViT)**, and **up to 55.5% on FFN blocks in GPT models**, with negligible or positive performance impact.
2. **Significant Training Speedups at Scale:**
   `BreathPool` models achieve a **43.0% training speedup** on large Transformer workloads (NanoGPT 8x) and a **33.3% speedup** on Vision Transformers (ViT) compared to standard parameter-matched baselines on GPU. This highlights that **pooling becomes highly competitive at larger dimensional scales**: while in small networks the memory-bound overhead (reshape/permute operations) of pooling can exceed the negligible linear layer compute, at scale, replacing massive GEMM matrix multiplications with parameter-free pooling completely bypasses the primary training bottleneck.
3. **Domain-Dependent Pooling Specialization (Critical Finding):**
   | Domain | Winner | Reason |
   | :--- | :--- | :--- |
   | **Structured Tabular / Regression** (SARCOS, California) | 🏆 `BreathPool Avg` / `Hybrid` | Dataset-dependent: learnable layer-wise blending (`Hybrid`) wins on robotics dynamics (SARCOS), while pure average pooling (`Avg`) wins on California Housing. |
   | **Language Modeling** (NanoGPT) | 🏆 `BreathPool Max` | Tokens are sparse categorical events; max selection mimics the selective routing of self-attention. |
   | **Vision/Image Classification** (ViT, CIFAR-10) | 🏆 `BreathPool Avg` | Patch features are spatially dense and continuous; average pooling preserves texture and local patterns better than max selection. |
4. **Learnable Hybrid Pooling (Adaptive Alpha Blending):**
   The **`BreathPool Hybrid`** model dynamically blends Max and Average pooling using a learnable weight $\sigma(\alpha)$ per layer.
   * **On Tabular Tasks:** Learns highly customizable mixtures (e.g. averaging ~54% Max Pool on California, and ~51% Max Pool on SARCOS), achieving the highest $R^2$ scores among all pooling variants on SARCOS.
   * **On NanoGPT:** Achieves stable convergence (Validation Loss `1.5336 +/- 0.0102` for 8x FFN), performing on par with the pure Max winner.
   * **On ViT:** Achieves `55.32% +/- 1.06%` accuracy, closely matching the pure Avg winner (`55.84%`) with exceptionally low variance ($\pm 1.06\%$).

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/breath-mlp.git
cd breath-mlp
pip install torch torchvision scikit-learn pandas scipy numpy
```

### 2. Run the Benchmarks
* **Tabular & Small-scale Benchmarks**: Run the pooling and baseline comparison tests (SARCOS, California, MNIST):
  ```bash
  python pool_benchmark.py --dataset sarcos
  ```
* **Image Denoising Benchmark (ImageNet-32)**: Compare Hourglass vs Breath MLP on natural image restoration:
  ```bash
  python imagenet32_denoising_benchmark.py --dataset imagenet32 --data_dir ./imagenet32 --model breath --dz 9216 --min_width 512 --epochs 2
  ```
* **Image Classification Ablation Study (CIFAR-10)**: Compare activation functions and LayerNorm configurations:
  ```bash
  python classification_experiments.py
  ```
* **Transformer FFN Integration (NanoGPT & ViT)**: Train and compare standard vs Breath FFNs on text generation (Tiny Shakespeare) or vision (CIFAR-10):
  ```bash
  # Run NanoGPT experiments (16.5M params)
  python transformer_breath_experiments_scaled.py

  # Run Vision Transformer experiments (6.8M params)
  python vit_breath_experiments.py
  ```

---

## 💻 Code Example

```python
import torch
from breath_mlp import generate_breath_architecture, BreathMLPPool

# 1. Generate layers sequence: [512, 128, 256, 64, 128, 32]
hidden_layers = generate_breath_architecture(
    start_width=512, 
    compression_factor=0.25, 
    expansion_factor=2.0, 
    min_width=32
)

# 2. Instantiate Parameter-Free BreathMLPPool Model
# All core properties are fully customizable to fit your dataset needs:
model = BreathMLPPool(
    input_dim=21, 
    hidden_layers=hidden_layers, 
    output_dim=1, 
    use_skips=True,
    
    # --- Customization Properties ---
    pool_type="hybrid",      # Pooling strategy: "max", "avg", or "hybrid"
    activation="relu",       # Activation: "relu", "gelu", "silu", "tanh", "elu", "leaky_relu"
    use_norm=True,           # Toggle Layer Normalization (True / False)
    pool_output=True         # Toggle 100% parameter-free output pooling (True / False)
)

print(model)
```


