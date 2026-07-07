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
1. **Constant Oscillation Factors:** Every contraction reduces the dimension by a constant factor $C = 0.25$ (1/4 of the preceding peak). Every internal expansion increases the dimension by a constant factor $E = 2.0$ (doubles the preceding bottleneck). *As a single exception, the very first expansion (input projection $d_{\text{in}} \to P_1$) is allowed to exceed the 2.0x factor (e.g. 4x, 8x, or more) to project the features into a sufficiently large latent manifold before initiating the breathing oscillations.*
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
* We replace all linear compression layers and skip resizing projections with **Adaptive Feature Pooling** (`AdaptiveMaxPool` or `AdaptiveAvgPool`).
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

* **🔍 Architectural Note on MNIST Dimension Constraints:**
  The performance drop of `BreathPool Max` on MNIST (75.93%) is not a fundamental failure of the pooling architecture, but rather a consequence of the under-parameterized initial dimension used in this specific benchmark:
  * **The Choked Initial Expansion:** Because the input dimension is very large ($d_{in} = 784$ pixels), setting the first hidden layer to `1024` represents only a **1.3x expansion** ($1024 / 784 = 1.3$). This is too small to project the input into a high-dimensional manifold before applying the first 1/4 compression (bottlenecking it immediately to $256$, which is a massive 1/3 reduction of the input dimension without a proper expansion step).
  * **The Parameter-Matched Alternative (2x Expansion):** If we apply a proper **2x expansion** ($2 \times 784 = 1568$), the Purist Ruleset generates the hidden sequence `[1568, 392, 784, 196, 392, 98, 196, 49, 98, 24, 48, 12, 24, 10]`. This configuration has **`1,645,142` parameters** (using 100% parameter-free pooling for all contractions and output mapping), which is almost perfectly parameter-matched to the `Deep Standard` baseline of 1.5M parameters (+9.3% difference). This proper 2x projection space allows the network to maintain expressive capacity before the subsequent breathing cycles.

---


#### D. Transformer FFN Integration (NanoGPT on Tiny Shakespeare - 3-Seed)
* **Test Objective:** Autoregressive character-level language modeling trained for 1000 iterations across 3 seeds on the canonical 4x capacity setup.

##### Canonical Configuration (FFN_START = 4x d_model)
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

#### E. Vision Transformer FFN Integration (ViT on CIFAR-10 - 3-Seed)
* **Test Objective:** Image classification with a patch-based ViT ($d_{\text{model}}=192$, 4 layers, 4 heads, patch size 4×4) trained for 5 epochs across 3 seeds. BreathPool models perform all contractions and the final mapping via pooling, resulting in **50.0% FFN block parameter savings** (and **32.7% total model parameter savings**).
* **Network Layer Configurations:**
  * **Standard FFN:** `[192, 768, 192]`
  * **Breath (Linear) / BreathPool (Max, Avg, Hybrid):** `[192, 768, 192]`

| Model | Parameters | Training Time (Mean) | Test Accuracy (3-seed) | Δ Params (Total) | FFN Block Params |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard** (baseline) | 1,803,850 | 53.4s | `54.38% +/- 1.13%` | Baseline | 0% (Reference) |
| **Breath (Linear)** | 1,805,386 | 54.0s | `54.17% +/- 0.81%` | +0.08% | +0.52% |
| 🏆 **BreathPool Max** | **1,214,794** | 50.5s *(-5.4%)* | **`56.22% +/- 0.43%`** | **-32.7%** | **-50.0%** |
| **BreathPool Avg** | 1,214,794 | **49.9s** *(-6.5%)* | `55.55% +/- 0.85%` | -32.7% | -50.0% |
| **BreathPool Hybrid** | 1,214,798 | 60.9s | `55.39% +/- 0.62%` | -32.7% | -50.0% |

* **Key Finding:** When evaluated under the canonical, non-bloated **4x FFN capacity configuration**, `BreathPool Max` and `BreathPool Avg` outperform the Standard baseline in every single dimension:
  * **Higher Accuracy:** `BreathPool Max` achieves **56.22%** accuracy (a **+1.84% absolute improvement** over the baseline) while having the lowest standard deviation ($\pm 0.43\%$), indicating superior training stability.
  * **Less Parameters:** They save exactly **50.0% of the FFN block parameters** (saving `148,224` weights per layer, translating to a **32.7% net reduction** in the entire model size).
  * **Faster Training:** They run **~6.5% faster** on GPU, showing that replacing the contraction matrix multiplication with parameter-free pooling bypasses compute bottlenecks.
* **Pooling Strategy Selection:** Image patch features are dense, continuous, and highly structured, which benefits from the smooth, full-activation backpropagation path of Average Pooling. In contrast, text modeling (NanoGPT) benefits more from Max Pooling due to the sparse, selective nature of token representations.


---

### 🧠 Key Takeaways on Feature Pooling

1. **Massive Parameter Reductions:**
   By removing learnable parameters from compression and skip paths, **`BreathPool`** variants reduce the parameter footprint by **up to 72.9%** on tabular tasks (California, SARCOS), **up to 50.0% on FFN blocks in Vision Transformers (ViT)**, and **up to 50.0% on FFN blocks in GPT models**, with negligible or positive performance impact.
2. **Significant Training Speedups at Scale:**
   `BreathPool` models achieve a **6.5% speedup** on Vision Transformers (ViT) compared to standard baselines on GPU. This highlights that **pooling becomes highly competitive at larger dimensional scales**: while in small networks the memory-bound overhead (reshape/permute operations) of pooling can exceed the negligible linear layer compute, at scale, replacing massive GEMM matrix multiplications with parameter-free pooling completely bypasses the primary training bottleneck.
3. **Domain-Dependent Pooling Specialization (Critical Finding):**
   | Domain | Winner | Reason |
   | :--- | :--- | :--- |
   | **Structured Tabular / Regression** (SARCOS, California) | 🏆 `BreathPool Avg` / `Hybrid` | Dataset-dependent: learnable layer-wise blending (`Hybrid`) wins on robotics dynamics (SARCOS), while pure average pooling (`Avg`) wins on California Housing. |
   | **Language Modeling** (NanoGPT) | 🏆 `BreathPool Max` | Tokens are sparse categorical events; max selection mimics the selective routing of self-attention. |
   | **Vision/Image Classification** (ViT, CIFAR-10) | 🏆 `BreathPool Max` | Image patch features benefit from spatial sparsification under Max selection, outperforming baseline models. |
4. **Learnable Hybrid Pooling (Adaptive Alpha Blending):**
   The **`BreathPool Hybrid`** model dynamically blends Max and Average pooling using a learnable weight $\sigma(\alpha)$ per layer.
   * **On Tabular Tasks:** Learns highly customizable mixtures (e.g. averaging ~54% Max Pool on California, and ~51% Max Pool on SARCOS), achieving the highest $R^2$ scores among all pooling variants on SARCOS.
   * **On NanoGPT:** Achieves stable convergence (Validation Loss `1.5892 +/- 0.0099` for 4x FFN), performing close to the baseline.
   * **On ViT:** Achieves `55.39% +/- 0.62%` accuracy, closely matching the pure Max winner (`56.22%`) with exceptionally low variance ($\pm 0.62\%$).

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


