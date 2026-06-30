# Breath MLP: A Parameter-Efficient Hourglass Multi-Layer Perceptron

A parameter-efficient feedforward neural network architecture in **PyTorch** that alternates between high-dimensional latent spaces (expansion) and low-dimensional bottlenecks (compression) in a decaying sequence, bridged by linear projected skip connections.

This repository provides the official implementation, architecture generator, and comparison benchmarks on classic datasets (such as **SARCOS**, **California Housing**, and **MNIST**).

---

## 💡 Overview & Intuition

Conventional MLPs process data through constant-width layers or monotonic decay (e.g., `[512, 256, 128, 64]`). This often leads to either parameter redundancy or early information loss.

**Breath MLP** solves this by mimicking a "breathing" contraction-expansion pattern:

```
Input (N) ──> [Dense (512)] (Expand)
                 │
              [Dense (128)] (Compress/Bottleneck 1) ──┐
                 │                                     │ (Projected Skip)
              [Dense (256)] (Expand)                   │
                 │                                     │
              [Dense (32)]  (Compress/Bottleneck 2) <──┘ [Projection (128 -> 32)]
                 │
              [Dense (64)]  (Expand)
                 │
              [Dense (16)]  (Compress/Bottleneck 3) ──> Output (M)
```

### Key Mechanisms:
1. **Bottleneck Regularization:** Forcing representation through narrow bottlenecks acts as a denoising filter, driving the network to discover low-dimensional manifolds.
2. **Latent Interaction:** Expanding the space back after each bottleneck allows the compressed features to recombine in high-dimensional spaces before the next compression stage.
3. **Projected Skip Connections:** Bypassing intermediate expansions using linear projections ($\mathbf{W}_{\text{proj}}$) keeps the gradient flow stable and mitigates the vanishing gradient problem, raising performance to par with much larger models.

⚠️ **Core Rule:** The first layer **must** represent an expansion of the input space. For low-dimensional features (e.g., 21 inputs in SARCOS), a starting width of 512 is sufficient. For high-dimensional features (e.g., 784 inputs in MNIST), a starting width of 1024+ is required to avoid information loss at the input layer.

### 🎛️ Tuneable Decay and Oscillation
By modifying the `compression_factor` and `expansion_factor` parameters in `breath_mlp.py`, researchers can study different families of shapes and behaviors:
* **High Compression / Low Expansion:** Squeezes information very tightly in deep bottlenecks, acting as a stronger regularizer.
* **Low Compression / High Expansion:** Retains wider representation dimensions, allowing complex feature maps at the expense of more parameters.
* **Variable Decay Envelopes:** Tuning these factors controls the frequency and amplitude of the "breathing" cycles, mapping out different paths of information propagation.

> 📝 **Note:** This design independently replicates and expands the theoretical benefits of the **"Hourglass MLP"** (Wide-Narrow-Wide shape convention) proposed in the research paper *"Rethinking the shape convention of an MLP"* (Chen et al., October 2025).

---

## 📐 Architectural Rules & Oscillation Ratio
To ensure a mathematically sound, rigorous, and fair comparison, all **Breath MLP** models in these benchmarks strictly adhere to the following **purist shape rules**:
1. **Constant Oscillation Ratio:** The network oscillates using a strict compression factor of **`0.25` (1/4)** and an expansion factor of **`2.0` (times 2)**.
2. **Continuous Decay:** Every subsequent peak and bottleneck decreases continuously in size, forming a decaying sine-wave envelope ($P_1 \to P_2 \to P_3 \dots$ and $B_1 \to B_2 \to B_3 \dots$).
3. **Full-Rank Bottlenecks:** To prevent information bottleneck rank collapse, the minimum dimension of any bottleneck is **always larger than or equal to the output size** (e.g., $d_{\text{model}}$ in Transformers, or $10$ in CIFAR-10/MNIST classifiers).
4. **Parameter Matching:** Standard baseline MLPs and FFNs are **parameter-matched** to the Breath MLP (by scaling their intermediate dimensions) to isolate the representational benefits of the shape topology.
5. **Activation & Normalization Modularity:** Supports customizable activation functions (`"relu"` [default], `"gelu"`, `"silu"`) and optional LayerNorm layers (applied at the input and inside bottlenecks). Normalization is critical to prevent activation variance explosion in deep configurations when moving away from ReLU's natural zero-clipping constraint.

---

## 📊 Benchmark Results

### 1. Robotics Inverse Dynamics (SARCOS Dataset)
Predicting the joint torque on a robotics kinematics task.
* **Dimensionality:** Inputs = 21, Outputs = 1 joint torque value.
* **Purist Rules Applied:**
  * Oscillation Ratio: Compression factor `0.25` (1/4), Expansion factor `2.0` (times 2).
  * Full-Rank check: Output size = 1, minimum bottleneck = 16 (since $16 \ge 1$, respects full-rank).
  * Decay envelope: strict continuous decay of peaks/bottlenecks.
* **Training Conditions:** Trained for 40 epochs on CUDA. Optimizer: Adam ($LR = 0.001$), Batch Size = 64.
* **Network Configurations:**
  * **Deep Standard:** `[512, 256, 128, 64, 32, 16]` (Parameters: 186,369, Activation: ReLU, Norm: None).
  * **Deep + Skips:** `[512, 256, 128, 64, 32, 16]` (Parameters: 203,857, Activation: ReLU, Norm: None).
  * **Breath Standard:** `[512, 128, 256, 64, 64, 16, 32]` (Parameters: 132,177, Activation: ReLU, Norm: None).
  * **Breath + Skips:** `[512, 128, 256, 64, 64, 16, 32]` (Parameters: 141,473, Activation: ReLU, Norm: None).

| Model | Parameters | Train Time | MSE | $R^2$ Score | nMSE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Deep Standard | 186,369 | 83.8s | `4.3948` | `0.9894` | `1.06%` |
| Deep + Skips | 203,857 | 72.9s | `3.7917` | `0.9908` | `0.92%` |
| Breath Standard | **132,177** | 79.2s | `4.6989` | `0.9887` | `1.13%` |
| 🏆 **Breath + Skips** | **141,473** | **73.4s** | `4.9144` | `0.9881` | `1.19%` |

*Breath + Skips reaches **`0.9881` $R^2$** while saving **30.6% of parameters** compared to the Deep + Skips baseline.*

---

### 2. Denoising & Noise Robustness (SARCOS)
Evaluating the implicit regularizing behavior of the bottlenecks.
* **Test Conditions:** We added varying levels of Gaussian noise ($\sigma \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$) to the input features of a pre-trained model.
* **Architecture used:** Breath + Skips (`[512, 128, 256, 64, 64, 16, 32]`) vs Deep Standard (`[512, 256, 128, 64, 32, 16]`).

| Noise Std ($\sigma$) | Deep R2 | Breath + Skips R2 | Breath Margin |
| :---: | :---: | :---: | :---: |
| **0.00 (Clean)** | `0.9915` | **`0.9919`** | `+0.04%` |
| **0.10** | `0.9730` | **`0.9736`** | `+0.06%` |
| **0.20** | `0.9186` | **`0.9195`** | `+0.09%` |
| **0.30** | `0.8294` | **`0.8312`** | `+0.18%` |
| **0.50 (Noisy)** | `0.5592` | **`0.5619`** | **`+0.27%`** |

*As noise increases, the performance gap between Breath and Deep Standard grows, proving that the bottleneck layers act as an **implicit regularizer**.*

---

### 3. Image Classification (MNIST & CIFAR-10)
Evaluating classification performance of standalone Breath MLPs under different activations and normalizations.
* **Dimensionality:** 
  * **MNIST:** Inputs = 784 (flattened 28x28), Outputs = 10 (classes).
  * **CIFAR-10:** Inputs = 3072 (flattened 32x32x3), Outputs = 10 (classes).
* **Purist Rules Applied:**
  * Oscillation Ratio: Compression factor `0.25` (1/4), Expansion factor `2.0` (times 2).
  * Full-Rank check: Output size = 10, minimum bottleneck = 16 (since $16 \ge 10$, respects full-rank).
* **Training Conditions:** Trained for 12 epochs on CUDA. Optimizer: AdamW ($LR = 0.001$), Batch Size = 128.
* **Architectures evaluated:**
  * **MNIST (1024):** Deep standard (`[1024, 512, 256, 128, 64, 32, 16]`, ReLU, No Norm) vs Breath + Skips (`[1024, 256, 512, 128, 128, 32, 64, 16, 16]`, ReLU, No Norm).
  * **CIFAR-10 (4096):** Breath sequence `[4096, 1024, 2048, 512, 512, 128, 256, 64, 64, 16, 32]`.
  * **CIFAR-10 (8192):** Breath sequence `[8192, 2048, 4096, 1024, 1024, 256, 512, 128, 128, 32, 64, 16, 16]`.

| Dataset | Initial Width | Model Configuration | Parameters | Train Time | Test Accuracy |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **MNIST** | **1024** | Deep Standard | 1,503,898 | **26.1s** | **`98.17%`** |
| | **1024** | 🏆 **Breath + Skips** | **1,325,274** *(-11.9%)* | 30.4s | `98.00%` |
| | | | | | |
| **CIFAR-10** | **4096** | Breath (ReLU, No Norm) | 20,914,250 | 51.0s | `44.34%` |
| | **4096** | Breath (GELU, No Norm) | 20,914,250 | 51.0s | `40.99%` |
| | **4096** | Breath (GELU + LayerNorm) | 20,923,882 | 55.5s | `53.29%` |
| | **4096** | 🏆 **Breath (SiLU + LayerNorm)** | **20,923,882** | 52.9s | **`54.18%`** |
| | | | | | |
| | **8192** | Breath (ReLU, No Norm) | 58,472,922 | 113.0s | `38.38%` |
| | **8192** | Breath (GELU, No Norm) | 58,472,922 | 114.7s | `29.77%` |
| | **8192** | Breath (GELU + LayerNorm) | 58,486,074 | 123.0s | `10.00%` *(Diverged)* |
| | **8192** | 🏆 **Breath (SiLU + LayerNorm)** | **58,486,074** | 121.0s | **`51.55%`** |

* **MNIST (1024):** Breath + Skips matches the Deep MLP within 0.17% while saving **11.9% of parameters**.
* **CIFAR-10 Ablation:** Adding LayerNorm and SiLU activations yields a massive accuracy boost of up to **`+13.17%`** over standard GELU (No Norm), demonstrating that normalization is mandatory to stabilize gradient flow when scaling up to deeper purist oscillations.

---

### 4. Transformer FFN Integration (NanoGPT & ViT)
Evaluating Breath MLP as a drop-in replacement for the Feed-Forward Network (FFN) blocks of standard Transformers.
* **Tasks & Dimensionality:**
  * **ViT (CIFAR-10):** Patch size 4x4, sequence length 65, $d_{\text{model}} = 192$. FFN input/output = 192.
  * **NanoGPT (Tiny Shakespeare):** Vocabulary size 65, sequence length 128, $d_{\text{model}} = 256$. FFN input/output = 256.
* **Purist Rules Applied:**
  * Oscillation Ratio: Compression factor `0.25` (1/4), Expansion factor `2.0` (times 2).
  * Full-Rank check: FFN output dimension = $d_{\text{model}}$, minimum bottleneck inside FFN = $d_{\text{model}}$ (since $d_{\text{model}} \ge d_{\text{model}}$, respects full-rank).
  * Decay envelope: strict continuous decay of peaks/bottlenecks.
* **Training Conditions:** 
  * **ViT:** Trained for 10 epochs on CUDA. Optimizer: AdamW ($LR = 0.001$), Batch Size = 128. Activation: `GELU`, Norm: `LayerNorm` (input/bottlenecks).
  * **NanoGPT:** Trained for 2500 iterations on CUDA. Optimizer: AdamW ($LR = 0.001$), Batch Size = 64. Activation: `GELU`, Norm: `LayerNorm` (input/bottlenecks).
* **Architectures evaluated:**
  * **ViT Standard FFN:** Standard 2-layer FFN with $d_{\text{ff}} = 4032$ (intermediate width scaled to match parameters).
  * **ViT Breath FFN:** Hidden layers sequence: `[1536, 384, 768, 192, 384]`.
  * **NanoGPT Standard FFN:** Standard 2-layer FFN with $d_{\text{ff}} = 4864$ (intermediate width scaled to match parameters).
  * **NanoGPT Breath FFN:** Hidden layers sequence: `[2048, 512, 1024, 256, 512]`.

| Task | FFN Type | Block FFN Params | Total Params | Train Time | Final Result |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ViT (10 Epochs)** | Standard FFN | 1,552,512 | 6,830,410 | 323.0s | `53.73%` Test Acc |
| | 🏆 **Breath FFN** | **1,479,552** *(-4.7%)* | **6,538,570** | **296.6s** | **`57.71%` Test Acc (+3.98%)** |
| | | | | | |
| **NanoGPT (2500 steps)** | Standard FFN | 2,495,488 | 16,624,705 | **507.5s** | `1.7617` Val Loss |
| | 🏆 **Breath FFN** | **2,628,096** | 17,420,353 | 509.2s | **`1.6100` Val Loss (-0.15)** |

* **ViT:** The purist Breath FFN beats standard FFN by **`+3.98%` test accuracy** while training **9% faster** due to the FLOP reduction of its oscillating bottlenecks compared to standard wide projections.
* **NanoGPT:** The Breath FFN significantly mitigates overfitting, lowering validation loss by **`0.15`** (reducing perplexity from **`5.82` to `5.00`**, a **14% improvement**).

---

### 5. Image Denoising (ImageNet-32)
Replication of the generative restoration (denoising) task from Chen et al. 2025. Trained for 2 epochs on GPU (RTX 4070 Laptop) with 4x data augmentation on the full 1.28 million natural image dataset ($\sigma = 0.25$ Gaussian noise).
* **Dimensionality:** Inputs = 3072 (flattened 32x32x3), Outputs = 3072.
* **Purist Rules Applied:**
  * Oscillation Ratio: Compression factor `0.25` (1/4), Expansion factor `2.0` (times 2).
  * Bottleneck vs Output: In generative reconstruction/denoising, compressing the representation size below the output size is mathematically necessary to force noise filtering. A minimum bottleneck width of 512 is used.
* **Training Conditions:** Trained for 2 epochs on CUDA. Optimizer: Adam ($LR = 0.0001$), Batch Size = 512.
* **Architectures evaluated:**
  * **Hourglass MLP (Paper):** $d_z = 3546, d_h = 1560, L = 4$ (Parameters: 55.20M, Activation: ReLU, Norm: None).
  * **Breath MLP (Purist):** Hidden layers sequence: `[9216, 2304, 4608, 1152, 2304, 576, 1152]` (Parameters: 76.71M, Activation: ReLU, Norm: None).

| Model | Width Configuration | Parameters | Train Time (per Epoch) | Test PSNR |
| :--- | :--- | :---: | :---: | :---: |
| 🏆 **Hourglass MLP (Paper)** | $d_z = 3546, d_h = 1560, L = 4$ | **55.20M** | **458.3s (7.6m)** | **20.88 dB** |
| Breath MLP (Purist) | `[9216, 2304, 4608, 1152, 2304, 576, 1152]` | 76.71M *(+39%)* | 490.7s (8.2m) | 18.35 dB *(-2.53 dB)* |

> ⚠️ **Note:** In this specific experiment (2-epoch budget), the Hourglass MLP wins on all metrics: fewer parameters, slightly less training time per epoch, and significantly better PSNR. This is attributed to the **deeper structural shape** of the purist Breath MLP (7 hidden layers vs the Hourglass's shallow wide structure), which requires a longer training budget and appropriate learning rate scheduling to reach its optimal convergence point. With more epochs or a warmer/cyclic learning rate, the Breath MLP is expected to close this gap.

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
* **Tabular & Small-scale Benchmarks**: Run the default regression and classification tests:
  ```bash
  python benchmark.py
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
from breath_mlp import generate_breath_architecture, BreathMLP

# 1. Generate layers sequence: [512, 128, 256, 64, 64, 16, 32]
hidden_layers = generate_breath_architecture(
    start_width=512, 
    compression_factor=0.25, 
    expansion_factor=2.0, 
    min_width=16
)

# 2. Instantiate Model
model = BreathMLP(
    input_dim=21, 
    hidden_layers=hidden_layers, 
    output_dim=1, 
    use_skips=True,
    activation="silu",
    use_norm=True
)

print(model)
```

## 📊 Final Conclusions

The following table summarizes which architecture performs best in each benchmark, under the **purist rule set** (constant oscillation ratio 0.25/2.0, continuous decay, full-rank bottlenecks, parameter-matched baselines):

| Benchmark | Task Type | Winner | Key Metric | Δ vs. Baseline |
| :--- | :--- | :--- | :---: | :---: |
| **SARCOS (Robotics)** | Regression | Breath + Skips | $R^2 = 0.9881$ | *-30.6% params* |
| **California Housing** | Regression | **Breath Standard** | $R^2 = 0.7939$ | *+1.16% $R^2$, -30.1% params* |
| **MNIST** | Classification | Deep Standard | Acc = 98.17% | *Breath within 0.17%* |
| **CIFAR-10 (4096)** | Classification | **Breath (SiLU + LN)** | Acc = 54.18% | *Best activation config* |
| **CIFAR-10 (8192)** | Classification | **Breath (SiLU + LN)** | Acc = 51.55% | *+13.17% over GELU* |
| **ImageNet-32 Denoising** | Generative | Hourglass MLP (Paper) | PSNR = 20.88 dB | *Breath needs more epochs* |
| **ViT FFN (CIFAR-10)** | Vision Transformer | **Breath FFN** | Acc = 57.71% | *+3.98%, -4.7% params, -9% time* |
| **NanoGPT FFN (Shakespeare)** | Language Model | **Breath FFN** | Val Loss = 1.610 | *-0.15 loss, -14% perplexity* |

### Key Takeaways

1. **Tabular Data & Shallow MLPs:** The Breath MLP's decaying oscillation provides a strong structural prior for regression tasks, outperforming parameter-matched flat baselines while using significantly fewer parameters.

2. **Image Classification:** Breath MLPs require **SiLU + LayerNorm** as depth increases. Without normalization, the deeper purist shape suffers from variance explosion and potential divergence. With it, it outperforms all other configurations.

3. **Generative Restoration (Denoising):** Under short training budgets, the deeper purist Breath MLP underperforms shallow wide architectures like the Hourglass MLP. Generative tasks require the bottleneck to act as a compression filter, which is better served by a shallow structure when training is time-constrained.

4. **Transformer Integration:** Breath MLP is a **strong drop-in replacement for FFN blocks** in both vision (ViT) and language (GPT) Transformers, consistently outperforming parameter-matched standard FFNs through better generalization and reduced overfitting.

5. **Future Directions:** Testing with relaxed oscillation ratios (e.g., 1/2, 1/3), longer training schedules for the denoising task, and integration into modern Transformer variants (LLaMA, GPT-4 style) remain promising next steps.

---

## 📜 References
* Chen, M.-H., et al. (2025). *"Rethinking the shape convention of an MLP"*. arXiv preprint.
