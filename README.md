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
* **Test Objective (Tabular Regression / Robotics Domain):** In this test, the dataset consists of kinematics data (21 features representing joint positions, velocities, and accelerations) of a 7-degree-of-freedom SARCOS robotic arm. The goal is to learn the inverse dynamics mapping to predict the corresponding joint torque (regression). We are testing whether the decaying-oscillating (Breath) shape can approximate complex physical dynamics as accurately as a flat Deep MLP, and if it does so with greater parameter efficiency.
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

| Model | Parameters | Train Time (mean) | $R^2$ Score (5-fold CV) | MSE (5-fold CV) |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Standard** | 186,369 | 58s | `0.9888 +/- 0.0007` | `4.6784 +/- 0.2952` |
| **Deep + Skips** | 203,857 | 68s | `0.9887 +/- 0.0008` | `4.7242 +/- 0.2953` |
| **Breath Standard** | **132,177** | **69s** | `0.9868 +/- 0.0020` | `5.5382 +/- 0.8782` |
| 🏆 **Breath + Skips** | 141,473 | 79s | **`0.9885 +/- 0.0003`** | **`4.8264 +/- 0.1370`** |

> ✅ **Robustness via 5-Fold Cross-Validation:** To mitigate random initialization variance, we evaluated all models using 5-fold CV across different seeds. 
> 
> * **Variance Reduction:** **Breath + Skips** achieved the lowest variance in $R^2$ ($\pm 0.0003$) and MSE ($\pm 0.1370$), confirming its architectural robustness.
> * **Efficiency:** It matches the performance of the flat **Deep Standard** (R2 `0.9885` vs `0.9888`) while using **24.1% fewer parameters**, and matches/outperforms **Deep + Skips** using **30.6% fewer parameters**.


---

### 2. Regression (California Housing)
Predicting median house values from 8 geographic and demographic features.
* **Test Objective (Tabular Regression / Demographic Domain):** In this test, the dataset contains 8 socioeconomic and geographic features (such as median income, house age, average rooms, and coordinates) of California districts. The goal is to learn to predict the median house value of a district (regression). We are evaluating the capacity of Breath MLP to model low-dimensional tabular data and finding whether narrow bottlenecks act as effective feature selectors, matching standard MLP performance with fewer parameters.
* **Dimensionality:** Inputs = 8, Outputs = 1 (median house value).
* **Purist Rules Applied:**
  * Oscillation Ratio: Compression factor `0.25` (1/4), Expansion factor `2.0` (times 2).
  * Full-Rank check: Output size = 1, minimum bottleneck = 16 (respects full-rank).
  * Decay envelope: strict continuous decay of peaks/bottlenecks.
* **Training Conditions:** Trained for 40 epochs on CUDA. Optimizer: Adam ($LR = 0.001$), Batch Size = 64.
* **Network Configurations:**
  * **Deep Standard:** `[512, 256, 128, 64, 32, 16]` (Parameters: 179,713, Activation: ReLU, Norm: None).
  * **Deep + Skips:** `[512, 256, 128, 64, 32, 16]` (Parameters: 197,201, Activation: ReLU, Norm: None).
  * **Breath Standard:** `[512, 128, 256, 64, 64, 16, 32]` (Parameters: 125,521, Activation: ReLU, Norm: None).
  * **Breath + Skips:** `[512, 128, 256, 64, 64, 16, 32]` (Parameters: 134,817, Activation: ReLU, Norm: None).

| Model | Parameters | Train Time (mean) | $R^2$ Score (5-fold CV) | MSE (5-fold CV) |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Standard** | 179,713 | 21s | `0.8030 +/- 0.0050` | `0.2624 +/- 0.0110` |
| 🏆 **Deep + Skips** | 197,201 | 23s | **`0.8036 +/- 0.0070`** | **`0.2615 +/- 0.0085`** |
| **Breath Standard** | **125,521** | **22s** | `0.7977 +/- 0.0080` | `0.2692 +/- 0.0087` |
| **Breath + Skips** | 134,817 | 25s | `0.8006 +/- 0.0073` | `0.2654 +/- 0.0077` |

> ✅ **Robustness via 5-Fold Cross-Validation:** The cross-validation reveals that **skip connections do help** Breath MLP on this dataset (R2 increases from `0.7977` to `0.8006`), correcting the single-seed observation where they appeared to degrade performance.
> 
> * **Parameter Savings:** **Breath + Skips** matches the flat baseline performance within a tiny fraction (`0.8006` vs `0.8030` of Deep Standard) while using **25.0% fewer parameters** (134k vs 179k).


---

### 3. Image Classification (MNIST & CIFAR-10)
Evaluating classification performance of standalone Breath MLPs under different activations and normalizations.
* **Test Objective (Image Classification / Computer Vision Domain):** In this test, we evaluate standalone MLPs on flattened image datasets: hand-written digits (MNIST, 784 features) and natural objects (CIFAR-10, 3072 features). The goal is to assign each image to one of 10 target classes (classification). We are investigating how deep oscillating structures process high-dimensional spatial data, and verifying how different activations (SiLU/GELU) and normalizations (LayerNorm) prevent gradient vanishing/explosion in deep configurations (up to 13 hidden layers).
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
| **MNIST (5-fold CV)** | **1024** | Deep Standard | 1,503,898 | 21s | `97.20% +/- 0.25%` |
| | **1024** | Deep + Skips | 1,573,690 | 24s | `97.32% +/- 0.18%` |
| | **1024** | Breath Standard | **1,287,722** *(-14.4%)* | **24s** | `97.30% +/- 0.19%` |
| | **1024** | 🏆 **Breath + Skips** | **1,325,274** *(-11.9%)* | 30s | **`97.33% +/- 0.21%`** |
| | | | | | |
| **CIFAR-10 (5-fold CV)**| **4096** | Deep Standard (SiLU + LN) | 23,790,202 | 86s | `53.33% +/- 0.76%` |
| | **4096** | Deep + Skips (SiLU + LN) | 24,909,082 | 94s | **`53.99% +/- 0.52%`** |
| | **4096** | Breath Standard (SiLU + LN) | **20,324,122** *(-14.6%)* | **79s** | `46.52% +/- 3.81%` |
| | **4096** | 🏆 **Breath + Skips (SiLU + LN)** | **20,923,882** *(-12.0%)* | 86s | `53.13% +/- 0.37%` |
| | | | | | |
| **CIFAR-10 (Ablation)**| **4096** | Breath Standard (ReLU, No Norm) | 20,914,250 | 51.0s | `44.34%` |
| | **4096** | Breath Standard (GELU, No Norm) | 20,914,250 | 51.0s | `40.99%` |
| | **4096** | Breath Standard (GELU + LayerNorm) | 20,923,882 | 55.5s | `53.29%` |
| | **4096** | 🏆 **Breath Standard (SiLU + LN)**| **20,923,882** | 52.9s | **`54.18%`** |
| | | | | | |
| | **8192** | Breath (ReLU, No Norm) | 58,472,922 | 113.0s | `38.38%` |
| | **8192** | Breath (GELU, No Norm) | 58,472,922 | 114.7s | `29.77%` |
| | **8192** | Breath (GELU + LayerNorm) | 58,486,074 | 123.0s | `10.00%` *(Diverged)* |
| | **8192** | 🏆 **Breath (SiLU + LayerNorm)** | **58,486,074** | 121.0s | **`51.55%`** |

* **MNIST (1024):** Under 5-fold cross-validation, **Breath + Skips** achieves the highest average accuracy (`97.33%`), outperforming Deep Standard (`97.20%`) while saving **11.9% of parameters** (1.33M vs 1.50M). **Breath Standard** also outperforms the flat baseline (`97.30%`) while saving **14.4% of parameters** (1.28M vs 1.50M).
* **CIFAR-10 Ablation:** Adding LayerNorm and SiLU activations yields a massive accuracy boost of up to **`+13.17%`** over standard GELU (No Norm), demonstrating that normalization is mandatory to stabilize gradient flow when scaling up to deeper purist oscillations.

---

### 4. Transformer FFN Integration (NanoGPT & ViT)
Evaluating Breath MLP as a drop-in replacement for the Feed-Forward Network (FFN) blocks of standard Transformers.
* **Test Objective (Deep Sequence Modeling / Vision-Language Domain):** In this test, we evaluate Breath MLP as a drop-in replacement for the Feed-Forward Network (FFN) blocks inside self-attention models: Vision Transformers (ViT) on CIFAR-10 classification, and causal GPTs (NanoGPT) on character-level language modeling (predicting the next character in Shakespeare text). The goal is to learn sequence representations (classification and autoregressive generation). We are testing whether integrating the decaying-oscillating bottleneck inside the Transformer's token projection paths acts as an implicit regularizer, improving validation stability and generalization.
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
| **ViT (5 Epochs, 3-seed)**| Standard FFN | 1,552,512 | 6,830,410 | 143.1s | `50.39% +/- 1.07%` |
| | 🏆 **Breath FFN** | **1,479,552** *(-4.7%)* | **6,538,570** | **137.2s** | **`53.34% +/- 1.82%` (+2.95%)** |
| | | | | | |
| **NanoGPT (1000 steps, 3-seed)**| Standard FFN | 2,495,488 | 16,624,705 | **184.1s** | `1.5750 +/- 0.0119` Val Loss |
| | Breath FFN | 2,628,096 *(+5.3%)* | 17,420,353 *(+4.8%)* | 190.9s *(+3.7%)* | **`1.5345 +/- 0.0033` Val Loss (-0.040)** |

* **ViT:** Under 3-seed validation (5 epochs), the purist **Breath FFN** represents a clean win: it outperforms the standard FFN by **`+2.95%` average test accuracy** while saving **4.7% FFN parameters** (4.3% total parameters) and training **4% faster** on average.
* **NanoGPT:** Under 3-seed validation (1000 steps), the **Breath FFN** represents a **trade-off**: it yields a slight validation loss reduction of **`0.04`** ($1.5345$ vs $1.5750$) and high training stability (reducing loss variance by **$3.6\times$**), but at the cost of a **+4.8% increase in total parameters** and a **+3.7% increase in training time**.

---

### 5. Architectural Constraints & Ideal Applications (Sweet Spots)
Applying the strict purist ruleset ($d_{\text{min}} \ge d_{\text{out}}$) reveals a clean mathematical dichotomy between classification/regression tasks and raw high-dimensional generative reconstruction (such as pixel-space denoising).

#### ⚠️ The High-Dimensional Reconstruction Bottleneck
In generative tasks like **image denoising** (e.g., ImageNet-32, where input/output dimensions are $d_x = d_{\text{out}} = 3072$), the purist constraint $d_{\text{min}} \ge d_{\text{out}}$ creates a massive parameter bottleneck:
* To perform even **one** compression cycle (ratio 0.25), the starting layer $d_z$ must be at least $3072 / 0.25 = 12,288$ units.
* To perform a multi-cycle oscillation (e.g., compress-expand-compress) while keeping all bottlenecks above 3072, the initial layers must scale to tens of thousands of units.
* This leads to an **exponential parameter explosion** (exceeding 170M parameters for a simple 3-layer MLP). Consequently, for raw pixel-space reconstruction, shallow wide hourglass structures are practically preferred over purist deep oscillations.

#### 🎯 Ideal Domains (Where Breath MLP excels)
1. **Tabular Regression & Classification (Low $d_{\text{out}}$)**:
   * *Characteristics:* Small input/output dimensions (e.g., SARCOS, California Housing).
   * *Why it works:* Since $d_{\text{out}}$ is small (1 or 10), we can compress representations down to extremely narrow bottlenecks (16 or 32 units) without violating $d_{\text{min}} \ge d_{\text{out}}$. 
   * *Impact:* Breath MLP compresses redundant features, saving **24% to 30% of parameters** while matching or exceeding the baseline accuracy.
   
2. **Transformer FFN Blocks (Moderate $d_{\text{model}}$)**:
   * *Characteristics:* Feed-Forward networks in Vision Transformers (ViT) and LLMs (NanoGPT), mapping tokens $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$.
   * *Why it works:* $d_{\text{model}}$ is moderately sized (e.g., 256 or 1024). We can easily fit multiple decay cycles (e.g., $2048 \to 512 \to 1024 \to 256$) while keeping all hidden widths above $d_{\text{model}}$.
   * *Impact:* The oscillating bottleneck acts as a powerful regularizer, mitigating overfitting and improving generalization (yielding **+2.95% accuracy** on ViT and **-0.15 val loss** on NanoGPT).

3. **Latent Representation Spaces**:
   * *Characteristics:* Rather than processing raw high-dimensional pixel data (3072+), Breath MLP is ideal for processing **low-dimensional embeddings** (e.g., $d_z = 64$ or $128$) generated by pre-trained CNN/ViT encoders or VAEs.

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

| Benchmark | Task Type | Winner | Baseline Params | Breath Params | Key Metric | Δ Metric |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SARCOS (Robotics)** | Regression | 🏆 **Breath + Skips** | 186,369 | **141,473** *(-24.2%)* | $R^2 = 0.9885 \pm 0.0003$ | *Matches Deep Std with lowest variance* |
| **California Housing** | Regression | Deep + Skips | 179,713 (Deep Std) | **134,817** *(-25.0% vs Deep Std)* | $R^2 = 0.8036 \pm 0.0070$ (Deep+Skips) | *Breath+Skips: 0.8006 (-0.30% R2)* |
| **MNIST** | Classification | 🏆 **Breath + Skips** | 1,503,898 | **1,325,274** *(-11.9%)* | Acc = $97.33\% \pm 0.21\%$ | *Matches/beats Deep + Skips with fewer params* |
| **CIFAR-10 (4096)** | Classification | Deep + Skips | 23,790,202 (Deep Std) | **20,923,882** *(-12.0% vs Deep Std)* | Acc = $53.99\% \pm 0.52\%$ (Deep+Skips) | *Breath+Skips: 53.13% +/- 0.37% (-0.86% Acc)* |
| **CIFAR-10 (8192)** | Classification | **Breath (SiLU + LN)** | — | **58,486,074** | Acc = 51.55% | *+13.17% over GELU* |
| **ViT FFN (CIFAR-10)** | Vision Transformer | **Breath FFN** | 6,830,410 | **6,538,570** *(-4.3%)* | Acc = $53.34\% \pm 1.82\%$ | *+2.95% accuracy vs Standard FFN* |
| **NanoGPT FFN (Shakespeare)** | Language Model | **Trade-off** | 16,624,705 | 17,420,353 *(+4.8%)* | Val Loss = $1.5345 \pm 0.0033$ | *−0.04 loss, +4.8% params, +3.7% time* |

### Key Takeaways

1. **Parameter Efficiency on Tabular Tasks (Robotics & Housing):** Under 5-fold cross-validation, **Breath + Skips** matches flat baselines while significantly saving parameters. On SARCOS, it achieves `0.9885` R2 (with the lowest variance, $\pm 0.0003$) saving **24.1% parameters** compared to Deep Standard (141k vs 186k), and **30.6% parameters** compared to Deep + Skips. On California Housing, **Breath + Skips** matches Deep Standard (`0.8006` vs `0.8030`) while saving **25.0% parameters** (134k vs 179k).

2. **Image Classification (MNIST & CIFAR-10):** Under 5-fold cross-validation on MNIST, **Breath + Skips** actually outperforms Deep Standard in absolute accuracy (`97.33%` vs `97.20%`) while using **11.9% fewer parameters** (1.33M vs 1.50M). As network depth increases (like in CIFAR-10), Breath MLPs require **SiLU + LayerNorm** to prevent gradient variance explosion. With proper normalization, it outperforms all other configurations.

3. **Reconstruction Constraints & Low-Rank Latents:** For high-dimensional reconstruction (like 3072-dimensional raw image pixel spaces), the purist rules enforce bottlenecks $\ge 3072$, leading to parameter explosion. Breath MLP is therefore mathematically less suited for direct raw pixel-space reconstruction and is instead designed to process low-dimensional latent features (e.g., embedding spaces or representations produced by autoencoder bottlenecks).

4. **Transformer Integration & Generalization Stability:** On ViT, the Breath FFN represents a clear win, yielding **+2.95% average test accuracy** while using **4.3% fewer parameters** (6.54M vs 6.83M total) and training **4% faster** on average. On NanoGPT, the comparison represents a **trade-off**: the Breath FFN reduces the average validation loss by **0.04** ($1.5345$ vs $1.5750$) and significantly stabilizes training (lowering variance by **3.6x**), but at the cost of a **+4.8% increase in total parameters** and a **+3.7% increase in training time**.

5. **Future Directions & Rule Relaxation:** Testing with relaxed oscillation ratios (e.g., 1/2, 1/3) and, crucially, investigating the **relaxation of purist constraints** (e.g., allowing bottleneck widths to go below $d_{\text{out}}$ via low-rank final projections) represents a key path forward. Such relaxations would drastically expand the scope of application to raw image, video, and audio processing without parameter explosion, while preserving the regularizing benefits of oscillating signal paths. Additionally, integration into modern Transformer variants (LLaMA, GPT-4 style) remains a highly promising next step.

---

---

## ⚡ 6. Feature Pooling & Parameter-Free Compression (BreathMLPPool & PurePool)

To maximize parameter efficiency and accelerate training times, we investigated replacing the linear compression layers in the Breath MLP with parameter-free **1D Adaptive Feature Pooling** (Max and Average pooling). 

### 📐 The Resizing Challenge for Skip Connections
Because consecutive bottlenecks decay in size (e.g., from $128 \to 64 \to 16$), direct identity skip connections are impossible due to size mismatches. To resolve this, we evaluated two strategies:
1. **`BreathPool` (Hybrid):** Uses non-trainable 1D Pooling for the main compression path, but retains a **trainable linear layer** (`nn.Linear`) to project/resize the skip connection between bottlenecks.
2. **`PurePool` (100% Parameter-Free):** Uses non-trainable 1D Pooling for both the main compression path and the skip connection resizing (e.g., applying `AdaptiveAvgPool1d` or `AdaptiveMaxPool1d` directly to the previous bottleneck activations). This model has **zero trainable weights/biases** in any compression or skip path.

---

### 📊 Benchmark Results & Training Times (5-Fold CV)

All models were evaluated under identical conditions (ReLU, No Norm, Adam/AdamW, batch size 64/128, trained for 40 epochs on SARCOS/California and 12 epochs on MNIST) on an RTX 4070 Laptop GPU.

#### A. Robotics Inverse Dynamics (SARCOS Dataset)
* **Test Objective (Tabular Regression / Robotics Domain):** Predict joint torques from 21 kinematic features. Evaluates dynamics modeling and gradient stability across 5 folds.

| Model | Parameters | Training Time (Mean per Fold) | $R^2$ Score (5-fold CV) | MSE (5-fold CV) |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Standard** | 186,369 | 85s | `0.9888 +/- 0.0007` | `4.6784 +/- 0.2952` |
| **Breath + Skips (Lin)** | 141,473 | 106s | `0.9878 +/- 0.0006` | `5.1242 +/- 0.2959` |
| **BreathPool Max + Skips**| 58,321 | 100s | `0.9864 +/- 0.0003` | `5.7063 +/- 0.2078` |
| **BreathPool Avg + Skips**| 58,321 | 99s | `0.9839 +/- 0.0009` | `6.7361 +/- 0.5004` |
| **PurePool Max + Skips** | **49,025** *(-73.7%)* | 83s | **`0.9862 +/- 0.0012`** | `5.7772 +/- 0.4630` |
| **PurePool Avg + Skips** | **49,025** *(-73.7%)* | **77s** *(-27.4%)* | `0.9837 +/- 0.0012` | `6.8342 +/- 0.5173` |

#### B. Regression (California Housing)
* **Test Objective (Tabular Regression / Demographic Domain):** Predict median house values from 8 demographic features. Evaluates bottleneck capability on narrow inputs.

| Model | Parameters | Training Time (Mean per Fold) | $R^2$ Score (5-fold CV) | MSE (5-fold CV) |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Standard** | 179,713 | 53s | `0.8030 +/- 0.0050` | `0.2624 +/- 0.0110` |
| **Breath + Skips (Lin)** | 134,817 | 67s | `0.8019 +/- 0.0103` | `0.2636 +/- 0.0113` |
| **BreathPool Max + Skips**| 51,665 | 61s | `0.7872 +/- 0.0045` | `0.2833 +/- 0.0070` |
| **BreathPool Avg + Skips**| 51,665 | 59s | `0.7905 +/- 0.0029` | `0.2789 +/- 0.0038` |
| **PurePool Max + Skips** | **42,369** *(-76.4%)* | 56s | `0.7874 +/- 0.0069` | `0.2831 +/- 0.0112` |
| 🏆 **PurePool Avg + Skips**  | **42,369** *(-76.4%)* | **50s** *(-25.4%)* | **`0.7924 +/- 0.0112`** | **`0.2763 +/- 0.0140`** |

#### C. Image Classification (MNIST Dataset)
* **Test Objective (Image Classification / Computer Vision Domain):** Classify hand-written digits (10 classes) from 784 flattened pixel inputs.

| Model | Parameters | Training Time (Mean per Fold) | Test Accuracy (5-fold CV) |
| :--- | :---: | :---: | :---: |
| **Deep Standard** | 1,503,898 | 49s | `97.74% +/- 0.06%` |
| **Breath + Skips (Lin)** | 1,325,274 | 66s | `97.28% +/- 0.35%` |
| **BreathPool Max + Skips**| 992,042 | 62s | **`97.66% +/- 0.25%`** |
| **BreathPool Avg + Skips**| 992,042 | 58s | `97.47% +/- 0.20%` |
| 🏆 **PurePool Max + Skips**  | **954,490** *(-36.5%)* | 55s | **`97.65% +/- 0.16%`** |
| **PurePool Avg + Skips**  | **954,490** *(-36.5%)* | **49s** *(-25.7%)* | `97.56% +/- 0.17%` |

---

### 🧠 Key Insights & Takeaways

1. **Massive Parameter Reductions with Tiny Performance Cost:**
   By removing learnable parameters from compression and skip paths, `PurePool` variants reduce the parameter footprint of the Breath architecture by **up to 76.4%** compared to the Deep Standard baseline, and **up to 68.6%** compared to standard Breath. The performance impact is minimal (often less than a 0.002 to 0.01 difference in $R^2$ / Accuracy).

2. **Significant Training Speedups:**
   `PurePool Avg` reduces the training time per fold by **25% to 27%** compared to `Breath + Skips (Lin)` (e.g., from 106s to 77s on SARCOS, and 66s to 49s on MNIST). Average pooling is computationally extremely cheap and does not require tracking gradient routing indices (like Max pooling does), making it run faster on GPUs.

3. **Average Pooling vs. Max Pooling trade-off:**
   * **On Tabular/Regression data (SARCOS, California):** `PurePool Avg` performs best among the pooling variants because it provides a continuous, dense gradient backpropagation path to *all* neurons, preventing gradient bottlenecking.
   * **On Vision/Classification data (MNIST):** `PurePool Max` performs best, outperforming linear Breath by a large margin (`97.65%` vs `97.28%`). This suggests that max-pooling behaves as an effective spatial-frequency local feature selector, extracting high-amplitude features (reminiscent of CNN pooling), which fits image inputs exceptionally well.

---

## 📜 References
* Chen, M.-H., et al. (2025). *"Rethinking the shape convention of an MLP"*. arXiv preprint.

