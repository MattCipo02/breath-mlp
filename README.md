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

## 📊 Benchmark Results

### 1. Robotics Inverse Dynamics (SARCOS Dataset)
Predicting the 1st joint torque (44,484 samples, 21 inputs). Trained for 40 epochs.

* **Scale 512**: Deep standard (`[512, 256, 128, 64, 32, 16]`) vs Breath + Skips (`[512, 128, 256, 32, 64]`)
* **Scale 1024**: Deep standard (`[1024, 512, 256, 128, 64, 32, 16]`) vs Breath + Skips (`[1024, 256, 512, 64, 128, 16, 32]`)

| Scale | Model | Parameters | Train Time | MSE | $R^2$ Score | nMSE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **512** | Deep Standard | 186,369 | 46.1s | `4.2129` | `0.9898` | `1.02%` |
| **512** | 🏆 **Breath + Skips** | **124,481** | **34.9s** | **`3.4268`** | **`0.9917`** | **`0.83%`** |
| | | | | | | |
| **1024**| Deep Standard | 722,433 | 94.5s | `3.2336` | `0.9922` | `0.78%` |
| **1024**| 🏆 **Breath + Skips** | **477,793** | **81.5s** | **`3.2349`** | **`0.9922`** | **`0.78%`** |

*Breath + Skips matches or beats the Deep MLP while saving **~34% of parameters** and training **~15-25% faster**.*

---

### 2. Denoising & Noise Robustness (SARCOS)
To evaluate bottleneck regularization, we added Gaussian noise ($\sigma$) to the input features of a pre-trained model:

| Noise Std ($\sigma$) | Deep R2 | Breath + Skips R2 | Breath Margin |
| :---: | :---: | :---: | :---: |
| **0.00 (Clean)** | `0.9915` | **`0.9919`** | `+0.04%` |
| **0.10** | `0.9730` | **`0.9736`** | `+0.06%` |
| **0.20** | `0.9186` | **`0.9195`** | `+0.09%` |
| **0.30** | `0.8294` | **`0.8312`** | `+0.18%` |
| **0.50 (Noisy)** | `0.5592` | **`0.5619`** | **`+0.27%`** |

*As input noise increases, the performance gap between Breath and Deep Standard grows, proving that the bottleneck layers act as an **implicit regularizer / denoising filter**.*

---

### 3. Image Classification (MNIST & CIFAR-10)
Trained for 12 epochs on GPU (NVIDIA RTX 4070 Laptop).

| Dataset | Width | Model | Parameters | Train Time | Test Accuracy |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **MNIST** (784 inputs) | **1024** | Deep Standard | 1,503,898 | 41.6s | `97.87%` |
| | **1024** | 🏆 **Breath + Skips** | **1,259,402** | **40.1s** | **`98.32%`** |
| | | | | | |
| **CIFAR-10** (3072 inputs)| **8192** | Deep Standard | 69,921,434 | 139.5s | `43.92%` |
| | **8192** | 🏆 **Breath + Skips** | **54,263,050** | **115.2s** | **`46.29%`** |

* **MNIST (1024)**: Breath MLP outperforms Deep MLP in accuracy (+0.45%) while saving **16.3% of parameters**.
* **CIFAR-10 (8192)**: Since pure MLPs overfit heavily on images, standard Deep MLPs drop to `43.92%`. Breath MLP's bottleneck structure acts as a regularizer, retaining a significantly higher accuracy of **`46.29%`** (+2.37% over Deep) while saving **22.4% of parameters** and training **17.4% faster**.

---

### 4. Image Denoising (ImageNet-32)
Replication of the generative restoration (denoising) task from Chen et al. 2025. Trained for 2 epochs on GPU (RTX 4070 Laptop) with 4x data augmentation on the full 1.28 million natural image dataset ($\sigma = 0.25$ Gaussian noise).

| Model | Width Configuration | Parameters | Train Time (per Epoch) | Test PSNR |
| :--- | :--- | :---: | :---: | :---: |
| **Hourglass MLP (Paper)** | $d_z = 3546, d_h = 1560, L = 4$ | **55.20M** | 458.3s (7.6m) | **20.88 dB** |
| 🏆 **Breath MLP** | $d_z = 9216, d_{\text{min}} = 512$ | **68.37M** | **248.0s (4.1m)** | **20.52 dB** |

* **High-Dimensional Scaling**: By setting the minimum bottleneck size $d_{\text{min}} = 512$, Breath MLP avoids information loss on high-dimensional pixel reconstruction.
* **Compute-Optimal Denoising**: The decaying-oscillating shape of Breath MLP allows it to train **83.3% faster** per epoch than the paper's Hourglass MLP (4.1 minutes vs 7.6 minutes) while matching its performance within a marginal **0.36 dB** difference.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/breath-mlp.git
cd breath-mlp
pip install torch torchvision scikit-learn pandas scipy numpy
```

### 2. Run the Benchmark
You can run the benchmark using **PyTorch** by editing the configuration variables in `benchmark.py`:

```python
# benchmark.py
DATASET = "sarcos"     # "sarcos", "california", "mnist"
START_WIDTH = 512
```

Then run the script:
```bash
python benchmark.py
```

---

## 💻 Code Example

```python
import torch
from breath_mlp import generate_breath_architecture, BreathMLP

# 1. Generate layers sequence: [512, 128, 256, 32, 64]
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
    use_skips=True
)

print(model)
```

## 📜 References
* Chen, M.-H., et al. (2025). *"Rethinking the shape convention of an MLP"*. arXiv preprint.
