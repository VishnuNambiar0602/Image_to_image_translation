# Pix2Pix Architecture & Implementation Details

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Generator (U-Net)](#generator-u-net)
3. [Discriminator (PatchGAN)](#discriminator-patchgan)
4. [Training Framework](#training-framework)
5. [Loss Functions](#loss-functions)
6. [Implementation Details](#implementation-details)

---

## Architecture Overview

Pix2Pix is a conditional GAN framework designed for paired image-to-image translation. It consists of:

```
┌─────────────────────────────────────────────────────────┐
│                   Pix2Pix Framework                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Source Image ──→ [Generator (U-Net)] ──→ Generated   │
│      (A)               ~9.0M params          Image      │
│                                                (B')     │
│                            │                            │
│  Target Image            │                              │
│      (B) ─────────────────┴──────────┐                 │
│                                       │                 │
│                           ┌───────────┴──────────┐      │
│                           │                      │      │
│               [Discriminator (PatchGAN)]         │      │
│                   ~1.8M params                   │      │
│                                                 │      │
│                           Real: 1   Fake: 0    │      │
│                                                │       │
└────────────────────────────────────────────────┘
```

---

## Generator (U-Net)

### Architecture

**Encoder-Decoder with Skip Connections**

```
Input (3 channels, HxW)
    ↓
[Conv 64] → H/2 × W/2
    ↓
[Conv 128] → H/4 × W/4
    ↓
[Conv 256] → H/8 × W/8
    ↓
[Conv 512] → H/16 × W/16
    ↓
[Conv 512] → H/32 × W/32
    ↓
[Conv 512] → H/64 × W/64
    ↓
[Conv 512] → H/128 × W/128
    ↓
[Bottleneck: Conv 512] ← deepest layer
    ↓
[ConvTranspose 512] → H/64 × W/64 ──┐ (skip from encoder)
    ↓                                 ↓
[Concat & ConvTranspose 512] → H/32 × W/32 ──┐
    ↓                                          ↓
[Concat & ConvTranspose 512] → H/16 × W/16 ──┐
    ↓                                          ↓
[Concat & ConvTranspose 256] → H/8 × W/8 ───┐
    ↓                                         ↓
[Concat & ConvTranspose 128] → H/4 × W/4 ───┐
    ↓                                         ↓
[Concat & ConvTranspose 64] → H/2 × W/2 ────┐
    ↓                                         ↓
[Concat & ConvTranspose 3] ─→ Output (HxW, 3 channels, range [-1,1])
```

### Key Features

| Component | Specification |
|-----------|---------------|
| **Input** | 3-channel image (normalized to [-1, 1]) |
| **Output** | 3-channel image (tanh activation → [-1, 1]) |
| **Encoder** | 7 convolution blocks (stride=2 downsampling) |
| **Bottleneck** | 512 feature maps at deepest layer |
| **Decoder** | 7 transpose convolution blocks (stride=2 upsampling) |
| **Skip Connections** | Preserves fine-grained details from encoder |
| **Normalization** | InstanceNorm (helps adaptation to new images) |
| **Activation** | ReLU in encoder, ReLU in decoder |
| **Parameters** | ~9.0 Million |
| **Dropout** | 50% in decoder layers (training only) |

### Mathematical Details

**Convolution Block:**
```
y = ReLU(InstanceNorm(Conv(x)))
```

**Transpose Convolution Block:**
```
y = Dropout(ReLU(InstanceNorm(ConvTranspose(x))))
```

**Skip Connection:**
```
y = ConvTranspose(x)
y = cat(y, encoder_feature)  # Concatenate with corresponding encoder layer
```

**Output Layer:**
```
y = Tanh(ConvTranspose(x))  # Values in [-1, 1]
```

---

## Discriminator (PatchGAN)

### Architecture

**Patch-based Discriminator with Spectral Normalization**

```
Concatenated Image (6 channels: 3 from source + 3 from target, H×W)
    ↓
[Conv 64 + LeakyReLU] → stride=2 (receptive field: 4)
    ↓
[Conv 128 + InstanceNorm + LeakyReLU] → stride=2 (RF: 16)
    ↓
[Conv 256 + InstanceNorm + LeakyReLU] → stride=2 (RF: 34)
    ↓
[Conv 512 + InstanceNorm + LeakyReLU] → stride=1 (RF: 70)
    ↓
[Conv 1] → Output (patch-wise classification)
    ↓
Output: (B, 1, H/16, W/16) with values [0, 1] (sigmoid)
```

### Key Features

| Component | Specification |
|-----------|---------------|
| **Input** | 6-channel concatenated image (source + target) |
| **Output** | Patch classification (70×70 receptive field) |
| **Architecture** | 4 convolutional blocks + 1 output block |
| **Normalization** | InstanceNorm + Spectral Normalization |
| **Activation** | LeakyReLU (α=0.2) |
| **Patch Size** | 70×70 (covers local region, not global) |
| **Parameters** | ~1.8 Million |
| **Advantage** | Fewer parameters, focuses on local realism |

### Why Patch-Based?

1. **Efficiency**: Calculates classification on overlapping patches
2. **Local Focus**: Encourages realistic textures and details
3. **Speed**: Faster than full-image discriminator
4. **Scalability**: Works with variable input sizes

**Example Output:**
```
Input 256×256 image → Output 16×16 patch predictions
Each output neuron classifies a 70×70 region
```

---

## Training Framework

### Training Algorithm

```
for epoch in range(num_epochs):
    for batch in train_loader:
        source, target = batch
        
        # ===== Train Discriminator =====
        # Generate fake images
        generated = generator(source)
        
        # Get predictions
        real_pair = concat(source, target)
        fake_pair = concat(source, generated)
        
        D_real = discriminator(real_pair)
        D_fake = discriminator(fake_pair)
        
        # Discriminator loss
        loss_D = BCE(D_real, 1) + BCE(D_fake, 0)
        loss_D.backward()
        optimizer_D.step()
        
        # ===== Train Generator =====
        # Generate new batch
        generated = generator(source)
        fake_pair = concat(source, generated)
        
        # Get discriminator output
        D_fake = discriminator(fake_pair)
        
        # Generator loss
        loss_gan = BCE(D_fake, 1)
        loss_L1 = L1(generated, target)
        loss_G = loss_gan + lambda_L1 * loss_L1
        
        loss_G.backward()
        optimizer_G.step()
```

### Key Training Components

**1. Image Buffer**
- Stores previously generated images
- Discriminator trains on mix of current and old samples
- Prevents overfitting to latest generator

**2. Batch Normalization Issues**
- Batch stats differ for generated vs. real in small batches
- Solution: Use InstanceNorm in generator
- Helps generator generalize better

**3. Learning Rate**
- Generator: 2e-4
- Discriminator: 2e-4
- Adam optimizer with β₁=0.5, β₂=0.999

**4. Loss Weighting**
- λ_gan = 1.0 (adversarial loss weight)
- λ_L1 = 100.0 (reconstruction loss weight)
- Reconstruction loss dominates to prevent artifact generation

---

## Loss Functions

### 1. Adversarial Loss (GAN Loss)

**For Discriminator:**
```
L_D = -E[log D(x, y)] - E[log(1 - D(x, G(x)))]
    = BinaryCrossEntropy(D(real), 1) + BinaryCrossEntropy(D(fake), 0)
```

**For Generator:**
```
L_gan = -E[log D(x, G(x))]
      = BinaryCrossEntropy(D(fake), 1)
```

### 2. L1 Reconstruction Loss

```
L_L1 = E[||y - G(x)||_1]
     = (1/N) Σ |y_i - G(x_i)|
```

**Why L1 instead of L2?**
- L1 less prone to blurry outputs
- Encourages sharper, more detailed images
- Better preserves high-frequency components

### 3. Combined Generator Loss

```
L_G = L_gan + λ_L1 * L_L1
    = BinaryCrossEntropy(D(fake), 1) + 100 * L1(generated, target)

where λ_L1 = 100 (strong weight on reconstruction)
```

### 4. Optional: Perceptual Loss

```
L_perceptual = Σ_i ||Φ_i(y) - Φ_i(G(x))||_2

where Φ_i = features from layer i of pre-trained VGG16
```

---

## Implementation Details

### Data Preprocessing

```python
# Normalization (from [0, 1] to [-1, 1])
normalized = image * 2 - 1

# Augmentation (only during training)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(10°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
```

### Forward Pass Example

```python
# Training step
source = batch['source']  # [B, 3, 256, 256]
target = batch['target']  # [B, 3, 256, 256]

# Generator forward
generated = generator(source)  # [B, 3, 256, 256]

# Concatenate for discriminator
real_pair = torch.cat([source, target], dim=1)  # [B, 6, 256, 256]
fake_pair = torch.cat([source, generated], dim=1)  # [B, 6, 256, 256]

# Discriminator forward
D_real = discriminator(real_pair)  # [B, 1, 16, 16]
D_fake = discriminator(fake_pair)  # [B, 1, 16, 16]
```

### Stability Techniques

1. **Spectral Normalization**: Constrains discriminator weights
2. **Instance Normalization**: Reduces batch dependency
3. **Image Buffer**: Prevents discriminator overfitting
4. **Large L1 Weight**: Provides pixel-level supervision
5. **Slow Learning Rate**: Prevents oscillation

### Computational Requirements

```
GPU Memory: 4-6 GB (batch_size=1, 256×256 images)
Training Time: ~37 hours per 200 epochs (single GPU)
Inference Time: ~0.28s per 256×256 image

Optimization Tips:
- Reduce image size: 256→128 (4x speedup)
- Use BatchNorm instead of InstanceNorm
- Enable mixed precision training
- Use distributed training for multiple GPUs
```

---

## Performance Characteristics

### Generator Behavior Over Time

```
Epoch 1-50:    Blurry outputs, learning basic structure
Epoch 51-100:  Improving details, better colors
Epoch 101-200: Sharp outputs, realistic textures
Epoch 201+:    Minor improvements, risk of overfitting
```

### Discriminator vs Generator Power

```
Early training: D dominates, G learns basic features
Middle training: Balanced competition
Late training: G catches up, both improve together
Plateau: Equilibrium reached (Nash equilibrium)
```

### Key Metrics During Training

```
Generator Loss:      2.0 → 0.8 (should decrease)
Discriminator Loss:  0.6 → 0.3 (should stabilize)
L1 Loss:            0.05 → 0.04 (pixel-level error decreases)
FID Score:          50 → 25 (distribution improves)
```

---

## Comparison with Other Methods

### Pix2Pix vs CycleGAN

| Aspect | Pix2Pix | CycleGAN |
|--------|---------|----------|
| **Paired Data Required** | ✓ Yes | ✗ No |
| **Translation Quality** | ✓✓✓✓✓ Excellent | ✓✓✓✓ Good |
| **Training Stability** | ✓✓✓✓ High | ✓✓✓ Medium |
| **FID Score** | 20-35 | 35-50 |
| **Perceptual Quality** | ✓ Sharp | ✓ May be blurry |
| **Artifact-Free** | ✓ Usually | ✗ Mode collapse risk |

### Advantages of Pix2Pix

1. Leverages paired training data effectively
2. Produces sharp, detailed outputs
3. Fast and stable training
4. Straightforward architecture
5. Excellent for structured domains

### Limitations

1. Requires paired training data
2. May struggle with high-variance translations
3. Mode collapse possible without careful regularization
4. Limited to learned mappings (no creativity without diversity)

---

## Code Structure

```
models.py
├── ConvBlock
├── SpectralNormConv2d
├── UNetGenerator
│   ├── _Encoder (7 blocks)
│   ├── _Bottleneck
│   ├── _Decoder (7 blocks)
│   └── forward()
└── PatchGANDiscriminator
    ├── _Block1-4
    ├── _OutputLayer
    └── forward()

losses.py
├── GANLoss
├── L1Loss
├── L2Loss
├── PerceptualLoss
└── Pix2PixLoss
    ├── discriminator_loss()
    └── generator_loss()

train.py
├── Pix2PixTrainer
│   ├── train_epoch()
│   ├── validate()
│   └── train()
└── main()
```

---

## Conclusion

The Pix2Pix architecture is a powerful and practical approach to paired image translation:

- **Generator (U-Net)**: Encodes structural information with skip connections
- **Discriminator (PatchGAN)**: Focuses on local realism with patch-based classification
- **Training**: Adversarial framework + strong L1 supervision = high-quality results
- **Applications**: Real-world impact in autonomous driving, GIS, architecture, and design

The combination of adversarial learning with pixel-level supervision creates a flexible framework capable of learning diverse image translation tasks effectively.

---

*For more information, see the accompanying results.md and README.md files.*
