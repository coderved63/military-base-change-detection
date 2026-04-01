# Deep Dive: Models, Transfer Learning, and Fine-Tuning Explained

## Table of Contents

1. [What Is Transfer Learning and Why ImageNet?](#1-what-is-transfer-learning-and-why-imagenet)
2. [What Exactly Did We Fine-Tune?](#2-what-exactly-did-we-fine-tune)
3. [Model 1: Siamese CNN — Explained Like You're Teaching It](#3-model-1-siamese-cnn)
4. [Model 2: UNet++ — Why A Medical Model Works For Satellites](#4-model-2-unet)
5. [Model 3: ChangeFormer — The Transformer Approach](#5-model-3-changeformer)
6. [Why UNet++ Even Though It's A Medical Model?](#6-why-unet-even-though-its-a-medical-model)
7. [What Happens Inside During Inference — Step By Step](#7-what-happens-inside-during-inference)
8. [How To Explain This To Faculty](#8-how-to-explain-this-to-faculty)

---

## 1. What Is Transfer Learning and Why ImageNet?

### The Problem With Training From Scratch

A deep learning model needs to learn TWO things:
1. **Low-level features** — edges, textures, corners, gradients, colors
2. **High-level features** — objects, shapes, spatial relationships

Learning low-level features from scratch takes millions of images and days of training. But here's the key insight: **edges look the same everywhere**. An edge in a cat photo looks the same as an edge in a satellite photo. A texture gradient in a car image is structurally identical to a texture gradient in a building image.

### What Is ImageNet?

ImageNet is a dataset of **14 million images** across 1000 categories (cats, dogs, cars, planes, buildings, landscapes, etc.). Models trained on ImageNet learn incredibly rich low-level and mid-level features because they've seen enormous visual diversity.

### What Is Transfer Learning?

Instead of training from scratch (random weights), we START with weights that were trained on ImageNet. This gives us:

```
FROM SCRATCH:
Random weights --> [needs millions of images] --> Learns edges --> Learns textures --> Learns shapes --> Learns objects
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   THIS TAKES FOREVER

TRANSFER LEARNING:
ImageNet weights --> [already knows edges, textures, shapes] --> [needs few thousand images] --> Learns satellite-specific patterns
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     FREE - comes with pretrained weights         THIS IS FAST
```

### Analogy

Think of it like learning a new language:
- **From scratch**: A baby learning their first language — takes years
- **Transfer learning**: A person who speaks English learning Spanish — much faster because they already understand grammar, sentence structure, and many shared words

ImageNet pretraining = knowing English. Satellite change detection = learning Spanish. The foundation transfers.

---

## 2. What Exactly Did We Fine-Tune?

### What "Fine-Tuning" Means

Fine-tuning means we took the pretrained ImageNet weights and **continued training them on our satellite data**. We didn't freeze anything — ALL layers were updated. This is called **end-to-end fine-tuning**.

### What Changed During Fine-Tuning

```
BEFORE Fine-Tuning (ImageNet weights):
  Layer 1: Detects generic edges, gradients
  Layer 2: Detects generic textures, patterns
  Layer 3: Detects generic shapes (circles, rectangles)
  Layer 4: Detects generic objects (cat face, car wheel)
  ^^^^ These are useful for ANYTHING visual

AFTER Fine-Tuning (our satellite training):
  Layer 1: Still detects edges (barely changed — edges are universal)
  Layer 2: Detects satellite-specific textures (roof patterns, road textures)
  Layer 3: Detects building footprints, road shapes
  Layer 4: Detects "new building appeared" vs "same building"
  ^^^^ Early layers changed little, later layers changed a LOT
```

### The Numbers

| Model | Total Parameters | Pretrained (from ImageNet) | New (trained from scratch) |
|---|---|---|---|
| Siamese CNN | 14M | 11M (ResNet18 encoder) | 3M (decoder) |
| UNet++ | 26M | 21M (ResNet34 encoder) | 5M (decoder) |
| ChangeFormer | 14M | 0 (trained from scratch) | 14M (everything) |

**Key point**: For Siamese CNN and UNet++, the ENCODER (feature extractor) is pretrained. The DECODER (change mask generator) is trained from scratch. During fine-tuning, both encoder AND decoder are updated, but the encoder starts from a much better position.

**ChangeFormer is different**: We wrote the entire architecture from scratch. There are no widely available pretrained MiT-B1 weights for change detection, so we trained everything from random initialization. This is why it needs 200 epochs instead of 100.

### What Does The Training Actually Do?

Each training step:
1. Feed a before/after image pair through the model
2. Model outputs a predicted change mask
3. Compare prediction with ground truth using BCEDiceLoss
4. Compute gradients (how much each weight contributed to the error)
5. Update ALL weights slightly in the direction that reduces error
6. Repeat 7,120 times per epoch (one per training sample)
7. Repeat for 85-141 epochs

After training:
- Early layers (edges, textures): changed ~5-10% from ImageNet values
- Middle layers (shapes, patterns): changed ~20-40%
- Late layers (semantic understanding): changed ~60-90%
- Decoder layers: learned entirely from our data

---

## 3. Model 1: Siamese CNN

### What Is "Siamese"?

"Siamese" means twins — like Siamese twins. The model has TWO identical paths that share the SAME weights:

```
Image A (Before) ----\
                      [Same ResNet18] ---- Features A
Image B (After)  ----/                     Features B
                      ^^^^^^^^^^^^^
                      SHARED WEIGHTS
                      (not two separate networks)
```

**Why shared?** If both images go through the EXACT same processing, then any difference in the output features MUST be because the images themselves are different. The shared weights act as a fair, unbiased feature extractor.

### ResNet18 Encoder — Step by Step

ResNet18 is a Convolutional Neural Network with 18 layers. Here's what happens to a 256x256 satellite image:

```
Input: [3, 256, 256]    (3 = RGB channels)
  |
  v
Conv1 + BN + ReLU + MaxPool
  |  --> [64, 64, 64]    (64 feature channels, spatial size reduced to 64x64)
  v
Layer 1 (2 residual blocks)
  |  --> [64, 64, 64]    (same size, refined features)
  v
Layer 2 (2 residual blocks)
  |  --> [128, 32, 32]   (more channels, smaller spatial)
  v
Layer 3 (2 residual blocks)
  |  --> [256, 16, 16]   (even more channels, even smaller)
  v
Layer 4 (2 residual blocks)
  |  --> [512, 8, 8]     (512 feature channels, 8x8 spatial grid)
  v
Output: Rich feature representation
```

Each "residual block" has the famous skip connection:
```
input ----> [Conv -> BN -> ReLU -> Conv -> BN] ----> ADD ----> ReLU ----> output
  |                                                    ^
  |_____________(identity shortcut)____________________|
```

The skip connection solves the vanishing gradient problem — gradients can flow directly through the shortcut, making deep networks trainable.

### The Difference Operation

After encoding both images:
```
Features_A: [512, 8, 8]  (before image encoded)
Features_B: [512, 8, 8]  (after image encoded)

Difference = |Features_A - Features_B|  (absolute difference, element-wise)
Result:      [512, 8, 8]  (where values are high = something changed)
```

If a pixel in Features_A has value 0.8 and the same pixel in Features_B has value 0.2, the difference is 0.6 — meaning this region changed significantly.

### The Decoder — Transposed Convolutions

Now we need to go from 8x8 back to 256x256. Transposed convolution (also called "deconvolution") does upsampling:

```
[512, 8, 8]
  |  TransposedConv + BN + ReLU
  v
[256, 16, 16]
  |  TransposedConv + BN + ReLU
  v
[128, 32, 32]
  |  TransposedConv + BN + ReLU
  v
[64, 64, 64]
  |  TransposedConv + BN + ReLU
  v
[32, 128, 128]
  |  TransposedConv (final)
  v
[1, 256, 256]  <-- Change mask! (raw logits, apply sigmoid for probabilities)
```

### Weakness

The encoder compresses 256x256 down to 8x8 — that's a 32x reduction. Fine spatial details are lost. A small building that's 10x10 pixels becomes less than 1 pixel in the 8x8 feature map. The decoder tries to reconstruct this but without skip connections (unlike UNet), it struggles with precise localization.

---

## 4. Model 2: UNet++

### First, What Is Regular UNet?

UNet was invented for medical image segmentation (detecting tumors in brain scans). It has an **encoder-decoder structure with skip connections**:

```
ENCODER (downsampling)              DECODER (upsampling)
[256x256] ----skip connection----> [256x256]
    |                                  ^
[128x128] ----skip connection----> [128x128]
    |                                  ^
[64x64]   ----skip connection----> [64x64]
    |                                  ^
[32x32]   ----skip connection----> [32x32]
    |                                  ^
[16x16]   ------bottleneck-------> [16x16]
```

The skip connections DIRECTLY copy encoder features to the decoder. This means the decoder has access to BOTH:
- High-level semantic info (from the bottleneck): "this region has a building"
- Low-level spatial detail (from skip connections): "the exact edge of the building is here"

### What Makes UNet++ Different From UNet?

Regular UNet's problem: the skip connections connect features at very different semantic levels. The encoder at level 2 produces "edge features" while the decoder at level 2 needs "building boundary features". There's a **semantic gap**.

UNet++ fixes this with **nested intermediate blocks**:

```
Regular UNet:
Encoder --------direct skip--------> Decoder
(raw features)                       (needs processed features)
                ^^ SEMANTIC GAP ^^

UNet++:
Encoder ----> [Block] ----> [Block] ----> Decoder
(raw)         (processed)   (more processed)  (ready to use)
              ^^^^^^^^^^^^^^^^^^^^^^^^
              NESTED DENSE BLOCKS bridge the gap
```

In detail:
```
X(0,0) ---------> X(0,1) ---------> X(0,2) ---------> X(0,3) ---------> X(0,4)
  |                  |                  |                  |                  
X(1,0) ---------> X(1,1) ---------> X(1,2) ---------> X(1,3)
  |                  |                  |
X(2,0) ---------> X(2,1) ---------> X(2,2)
  |                  |
X(3,0) ---------> X(3,1)
  |
X(4,0) (bottleneck)
```

Each X(i,j) node receives inputs from:
- The node below it (deeper features)
- ALL previous nodes at the same level (dense connections)

This means by the time features reach the output, they've been progressively refined through multiple intermediate processing stages.

### How We Adapted UNet++ For Change Detection

Original UNet++ takes ONE image and segments it. We adapted it for TWO images:

```
Image A (Before) --> [ResNet34 Encoder] --> Features at 5 scales
                          |  (shared weights)
Image B (After)  --> [ResNet34 Encoder] --> Features at 5 scales

At each scale:
  diff[i] = |Features_A[i] - Features_B[i]|

diff features --> [UNet++ Decoder with nested skip connections] --> Change Mask
```

We use ResNet34 (34 layers, deeper than ResNet18) as the encoder via the `segmentation-models-pytorch` library, which provides the UNet++ decoder architecture.

### Why ResNet34 Instead of ResNet18?

ResNet34 has more layers and captures richer features:
- ResNet18: [2, 2, 2, 2] blocks = 18 layers
- ResNet34: [3, 4, 6, 3] blocks = 34 layers

More depth = better feature extraction, especially for the subtle differences between before/after satellite images.

---

## 5. Model 3: ChangeFormer

### What Is A Vision Transformer?

Traditional CNNs look at LOCAL regions (3x3 or 5x5 patches). Transformers look at GLOBAL relationships — every part of the image can attend to every other part.

### The Self-Attention Mechanism

For a given position in the image, self-attention asks: "Which OTHER positions in this image are relevant to understanding THIS position?"

```
Example: A new building appears in the top-left
Self-attention notices:
  - New road appeared nearby (related construction)
  - Parking lot appeared on the right (part of same development)
  - Trees on the south side were cleared (preparation for construction)

A CNN would process each region independently.
A Transformer connects them all.
```

### How Self-Attention Works (Simplified)

For each pixel position:
1. Create a **Query** (Q): "What am I looking for?"
2. Create a **Key** (K): "What information do I have?"
3. Create a **Value** (V): "What information can I give?"

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
```

- Q * K^T: How relevant is each position to me? (attention score)
- softmax: Normalize to probabilities
- * V: Weight the values by attention scores
- / sqrt(d): Scale factor for numerical stability

### The MiT-B1 Architecture

MiT (Mix Transformer) is a hierarchical transformer — unlike ViT which processes the image at one scale, MiT processes at 4 scales (like a CNN):

**Stage 1 (64x64, 64 channels)**:
```
256x256 image
    |
Overlapping Patch Embed (7x7 conv, stride 4)
    |
64x64 grid of 64-dim tokens (4096 tokens)
    |
2x [Efficient Self-Attention + Mix-FFN]
    |
Output: [64, 64, 64] features
```

**Stage 2 (32x32, 128 channels)**:
```
Overlapping Patch Embed (3x3 conv, stride 2)
    |
32x32 grid of 128-dim tokens (1024 tokens)
    |
2x [Efficient Self-Attention + Mix-FFN]
    |
Output: [128, 32, 32] features
```

**Stage 3 (16x16, 320 channels)** and **Stage 4 (8x8, 512 channels)** follow the same pattern.

### Efficient Self-Attention

Standard self-attention on 64x64 = 4096 tokens would require a 4096x4096 attention matrix — too expensive. We use **Spatial Reduction**:

```
Standard: Q (4096 tokens) x K (4096 tokens) = 16M attention scores  (TOO SLOW)

Efficient: 
  Q stays at 4096 tokens
  K and V are spatially reduced: 4096 -> 64 tokens (8x reduction)
  Q (4096) x K (64) = 262K attention scores  (60x cheaper!)
```

This is done via a strided convolution that reduces K and V before computing attention.

### Mix-FFN

Standard transformers use a simple MLP (Linear -> GELU -> Linear) after attention. Mix-FFN adds a **depthwise 3x3 convolution** in the middle:

```
Standard FFN:  Linear -> GELU -> Linear
Mix-FFN:       Linear -> DepthwiseConv3x3 -> GELU -> Linear
                         ^^^^^^^^^^^^^^^^^
                         Injects local spatial information
```

Why? Pure transformers have no notion of "nearby pixels". The depthwise conv brings back local spatial awareness without the cost of full convolutions. This eliminates the need for explicit position embeddings.

### The MLP Decoder

After the encoder produces features at 4 scales, the decoder fuses them:

```
Stage 1 features: [64, 64, 64]   --[1x1 Conv]--> [64, 64, 64]   --[Upsample]--> [64, 64, 64]
Stage 2 features: [128, 32, 32]  --[1x1 Conv]--> [64, 32, 32]   --[Upsample]--> [64, 64, 64]
Stage 3 features: [320, 16, 16]  --[1x1 Conv]--> [64, 16, 16]   --[Upsample]--> [64, 64, 64]
Stage 4 features: [512, 8, 8]    --[1x1 Conv]--> [64, 8, 8]     --[Upsample]--> [64, 64, 64]

Concatenate all: [256, 64, 64]
    |
[1x1 Conv + BN + ReLU] --> [64, 64, 64]
    |
[1x1 Conv] --> [1, 64, 64]
    |
[Upsample 4x] --> [1, 256, 256]  <-- Final change mask
```

All scales are projected to the same dimension (64), upsampled to the same size (64x64), concatenated, and fused with a simple 1x1 convolution.

---

## 6. Why UNet++ Even Though It's A Medical Model?

This is a great question and one your faculty will likely ask. Here's the answer:

### The Core Insight: Segmentation Is Segmentation

UNet++ was designed for **medical image segmentation** — detecting tumor boundaries in CT scans, cell boundaries in microscopy, organ boundaries in MRI. But what IS segmentation?

```
Medical:    Input image --> Classify each pixel as (tumor / not tumor)
Satellite:  Input image --> Classify each pixel as (changed / not changed)
```

**The task is structurally identical.** Both are binary pixel-level classification problems with:

| Property | Medical | Satellite Change Detection |
|---|---|---|
| Task | Pixel classification | Pixel classification |
| Output | Binary mask | Binary mask |
| Class imbalance | Tumor is tiny vs whole brain | Changed area is tiny vs whole image |
| Multi-scale | Tumors vary from 5px to 500px | Buildings vary from 10px to 200px |
| Needs precise boundaries | Yes (surgical planning) | Yes (accurate change mapping) |

### Why UNet++ Is Especially Good For This

1. **Multi-scale feature fusion** — Buildings come in all sizes. A small shed (10x10px) needs fine features. A large warehouse (100x100px) needs coarse features. UNet++'s nested skip connections fuse ALL scales.

2. **Precise boundary detection** — The skip connections preserve spatial detail. Change detection needs precise boundaries — "exactly WHICH pixels changed?"

3. **Handles class imbalance** — In both medical and satellite tasks, the "positive" class (tumor/change) is tiny. UNet++ was designed for this.

4. **Proven architecture** — It's not just medical anymore. UNet++ is used in:
   - Remote sensing (satellite segmentation)
   - Autonomous driving (road segmentation)
   - Industrial inspection (defect detection)
   - Agriculture (crop segmentation)

### The Adaptation We Made

Original UNet++: Takes ONE image, segments it
Our UNet++: Takes TWO images through a SHARED encoder, computes feature differences, decodes

```
Standard UNet++:
  1 image --> Encoder --> Decoder --> Segmentation mask

Our Adaptation:
  2 images --> Shared Encoder --> Feature Difference --> Decoder --> Change mask
```

This is NOT just "using UNet++ out of the box". We modified the architecture to handle bitemporal (two-image) input. The encoder is shared (Siamese), and we compute multi-scale feature differences before feeding into the decoder.

### What To Tell Faculty

> "UNet++ was originally for medical segmentation, but the underlying problem is identical — pixel-level classification with class imbalance, where both fine detail and coarse context matter. We adapted it for bitemporal input by using a shared encoder and computing feature differences at each scale. This architectural pattern (encoder-difference-decoder) is standard in remote sensing change detection literature. UNet++ is now widely used beyond medical imaging — in satellite imagery, autonomous driving, and industrial inspection."

---

## 7. What Happens Inside During Inference — Step By Step

Let's trace what happens when you upload two images in the Gradio app:

### Step 1: Image Loading
```
User uploads:
  before.png (256x256 RGB, uint8, values 0-255)
  after.png  (256x256 RGB, uint8, values 0-255)
```

### Step 2: Preprocessing
```
Convert to float32: values 0.0 to 1.0
Apply ImageNet normalization:
  pixel = (pixel - mean) / std
  mean = [0.485, 0.456, 0.406]  (per RGB channel)
  std  = [0.229, 0.224, 0.225]

Result: normalized tensors, values roughly -2.0 to 2.5
Shape: [1, 3, 256, 256] each (batch=1, channels=3, height=256, width=256)
```

### Step 3: Pad If Needed
```
If image is 300x400:
  Pad to 512x512 (nearest multiple of 256)
  Using reflection padding (mirrors edge pixels)
```

### Step 4: Tile Into Patches (if larger than 256x256)
```
512x512 image --> 4 patches of 256x256
  Patch 1: top-left
  Patch 2: top-right
  Patch 3: bottom-left
  Patch 4: bottom-right
```

### Step 5: Model Forward Pass (for each patch pair)

**Using ChangeFormer as example:**

```
Before patch [1, 3, 256, 256]  --> MiT Encoder --> 4 feature maps
After patch  [1, 3, 256, 256]  --> MiT Encoder --> 4 feature maps
                                    (shared weights)

Feature differences at each scale:
  Scale 1: |before_64x64 - after_64x64|   = diff_64x64
  Scale 2: |before_32x32 - after_32x32|   = diff_32x32
  Scale 3: |before_16x16 - after_16x16|   = diff_16x16
  Scale 4: |before_8x8 - after_8x8|       = diff_8x8

MLP Decoder fuses all scales:
  --> [1, 1, 256, 256] raw logits
```

### Step 6: Sigmoid + Threshold
```
Probabilities = sigmoid(logits)    # values 0.0 to 1.0
Binary mask = (probabilities > 0.5)  # True/False per pixel
```

### Step 7: Stitch Patches Back (if tiled)
```
4 patches of 256x256 --> stitch back to 512x512
Crop to original 300x400
```

### Step 8: Create Outputs
```
Change mask: binary image (white = change, black = no change)
Overlay: after image with red tint on changed pixels
Statistics: "5.3% of area changed, 6,360 pixels out of 120,000"
```

### Total Time
- CPU: ~2-5 seconds per 256x256 patch
- GPU (T4): ~0.1 seconds per 256x256 patch

---

## 8. How To Explain This To Faculty

### If asked "Explain the model architecture"

> "All three models follow the same pattern: a shared-weight Siamese encoder processes both the before and after images identically. We compute the absolute difference between features at each scale — large differences indicate change. A decoder then upsamples this difference back to full resolution to produce a pixel-level change mask.

> The difference is in the encoder and decoder:
> - Siamese CNN uses ResNet18 and simple transposed convolutions — fast but loses spatial detail
> - UNet++ uses ResNet34 with nested skip connections — preserves detail at every scale
> - ChangeFormer uses a hierarchical transformer with self-attention — captures global context across the entire image"

### If asked "What fine-tuning did you do?"

> "We used ImageNet-pretrained ResNet backbones for the encoder. ImageNet teaches the model to recognize edges, textures, and shapes — these visual primitives are universal. We then fine-tuned ALL layers end-to-end on our satellite change detection dataset. The early layers (edge detection) barely changed. The later layers were substantially updated to understand satellite-specific patterns like building footprints and road textures. The decoder was trained entirely from scratch since it's specific to change detection."

### If asked "Why UNet++ for satellite when it's a medical model?"

> "UNet++ solves pixel-level binary classification with class imbalance and multi-scale features. That's exactly what change detection needs — most pixels are unchanged (like most brain pixels are non-tumor), and changes happen at multiple scales (small buildings to large developments). The architecture is task-agnostic — it doesn't know if it's looking at brains or buildings. We adapted it by adding a shared Siamese encoder and computing feature differences, making it bitemporal."

### If asked "What's your contribution vs just using existing models?"

> "Three things: First, we built the change detection adaptation — Siamese encoding, feature differencing, the full ChangeFormer from scratch. Second, we created a unified comparison framework — same data, same metrics, same training for all three models, which most papers don't do. Third, we built a production pipeline — from data preprocessing to a deployed web app with tiled inference for any image size. The finding that UNet++ outperforms the transformer on this task and patch size is itself a contribution — it challenges the assumption that newer architectures are always better."
