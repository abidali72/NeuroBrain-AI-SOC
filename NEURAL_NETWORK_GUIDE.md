# 🧠 Neuro Brain — Neural Network Architecture Guide

> Understand every layer, every activation function, and why each design choice matters.

---

## 📋 Table of Contents

1. [How a Neural Network Works](#1-how-a-neural-network-works)
2. [Activation Functions Explained](#2-activation-functions-explained)
3. [Network 1: Number Predictor (FNN)](#3-network-1-number-predictor-fnn)
4. [Network 2: Object Detector (CNN)](#4-network-2-object-detector-cnn)
5. [Network 3: Text Processor (Transformer)](#5-network-3-text-processor-transformer)
6. [Choosing the Right Activation](#6-choosing-the-right-activation)

---

## 1. How a Neural Network Works

A neural network is layers of connected "neurons" that transform input data step by step:

```
INPUT DATA                HIDDEN LAYERS               OUTPUT
                    (learn patterns)
 [features] ──→  [ Layer 1 ] ──→ [ Layer 2 ] ──→  [prediction]
                  64 neurons       32 neurons        1 neuron
                  + ReLU           + ReLU            + Linear
```

**Each neuron does 3 things:**
1. **Multiply** — Multiplies each input by a learned weight
2. **Sum** — Adds all weighted inputs together + a bias term
3. **Activate** — Passes the result through an activation function

```
             w₁
 x₁ ──────┐
           ├──→ Σ(wᵢxᵢ + b) ──→ f(z) ──→ output
 x₂ ──────┤     ↑ weighted       ↑ activation
           │     sum              function
 x₃ ──────┘
             w₃
```

**Without activation functions**, a neural network is just a linear equation — it can only draw straight lines. Activations add **non-linearity**, letting the network learn curves, boundaries, and complex patterns.

---

## 2. Activation Functions Explained

### ReLU (Rectified Linear Unit)

```
Output │        ╱
       │       ╱
       │      ╱
───────┼─────╱──────── Input
       │    0
       │
```

- **Formula:** `f(x) = max(0, x)`
- **Behavior:** If input is positive → pass it through. If negative → output 0.
- **Why use it:** Fast, simple, avoids the vanishing gradient problem. The go-to default for hidden layers.
- **Where:** Hidden layers of FNN, CNN, most modern networks.
- **Weakness:** "Dead neurons" — if a neuron always outputs 0, it stops learning.

---

### LeakyReLU

```
Output │        ╱
       │       ╱
       │      ╱
───────┼─────╱──────── Input
      ╱│    0
     ╱ │
```

- **Formula:** `f(x) = x if x > 0, else 0.01x`
- **Behavior:** Like ReLU but allows a small gradient (0.01) for negative values.
- **Why use it:** Prevents dead neurons. Slightly better than ReLU in some cases.
- **Where:** Hidden layers — drop-in replacement for ReLU.

---

### Sigmoid

```
Output │    ──────── 1.0
       │   ╱
       │  │
───────┼──│──────── Input
       │  │
       │   ╲
       │    ──────── 0.0
```

- **Formula:** `f(x) = 1 / (1 + e⁻ˣ)`
- **Behavior:** Squashes any value to range (0, 1).
- **Why use it:** Outputs a probability. Perfect for "yes/no" binary decisions.
- **Where:** Output layer for binary classification (spam/not spam, cat/dog).
- **Weakness:** Vanishing gradients in deep networks, slow convergence.

---

### Tanh (Hyperbolic Tangent)

```
Output │    ──────── +1.0
       │   ╱
       │  │
───────┼──│──────── Input
       │  │
       │   ╲
       │    ──────── -1.0
```

- **Formula:** `f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`
- **Behavior:** Like sigmoid but outputs range (-1, +1). Zero-centered.
- **Why use it:** Good for data with negative values, gates in LSTMs.
- **Where:** LSTM/GRU gates, some hidden layers.

---

### Softmax

```
Raw Scores:  [2.0, 1.0, 0.5]
     ↓ softmax
Probabilities: [0.59, 0.24, 0.17]  ← sum = 1.0
```

- **Formula:** `f(xᵢ) = eˣⁱ / Σ(eˣʲ)` for all classes j
- **Behavior:** Converts raw scores into probability distribution (all sum to 1.0).
- **Why use it:** Tells you "30% cat, 65% dog, 5% bird" — probabilities across multiple classes.
- **Where:** Output layer for multi-class classification.

---

### Linear (No Activation)

- **Formula:** `f(x) = x` (identity function)
- **Behavior:** Passes value through unchanged.
- **Why use it:** For regression — the output should be any real number, not bounded.
- **Where:** Output layer for number prediction (house prices, temperature, stock prices).

---

### GELU (Gaussian Error Linear Unit)

```
Output │        ╱
       │      ╱╱
       │    ╱╱
───────┼──╱╱─────── Input
       │╱╱
       │
```

- **Formula:** `f(x) = x · Φ(x)` where Φ is the Gaussian CDF
- **Behavior:** Smooth approximation of ReLU, slightly curves near 0.
- **Why use it:** Used by Transformers (BERT, GPT). Slightly better than ReLU for NLP.
- **Where:** Hidden layers of Transformer models.

---

## 3. Network 1: Number Predictor (FNN)

**Task:** Predict a number from input features (e.g., house price from size, bedrooms, age).

```
Architecture: Feedforward Neural Network (FNN)
Framework:    TensorFlow / Keras


INPUT LAYER          HIDDEN LAYERS              OUTPUT LAYER
(4 features)     (learn complex patterns)      (1 prediction)

 square_feet ─┐
              │   ┌──────────┐   ┌──────────┐   ┌──────────┐
 bedrooms ────┼──→│ Dense(64)│──→│ Dense(32)│──→│ Dense(1) │──→ price
              │   │  + ReLU  │   │  + ReLU  │   │ + Linear │
 age_years ───┤   │+Dropout  │   │+Dropout  │   │(no activ)│
              │   └──────────┘   └──────────┘   └──────────┘
 distance ────┘
                   Hidden 1        Hidden 2       Output
```

### Why these activations?

| Layer | Activation | Why |
|-------|-----------|-----|
| Hidden 1 (64 neurons) | **ReLU** | Fast, effective, learns non-linear patterns between features |
| Hidden 2 (32 neurons) | **ReLU** | Continues pattern extraction; fewer neurons = distills features |
| Output (1 neuron) | **Linear** (none) | Price can be any positive number — no bounding needed |

### Why NOT other activations here?
- ❌ **Sigmoid** at output → would clamp price to 0-1 range
- ❌ **Softmax** at output → that's for classification, not regression
- ❌ **Sigmoid** in hidden layers → causes vanishing gradients, slow training

See: [number_predictor.py](file:///C:/Users/HP/OneDrive/Desktop/brain/src/models/number_predictor.py)

---

## 4. Network 2: Object Detector (CNN)

**Task:** Classify images into categories (e.g., airplane, car, bird, cat...).

```
Architecture: Convolutional Neural Network (CNN)
Framework:    PyTorch

INPUT           FEATURE EXTRACTION                  CLASSIFICATION
(32×32 image)   (learn visual patterns)             (decide class)

                ┌────────────────────┐
                │ Conv2d(3→32)       │  Detect edges, colors
                │ + BatchNorm        │  Stabilize values
                │ + ReLU             │  Add non-linearity
 [32×32×3] ───→│ + MaxPool(2)       │  Shrink to 16×16
  RGB image     ├────────────────────┤
                │ Conv2d(32→64)      │  Detect shapes, textures
                │ + BatchNorm        │
                │ + ReLU             │
                │ + MaxPool(2)       │  Shrink to 8×8
                ├────────────────────┤
                │ Conv2d(64→128)     │  Detect object parts
                │ + BatchNorm        │
                │ + ReLU             │
                │ + MaxPool(2)       │  Shrink to 4×4
                ├────────────────────┤
                │ Flatten            │  4×4×128 = 2048 values
                │ Dense(256) + ReLU  │  Combine all features
                │ + Dropout(0.5)     │  Prevent overfitting
                │ Dense(10)          │  One score per class
                │ + Softmax          │  → probabilities
                └────────────────────┘
                                        [0.02, 0.01, 0.85, ...]
                                         airplane  car   bird
```

### Why these activations?

| Layer | Activation | Why |
|-------|-----------|-----|
| Conv blocks (×3) | **ReLU** | Standard for CNNs — fast, effective for detecting visual features. Each block detects progressively complex patterns. |
| Dense(256) | **ReLU** | Combines spatial features into abstract representations |
| Output Dense(10) | **Softmax** (via CrossEntropyLoss) | Converts 10 raw scores into probabilities summing to 1.0 |

### Supporting layers explained:

| Layer | Purpose |
|-------|---------|
| **BatchNorm** | Normalizes activations between layers → faster, more stable training |
| **MaxPool(2)** | Halves spatial dimensions → forces network to learn important features, reduces computation |
| **Dropout(0.5)** | Randomly zeros 50% of neurons during training → prevents overfitting |
| **Flatten** | Converts 3D feature maps (4×4×128) into 1D vector (2048) for classification |

See: [object_detector.py](file:///C:/Users/HP/OneDrive/Desktop/brain/src/models/object_detector.py)

---

## 5. Network 3: Text Processor (Transformer)

**Task:** Understand and process text (sentiment, generation, entities).

```
Architecture: Transformer (DistilBERT)
Framework:    Hugging Face

INPUT                  TRANSFORMER BLOCKS              OUTPUT
(text tokens)          (understand context)            (task result)

                     ┌──────────────────────┐
 "I love AI"         │  EMBEDDING           │  Words → 768-dim vectors
      ↓              │  + Position Encoding │  + position information
  [101, 1045,        ├──────────────────────┤
   2293, 9932,       │  SELF-ATTENTION      │  Each word looks at
   102]              │  (Multi-Head × 12)   │  every other word
  token IDs          │  + GELU activation   │  "love" ↔ "AI" context
                     ├──────────────────────┤
                     │  FEED-FORWARD        │  Process attention output
                     │  Dense(3072) + GELU  │  Expand, add non-linearity
                     │  Dense(768)          │  Compress back
                     │  × 6 layers          │  Stack for depth
                     ├──────────────────────┤
                     │  CLASSIFICATION HEAD │
                     │  Dense(768→2)        │  Map to task output
                     │  + Softmax           │  → probabilities
                     └──────────────────────┘
                                               [0.95, 0.05]
                                               POSITIVE  NEGATIVE
```

### Why these activations?

| Layer | Activation | Why |
|-------|-----------|-----|
| Feed-Forward layers | **GELU** | Smooth, probabilistic non-linearity. Outperforms ReLU in Transformers because it softly gates values near zero instead of hard-cutting. |
| Classification output | **Softmax** | Converts logits into probabilities for each class |
| Generation output | **Softmax** | Probability distribution over entire vocabulary for next word |

### Why GELU over ReLU in Transformers?
- ReLU has a hard cutoff at 0 → information is lost abruptly
- GELU has a smooth curve → small negative values are partially kept
- This matters for language because subtle word relationships need gradual activation

See: [text_processor.py](file:///C:/Users/HP/OneDrive/Desktop/brain/src/models/text_processor.py)

---

## 6. Choosing the Right Activation

### Quick Reference Table

| Situation | Use This | Reason |
|-----------|---------|--------|
| **Hidden layers** (default) | ReLU | Fast, works for most tasks |
| **Hidden layers** (dead neurons) | LeakyReLU | Allows small gradient for negatives |
| **Hidden layers** (Transformers) | GELU | Smooth gating, better for NLP |
| **Output: regression** | Linear (none) | Output can be any number |
| **Output: binary class** | Sigmoid | Output is probability 0-1 |
| **Output: multi-class** | Softmax | Outputs are probabilities summing to 1 |
| **LSTM/GRU gates** | Sigmoid + Tanh | Control memory flow |

### Decision Flowchart

```
What layer are you choosing for?
│
├── HIDDEN LAYER
│   ├── CNN or FNN? ──→ ReLU (or LeakyReLU)
│   └── Transformer? ──→ GELU
│
└── OUTPUT LAYER
    │
    ├── Predicting a number? ──→ Linear (no activation)
    ├── Yes/No classification? ──→ Sigmoid
    └── Multiple classes? ──→ Softmax
```

---

*Your neural network architectures are in `src/models/` — run them to see these layers in action! 🧠*
