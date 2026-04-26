# 🧠 Neuro Brain — Training Guide

> Understand every training parameter: loss functions, optimizers, epochs, batch size, and validation split.

---

## 📋 Table of Contents

1. [The Training Loop](#1-the-training-loop)
2. [Loss Functions](#2-loss-functions)
3. [Optimizers](#3-optimizers)
4. [Epochs](#4-epochs)
5. [Batch Size](#5-batch-size)
6. [Validation Split](#6-validation-split)
7. [Learning Rate](#7-learning-rate)
8. [Quick Reference](#8-quick-reference)

---

## 1. The Training Loop

Every neural network learns through the same cycle, repeated thousands of times:

```
┌──────────────────────────────────────────────────────────────┐
│                    ONE TRAINING STEP                          │
│                                                              │
│  1. FORWARD PASS                                             │
│     Feed data through the network → get prediction           │
│                                                              │
│  2. COMPUTE LOSS                                             │
│     Compare prediction to real answer → how wrong are we?    │
│                                                              │
│  3. BACKWARD PASS (Backpropagation)                          │
│     Calculate how each weight contributed to the error       │
│                                                              │
│  4. UPDATE WEIGHTS (Optimizer)                               │
│     Adjust weights to reduce the error                       │
│                                                              │
│  Repeat for every batch → that's one epoch                   │
│  Repeat for many epochs → model learns                       │
└──────────────────────────────────────────────────────────────┘
```

```python
# Pseudocode for every neural network ever trained:
for epoch in range(num_epochs):
    for batch_X, batch_y in data_loader:
        prediction = model(batch_X)           # 1. Forward pass
        loss = loss_function(prediction, batch_y)  # 2. Compute loss
        loss.backward()                       # 3. Backpropagation
        optimizer.step()                      # 4. Update weights
        optimizer.zero_grad()                 # Reset for next step
```

---

## 2. Loss Functions

The loss function measures **how wrong** the model's prediction is. The model's goal is to minimize this number.

### MSE (Mean Squared Error) — for Number Prediction

```
Formula: MSE = (1/n) × Σ(predicted - actual)²
```

```python
# Example: Predicting house prices
actual    = [200000, 350000, 150000]
predicted = [210000, 340000, 180000]

errors = [(210-200)², (340-350)², (180-150)²] = [100, 100, 900]
MSE = average(100, 100, 900) = 366.67
```

- **What it does:** Squares each error and averages them.
- **Why squared?** Big errors get punished exponentially more than small ones. An error of 30 is penalized 9× more than an error of 10.
- **When to use:** Regression tasks (predicting continuous numbers).

```python
# TensorFlow
model.compile(loss='mse')

# PyTorch
criterion = nn.MSELoss()
```

### MAE (Mean Absolute Error) — for Robust Number Prediction

```
Formula: MAE = (1/n) × Σ|predicted - actual|
```

- **What it does:** Takes the absolute error (no squaring).
- **vs MSE:** Less sensitive to outliers. If one house price is wildly off, MAE won't overreact.
- **When to use:** When outliers are expected in your data.

```python
# TensorFlow
model.compile(loss='mae')

# PyTorch
criterion = nn.L1Loss()
```

### Cross-Entropy Loss — for Classification

```
Formula: CE = -Σ(actual × log(predicted))
```

```python
# Example: Is this image a cat (1) or dog (0)?
actual      = 1 (cat)
predicted   = 0.9 (90% confident it's a cat)
loss        = -log(0.9) = 0.105  ← small loss, good prediction!

predicted   = 0.1 (10% confident it's a cat)
loss        = -log(0.1) = 2.302  ← BIG loss, bad prediction!
```

- **What it does:** Penalizes confident wrong predictions severely.
- **Why logarithm?** If the model says "99% cat" but it's actually a dog, the penalty is enormous — this forces the model to be calibrated.
- **When to use:** Classification (binary or multi-class).

```python
# TensorFlow — binary (2 classes)
model.compile(loss='binary_crossentropy')

# TensorFlow — multi-class (3+ classes)
model.compile(loss='sparse_categorical_crossentropy')

# PyTorch — handles both (includes Softmax internally)
criterion = nn.CrossEntropyLoss()
```

### Which loss to choose?

| Task | Loss Function | PyTorch | TensorFlow |
|------|--------------|---------|------------|
| Predict a number | **MSE** | `nn.MSELoss()` | `'mse'` |
| Predict a number (with outliers) | **MAE** | `nn.L1Loss()` | `'mae'` |
| Yes/No classification | **Binary CE** | `nn.BCEWithLogitsLoss()` | `'binary_crossentropy'` |
| Multiple classes | **Cross-Entropy** | `nn.CrossEntropyLoss()` | `'sparse_categorical_crossentropy'` |

---

## 3. Optimizers

The optimizer decides **how** to update the model's weights after seeing the error.

### SGD (Stochastic Gradient Descent)

```
Update rule: weight = weight - learning_rate × gradient
```

- **What it does:** Moves weights in the direction that reduces loss, by a fixed step size.
- **Analogy:** Walking downhill with fixed-size steps. Simple but can get stuck or overshoot.
- **When to use:** Large datasets, when you need stable convergence.

### Adam (Adaptive Moment Estimation) ⭐ RECOMMENDED

```
Update rule: weight = weight - adaptive_lr × gradient
              (adaptive_lr adjusts per-parameter based on history)
```

- **What it does:** Combines the best of two ideas:
  - **Momentum:** Remembers past gradient directions (like a rolling ball).
  - **Adaptive rates:** Each weight gets its own learning rate that adjusts based on how much it was updated in the past.
- **Why it's the best default:** Converges fast, handles noisy gradients, works on most problems without tuning.
- **When to use:** Almost always. Especially for:
  - Small to medium datasets
  - First attempt at training any model

### AdamW (Adam with Weight Decay)

- **What it does:** Adam + weight decay regularization (shrinks large weights).
- **When to use:** Transformer models (BERT, GPT), when you need to prevent overfitting.

### Comparison

| Optimizer | Speed | Stability | Tuning Needed | Best For |
|-----------|-------|-----------|--------------|----------|
| **SGD** | Slow start | Very stable | Lots (lr, momentum) | Large datasets, fine-tuning |
| **Adam** | Fast | Good | Minimal | General purpose, first try |
| **AdamW** | Fast | Great | Minimal | Transformers, regularized training |

```python
# TensorFlow
model.compile(optimizer='adam')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 4. Epochs

```
1 epoch = the model has seen every single training sample once
```

```
Dataset: 10,000 images
Batch size: 100

1 epoch = 10,000 ÷ 100 = 100 training steps
50 epochs = 5,000 training steps total
           (each image seen 50 times)
```

### How many epochs?

| Scenario | Epochs | Why |
|----------|--------|-----|
| Quick test | 5-10 | Just checking if the model structure works |
| Normal training | 20-50 | Enough for most tasks to converge |
| Complex tasks | 50-200 | Large datasets, difficult patterns |
| Fine-tuning pre-trained | 3-10 | Model already learned; just needs adjustment |

### The Danger: Overfitting

```
Training Loss ↓       Validation Loss
   ╲                       ╱╲
    ╲                     ╱  ╲  ← OVERFITTING starts here!
     ╲                   ╱    ╲    (val loss goes UP)
      ╲                 ╱      ╲
       ╲───────────────╱
        ╲             ╱
         ─────────────    ← UNDERFITTING
                              (still learning)
   Epoch 0              Epoch 50
```

- **Underfitting** (too few epochs): Model hasn't learned enough.
- **Good fit** (right number): Training and validation loss both decrease.
- **Overfitting** (too many epochs): Model memorizes training data, fails on new data.

### Solution: Early Stopping

```python
# TensorFlow — auto-stop when validation loss stops improving
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Watch validation loss
    patience=5,           # Wait 5 epochs before stopping
    restore_best_weights=True  # Go back to best model
)

model.fit(X_train, y_train, epochs=200, callbacks=[early_stop])
# Will stop early if val_loss doesn't improve for 5 epochs
```

---

## 5. Batch Size

```
Batch size = how many samples the model processes before updating weights
```

```
Dataset: 10,000 samples
Batch size: 32

Per epoch: 10,000 ÷ 32 = 312 weight updates (training steps)
Each step: model sees 32 samples → computes average loss → updates weights
```

### Visual Explanation

```
FULL DATASET (10,000 samples)
┌─────┬─────┬─────┬─────┬─────┬──── ─ ─ ──┬─────┐
│ B1  │ B2  │ B3  │ B4  │ B5  │    ...     │B312 │
│ 32  │ 32  │ 32  │ 32  │ 32  │            │ 32  │
└──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──── ─ ─ ──┴──┬──┘
   │     │     │     │     │                   │
   ▼     ▼     ▼     ▼     ▼                   ▼
 update update update update update          update
   1     2     3     4     5                  312

           ════════════════════════
           One complete epoch (312 updates)
```

### How to choose batch size?

| Batch Size | Effect | When to Use |
|-----------|--------|-------------|
| **1** | Very noisy updates, slow | Almost never (too slow) |
| **16** | Smoother, good for small datasets | < 5,000 samples |
| **32** ⭐ | Good default balance | Most tasks |
| **64** | Faster training per epoch | Medium datasets, GPUs |
| **128-256** | Very fast, needs more epochs | Large datasets + GPU |
| **Full dataset** | One update per epoch (batch GD) | Tiny datasets only |

### Memory Rule

```
Bigger batch = more GPU memory needed
If you get "CUDA out of memory" → reduce batch size
```

```python
# TensorFlow
model.fit(X_train, y_train, batch_size=32)

# PyTorch
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 6. Validation Split

```
Validation split = fraction of training data reserved for monitoring
```

```
TRAINING DATA (before validation split)
┌──────────────────────────────────────────────────────┐
│                     10,000 samples                    │
└──────────────────────────────────────────────────────┘
                    │
                    ▼ validation_split = 0.2
                    │
┌──────────────────────────────────────┬───────────────┐
│         TRAIN (8,000)                │  VAL (2,000)  │
│     Model learns from these          │ Monitor only  │
│     Weights ARE updated              │ NO updates    │
└──────────────────────────────────────┴───────────────┘
```

### Why validate?

Without validation, you only see training loss — which always decreases (the model can simply memorize). Validation loss tells you how the model performs on **unseen data** — the real test.

```
Epoch 10:  Train Loss = 0.05   Val Loss = 0.08   ← Good! Both low
Epoch 30:  Train Loss = 0.01   Val Loss = 0.06   ← Good! Both decreasing
Epoch 50:  Train Loss = 0.001  Val Loss = 0.15   ← BAD! Overfitting!
                 ↑                    ↑
           Still learning       Getting worse on new data
```

### How to choose validation split?

| Dataset Size | Recommended Split | Train | Val |
|-------------|-------------------|-------|-----|
| < 1,000 | 0.3 (30%) | 700 | 300 |
| 1,000 - 10,000 | 0.2 (20%) ⭐ | 8,000 | 2,000 |
| 10,000 - 100,000 | 0.15 (15%) | 85,000 | 15,000 |
| > 100,000 | 0.1 (10%) | 900,000 | 100,000 |

```python
# TensorFlow — automatic split
model.fit(X_train, y_train, validation_split=0.2)

# PyTorch — manual split
from torch.utils.data import random_split
train_set, val_set = random_split(dataset, [8000, 2000])
```

---

## 7. Learning Rate

```
Learning rate = how big of a step the optimizer takes toward the answer
```

```
                         Learning Rate Effect
 Loss
  │
  │  lr=0.1 (too high)
  │  ╱╲  ╱╲  ╱╲              Bounces around, never converges
  │ ╱  ╲╱  ╲╱  ╲╱╲
  │
  │  lr=0.001 (just right)
  │  ╲
  │   ╲                       Smooth descent to minimum
  │    ╲──────────
  │
  │  lr=0.00001 (too low)
  │  ╲
  │   ╲                       Takes forever, might get stuck
  │    ╲
  │     ╲
  │      ╲
  └──────────────────── Epochs
```

| Learning Rate | Effect |
|--------------|--------|
| **0.1** | Too high — loss bounces wildly |
| **0.01** | Good for SGD |
| **0.001** ⭐ | Best default for Adam |
| **0.0001** | Good for fine-tuning pre-trained models |
| **0.00001** | Very slow, use for final polish |

### Learning Rate Scheduling

Reduce learning rate as training progresses (take big steps early, small steps later):

```python
# TensorFlow
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3
)
model.fit(X, y, callbacks=[lr_scheduler])

# PyTorch
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
```

---

## 8. Quick Reference

### All Parameters at a Glance

| Parameter | Default | Meaning |
|-----------|---------|---------|
| **Loss (regression)** | `MSE` | How wrong is the predicted number |
| **Loss (classification)** | `CrossEntropy` | How wrong is the predicted class |
| **Optimizer** | `Adam` | How to update weights (adaptive step sizes) |
| **Learning Rate** | `0.001` | Step size per weight update |
| **Epochs** | `50` | How many full passes through the data |
| **Batch Size** | `32` | Samples per weight update |
| **Validation Split** | `0.2` | 20% of data reserved for monitoring |

### The "Just Works" Defaults

```python
# These settings work well for 90% of tasks:
optimizer     = Adam(lr=0.001)
loss          = MSE (regression) or CrossEntropy (classification)
epochs        = 50 + EarlyStopping(patience=5)
batch_size    = 32
val_split     = 0.2
```

---

*Now see `src/train.py` for the complete training pipeline code! 🧠*
