"""
🧠 Neuro Brain — Training Pipeline
Unified training script for all three neural network types.

Training Parameters Explained (in-code):
─────────────────────────────────────────────────────────────
  LOSS FUNCTION    How wrong is the prediction? Model tries to minimize this.
  OPTIMIZER        How to adjust weights to reduce loss. Adam is the best default.
  EPOCHS           Full passes through the entire training dataset.
  BATCH SIZE       Samples processed before each weight update.
  VALIDATION SPLIT Fraction of training data held out to monitor overfitting.
  LEARNING RATE    Step size for weight updates. Too high = unstable, too low = slow.
─────────────────────────────────────────────────────────────

Usage:
    python -m src.train --task ids        # Train the NIDS Model
    python -m src.train --task waf        # Train the WAF Model
    python -m src.train --task forensics  # Train the Memory Forensics Model
    python -m src.train --task all        # Train all SOC models
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  TRAINER 1: Number Prediction (TensorFlow/Keras)
# ═══════════════════════════════════════════════════════════════

def train_number_predictor():
    """
    Train a Feedforward Neural Network for number prediction.

    PARAMETER BREAKDOWN:
    ─────────────────────────────────────────────────────────────
    • loss='mse' (Mean Squared Error)
      ┌──────────────────────────────────────────────────────┐
      │ MSE = average of (predicted - actual)²               │
      │ Example: predicted=105, actual=100 → error = 25      │
      │ Why? Squaring penalizes BIG errors more → the model  │
      │ focuses on fixing its worst predictions first.        │
      │ When? Predicting continuous numbers (regression).     │
      └──────────────────────────────────────────────────────┘

    • optimizer=Adam(lr=0.001)
      ┌──────────────────────────────────────────────────────┐
      │ Adam = Adaptive Moment Estimation                     │
      │ It adapts the learning rate for EACH weight           │
      │ individually. Weights that need big updates get big   │
      │ steps. Weights that are almost right get tiny steps.  │
      │ lr=0.001 is the default — works for 90%+ of tasks.   │
      │ Why? Fastest convergence, least tuning needed.        │
      └──────────────────────────────────────────────────────┘

    • epochs=50
      ┌──────────────────────────────────────────────────────┐
      │ 1 epoch = model sees ALL training samples once.       │
      │ 50 epochs = model sees each sample 50 times.          │
      │ Too few → underfitting (hasn't learned enough).       │
      │ Too many → overfitting (memorizes instead of learns). │
      │ With EarlyStopping, we set a high number and let      │
      │ the callback decide when to stop.                     │
      └──────────────────────────────────────────────────────┘

    • batch_size=32
      ┌──────────────────────────────────────────────────────┐
      │ 32 samples processed → compute avg loss → update      │
      │ weights. Then next 32 samples, and so on.             │
      │ Smaller batch = noisier but covers more of the data   │
      │   landscape (can escape local minima).                │
      │ Larger batch = smoother loss curve, faster per epoch,  │
      │   but needs more memory and may converge to poorer    │
      │   solutions.                                          │
      │ 32 is the sweet spot for most tasks.                  │
      └──────────────────────────────────────────────────────┘

    • validation_split=0.2
      ┌──────────────────────────────────────────────────────┐
      │ Holds 20% of training data for validation.            │
      │ The model NEVER trains on this data — only evaluates. │
      │ If train_loss ↓ but val_loss ↑ → OVERFITTING.         │
      │ EarlyStopping watches val_loss to stop at the         │
      │ right moment.                                         │
      └──────────────────────────────────────────────────────┘
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("\n[!] Error: TensorFlow/Keras not found. Skipping Number Predictor training.")
        print("    Please install with: pip install tensorflow")
        return
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("  🔢 TRAINING: Number Predictor (FNN)")
    print("=" * 60)

    # ── 1. Generate Sample Dataset ─────────────────────────────
    # (Replace this section with your own data loading)
    print("\n📦 Step 1: Preparing dataset...")
    np.random.seed(42)
    n_samples = 5000

    # Simulating house price prediction
    square_feet = np.random.randint(500, 5000, n_samples).astype(float)
    bedrooms = np.random.randint(1, 6, n_samples).astype(float)
    age = np.random.randint(0, 50, n_samples).astype(float)
    distance = np.random.uniform(0.5, 30, n_samples)

    X = np.column_stack([square_feet, bedrooms, age, distance])
    y = (square_feet * 150 + bedrooms * 20000 - age * 1000 -
         distance * 5000 + np.random.randn(n_samples) * 10000)

    print(f"   Dataset: {n_samples} samples, {X.shape[1]} features")
    print(f"   Features: square_feet, bedrooms, age, distance_to_city")
    print(f"   Target: house price (${y.min():,.0f} — ${y.max():,.0f})")

    # ── 2. Preprocess + Split ──────────────────────────────────
    # IMPORTANT: fit scaler on train data ONLY, then transform both
    print("\n🔧 Step 2: Preprocessing (normalize + split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% held out for final testing
        random_state=42      # Reproducible results
    )

    scaler_X = StandardScaler()    # Standardize: mean=0, std=1
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)   # Fit on training data ONLY
    X_test = scaler_X.transform(X_test)          # Transform test with same params

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"   X range after scaling: [{X_train.min():.2f}, {X_train.max():.2f}]")

    # ── 3. Build Model ─────────────────────────────────────────
    print("\n🏗️ Step 3: Building model...")
    model = keras.Sequential([
        # Hidden layer 1: 128 neurons + ReLU activation
        keras.layers.Dense(128, activation='relu', input_shape=(4,),
                           name='hidden_1'),
        keras.layers.Dropout(0.2, name='dropout_1'),

        # Hidden layer 2: 64 neurons + ReLU activation
        keras.layers.Dense(64, activation='relu', name='hidden_2'),
        keras.layers.Dropout(0.2, name='dropout_2'),

        # Hidden layer 3: 32 neurons + ReLU activation
        keras.layers.Dense(32, activation='relu', name='hidden_3'),

        # Output layer: 1 neuron, linear (no activation) for regression
        keras.layers.Dense(1, name='output'),
    ], name='NeuroBrain_NumberPredictor')

    # ── 4. Compile: SET LOSS + OPTIMIZER ───────────────────────
    print("\n⚙️ Step 4: Configuring training parameters...")

    LOSS_FUNCTION = 'mse'          # Mean Squared Error — for regression
    OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)  # Adaptive optimizer
    METRICS = ['mae']              # Mean Absolute Error — human-readable metric

    model.compile(
        loss=LOSS_FUNCTION,        # HOW WRONG is the prediction?
        optimizer=OPTIMIZER,       # HOW to update weights
        metrics=METRICS            # WHAT to track during training
    )

    print(f"   Loss:       {LOSS_FUNCTION} (penalizes big errors exponentially)")
    print(f"   Optimizer:  Adam (lr=0.001, adaptive per-weight learning rates)")
    print(f"   Metric:     MAE (average dollar error in predictions)")

    model.summary()

    # ── 5. Train: SET EPOCHS, BATCH SIZE, VALIDATION ──────────
    print("\n🚀 Step 5: Training...")

    EPOCHS = 100               # Max passes (EarlyStopping may stop sooner)
    BATCH_SIZE = 32            # Samples per weight update
    VALIDATION_SPLIT = 0.2    # 20% of training data for validation

    print(f"   Epochs:     {EPOCHS} (with EarlyStopping, patience=10)")
    print(f"   Batch size: {BATCH_SIZE} (32 samples → 1 weight update)")
    print(f"   Val split:  {VALIDATION_SPLIT} (20% = {int(len(X_train)*VALIDATION_SPLIT)} samples)")
    print(f"   Train on:   {int(len(X_train)*(1-VALIDATION_SPLIT))} samples per epoch")
    print(f"   Steps/epoch: {int(len(X_train)*(1-VALIDATION_SPLIT)) // BATCH_SIZE}\n")

    # Callbacks: auto-stop + reduce LR on plateau
    callbacks = [
        # EarlyStopping: stop training when val_loss stops improving
        # patience=10: wait 10 epochs of no improvement before stopping
        # restore_best_weights: go back to the model with lowest val_loss
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # ReduceLROnPlateau: if val_loss stalls, reduce learning rate
        # factor=0.5: multiply LR by 0.5 (halve it)
        # patience=5: wait 5 epochs before reducing
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,                       # How many full passes
        batch_size=BATCH_SIZE,               # Samples per update
        validation_split=VALIDATION_SPLIT,   # Hold 20% for validation
        callbacks=callbacks,                 # EarlyStopping + LR scheduler
        verbose=1
    )

    train_time = time.time() - start_time

    # ── 6. Evaluate ────────────────────────────────────────────
    print(f"\n📊 Step 6: Evaluation")
    print(f"   Training time: {train_time:.1f}s")
    print(f"   Epochs trained: {len(history.history['loss'])}/{EPOCHS}")

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test MSE:  {test_loss:.6f}")
    print(f"   Test MAE:  {test_mae:.6f}")

    # Convert back to actual dollar amounts
    test_preds = model.predict(X_test, verbose=0)
    test_preds_actual = scaler_y.inverse_transform(test_preds)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    mae_dollars = np.mean(np.abs(test_preds_actual - y_test_actual))
    print(f"   Actual MAE: ${mae_dollars:,.0f} average error per prediction")

    # ── 7. Save + Plot ─────────────────────────────────────────
    Path("data/models").mkdir(parents=True, exist_ok=True)
    model.save("data/models/number_predictor.keras")
    print(f"\n💾 Model saved to data/models/number_predictor.keras")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Loss (MSE) — Lower is Better', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_title('Error (MAE) — Lower is Better', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('🧠 Number Predictor — Training Progress', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/models/number_predictor_training.png', dpi=150, bbox_inches='tight')
    print(f"📈 Training plot saved to data/models/number_predictor_training.png")

    # Show sample predictions
    print(f"\n🎯 Sample Predictions (first 5):")
    print(f"   {'Predicted':>12}  {'Actual':>12}  {'Error':>10}")
    print(f"   {'─'*12}  {'─'*12}  {'─'*10}")
    for pred, actual in zip(test_preds_actual[:5], y_test_actual[:5]):
        error = abs(pred[0] - actual[0])
        print(f"   ${pred[0]:>10,.0f}  ${actual[0]:>10,.0f}  ${error:>8,.0f}")

    return model, history


# ═══════════════════════════════════════════════════════════════
#  TRAINER 2: Object Recognition (PyTorch)
# ═══════════════════════════════════════════════════════════════

def train_object_detector():
    """
    Train a CNN for image classification on CIFAR-10.

    PARAMETER BREAKDOWN:
    ─────────────────────────────────────────────────────────────
    • loss = CrossEntropyLoss()
      Combines Softmax + NegativeLogLikelihood.
      Penalizes confident wrong predictions SEVERELY.
      If model says "99% cat" but it's a dog → enormous loss.
      When? Multi-class classification (10 object classes).

    • optimizer = Adam(lr=0.001)
      Same as above — adaptive step sizes per weight.

    • epochs = 20
      Fewer epochs for images because:
      - Dataset is large (50,000 images)
      - Each epoch takes longer (image processing is compute-heavy)
      - CNN converges faster than FNN on visual tasks

    • batch_size = 64
      Larger than 32 because:
      - Image data benefits from batch statistics (BatchNorm)
      - GPU can process 64 images in parallel efficiently
      - Smoother gradient estimates for visual features

    • validation: 15% of training data
      Slightly less than 20% because CIFAR-10 already has a
      separate 10,000-image test set.
    ─────────────────────────────────────────────────────────────
    """
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("  👁️ TRAINING: Object Detector (CNN)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   Device: {device}")

    # ── 1. Load + Preprocess Data ──────────────────────────────
    print("\n📦 Step 1: Loading CIFAR-10 dataset...")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root='data/raw', train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='data/raw', train=False, download=True, transform=test_transform
    )

    # ── 2. Validation Split ────────────────────────────────────
    # Split 15% of training data for validation
    val_size = int(0.15 * len(full_trainset))   # 7,500 images
    train_size = len(full_trainset) - val_size   # 42,500 images
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    BATCH_SIZE = 64   # Process 64 images per weight update

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   Train: {train_size} images | Val: {val_size} images | Test: {len(testset)} images")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps per epoch: {len(train_loader)}")

    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # ── 3. Build Model ─────────────────────────────────────────
    print("\n🏗️ Step 2: Building CNN model...")
    from src.models.object_detector import NeuroBrainCNN
    model = NeuroBrainCNN(num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # ── 4. Configure Training ──────────────────────────────────
    print("\n⚙️ Step 3: Configuring training parameters...")

    # LOSS FUNCTION: CrossEntropyLoss
    # Combines Softmax + NegativeLogLikelihood in one step.
    # Perfect for multi-class classification (10 classes).
    # Penalizes confident wrong predictions exponentially.
    criterion = nn.CrossEntropyLoss()

    # OPTIMIZER: Adam with learning rate 0.001
    # Adapts learning rate per weight based on gradient history.
    LEARNING_RATE = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # LEARNING RATE SCHEDULER: reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    EPOCHS = 20  # Full passes through training data

    print(f"   Loss:       CrossEntropyLoss (Softmax + NLL for 10 classes)")
    print(f"   Optimizer:  Adam (lr={LEARNING_RATE})")
    print(f"   Epochs:     {EPOCHS}")
    print(f"   Scheduler:  ReduceLROnPlateau (halve LR after 3 stale epochs)")

    # ── 5. Training Loop ───────────────────────────────────────
    print(f"\n🚀 Step 4: Training...\n")
    start_time = time.time()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()  # Enable dropout, batchnorm in training mode
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # FORWARD PASS: input → model → predictions
            outputs = model(images)

            # COMPUTE LOSS: how wrong are predictions?
            loss = criterion(outputs, labels)

            # BACKWARD PASS: compute gradients (how each weight contributed to error)
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute new gradients

            # UPDATE WEIGHTS: optimizer adjusts weights using gradients
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})

        train_loss = running_loss / total
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ── Validate ──
        model.eval()  # Disable dropout, use running stats for batchnorm
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # No gradients needed for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Step the LR scheduler based on validation loss
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'data/models/object_detector_best.pth')

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:>2}/{EPOCHS} │ "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% │ "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% │ "
              f"LR: {current_lr:.6f}")

    train_time = time.time() - start_time

    # ── 6. Final Test Evaluation ───────────────────────────────
    print(f"\n📊 Step 5: Final evaluation on test set...")
    model.load_state_dict(torch.load('data/models/object_detector_best.pth',
                                      map_location=device))
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"   Training time: {train_time:.1f}s")
    print(f"   Best Val Acc:  {best_val_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")

    # ── 7. Plot Training Progress ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_title('Loss (CrossEntropy) — Lower is Better', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(val_accs, label='Val Accuracy', linewidth=2)
    axes[1].set_title('Accuracy — Higher is Better', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('🧠 Object Detector — Training Progress', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/models/object_detector_training.png', dpi=150, bbox_inches='tight')
    print(f"📈 Training plot saved to data/models/object_detector_training.png")

    return model


# ═══════════════════════════════════════════════════════════════
#  TRAINER 3: Text Classification (PyTorch + Transformers)
# ═══════════════════════════════════════════════════════════════

def train_text_processor():
    """
    Train/Fine-tune a Transformer for text sentiment classification.

    PARAMETER BREAKDOWN:
    ─────────────────────────────────────────────────────────────
    • loss = CrossEntropyLoss()
      Same as image classification — multi-class problem.
      Here: POSITIVE vs NEGATIVE sentiment (2 classes).

    • optimizer = AdamW(lr=2e-5, weight_decay=0.01)
      AdamW (NOT Adam) — the standard for Transformers.
      - weight_decay=0.01: gently shrinks large weights to
        prevent overfitting (L2 regularization done right).
      - lr=2e-5 (0.00002): much smaller than the usual 0.001
        because we're fine-tuning a pre-trained model. Large
        steps would destroy what the model already learned.

    • epochs = 3
      Very few! Pre-trained Transformers already know language.
      We just teach them our specific task. 3-5 epochs is enough.
      More would overfit on small datasets.

    • batch_size = 16
      Smaller than CNN (64) because Transformers use MUCH more
      memory. Each text sample is 128 tokens × 768 dimensions.
      16 is the max that fits on most GPUs.
    ─────────────────────────────────────────────────────────────
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("\n" + "=" * 60)
    print("  📝 TRAINING: Text Processor (Transformer)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   Device: {device}")

    # ── 1. Prepare Dataset ─────────────────────────────────────
    print("\n📦 Step 1: Preparing text dataset...")

    # Sample movie review dataset (replace with your own!)
    texts = [
        "This movie was absolutely fantastic! Best film of the year.",
        "Terrible waste of time. Poor acting and boring plot.",
        "A masterpiece of cinema. Every scene was breathtaking.",
        "I fell asleep halfway through. Extremely dull.",
        "Incredible performances by the entire cast. Loved it!",
        "The worst movie I have ever seen. Complete garbage.",
        "Beautiful cinematography and a moving storyline.",
        "Predictable plot with weak character development.",
        "An instant classic that will stand the test of time.",
        "Painful to watch. Save your money and skip this one.",
        "Heartwarming and inspirational. Brought tears to my eyes.",
        "So bad it's almost funny. The dialogue was cringe-worthy.",
        "A thrilling adventure from start to finish. Edge of my seat!",
        "Boring, slow, and completely pointless. What a letdown.",
        "Outstanding direction and a stellar script. Pure gold.",
        "Nothing original here. Just a rehash of better films.",
        "A feel-good movie that the whole family will enjoy.",
        "Overrated and overhyped. Expected much more from this director.",
        "The special effects were mind-blowing. A visual feast!",
        "Laughably bad acting ruins what could have been a decent story.",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    print(f"   Total samples: {len(texts)}")
    print(f"   Positive: {sum(labels)} | Negative: {len(labels) - sum(labels)}")

    # ── 2. Tokenize ────────────────────────────────────────────
    print("\n🔧 Step 2: Tokenizing with DistilBERT...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    MAX_LENGTH = 128   # Max tokens per text (longer texts are truncated)

    encodings = tokenizer(
        texts,
        padding='max_length',    # Pad shorter texts to max_length
        truncation=True,         # Truncate longer texts to max_length
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )

    print(f"   Token shape: {encodings['input_ids'].shape}")
    print(f"   Max length: {MAX_LENGTH} tokens")

    # ── 3. Split ───────────────────────────────────────────────
    val_size = max(2, int(0.2 * len(dataset)))  # At least 2 samples
    train_size = len(dataset) - val_size

    trainset, valset = random_split(dataset, [train_size, val_size])

    BATCH_SIZE = 8   # Small batch for Transformer + small dataset

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n   Train: {train_size} | Val: {val_size}")
    print(f"   Batch size: {BATCH_SIZE}")

    # ── 4. Load Pre-trained Model ──────────────────────────────
    print("\n🏗️ Step 3: Loading pre-trained DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2    # Binary classification: positive/negative
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,} (most are pre-trained!)")

    # ── 5. Configure Training ──────────────────────────────────
    print("\n⚙️ Step 4: Configuring training parameters...")

    # AdamW: Adam + weight decay (L2 regularization)
    # lr=2e-5: very small because we're fine-tuning, not training from scratch
    # weight_decay=0.01: gently shrink large weights to prevent overfitting
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    EPOCHS = 3  # Pre-trained models need very few epochs

    print(f"   Loss:         CrossEntropyLoss (binary sentiment)")
    print(f"   Optimizer:    AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"   Epochs:       {EPOCHS} (pre-trained model, few epochs needed)")
    print(f"   Batch size:   {BATCH_SIZE}")

    # ── 6. Training Loop ───────────────────────────────────────
    print(f"\n🚀 Step 5: Fine-tuning...\n")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        train_loss = total_loss / total
        train_acc = 100. * correct / total

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, batch_labels)
                val_loss += loss.item() * input_ids.size(0)
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_loss = val_loss / max(val_total, 1)
        val_acc = 100. * val_correct / max(val_total, 1)

        print(f"  Epoch {epoch+1}/{EPOCHS} │ "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% │ "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

    train_time = time.time() - start_time
    print(f"\n   Training time: {train_time:.1f}s")

    # ── 7. Test Predictions ────────────────────────────────────
    print(f"\n🎯 Step 6: Testing predictions...\n")
    model.eval()
    test_texts = [
        "This is the best thing I've ever experienced!",
        "Absolutely horrible. I want my money back.",
        "It was okay, nothing special really.",
    ]

    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True,
                          truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax().item()
            conf = probs.max().item()

        label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
        print(f"   \"{text}\"")
        print(f"   → {label} ({conf:.1%} confidence)\n")

    # Save model
    model.save_pretrained('data/models/text_processor')
    tokenizer.save_pretrained('data/models/text_processor')
    print(f"💾 Model saved to data/models/text_processor/")

    return model


# ═══════════════════════════════════════════════════════════════
#  MAIN: Run training from command line
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  TRAINER 7: IP Intelligence Profiler (TensorFlow)
# ═══════════════════════════════════════════════════════════════

def train_ip_profiler():
    """
    Train a neural network to predict IP risk scores based on tracking data using PyTorch.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.data_loader import TabularLoader
    from pathlib import Path
    import pandas as pd

    print("\n" + "=" * 60)
    print("  🌍 Training Stage: IP Intelligence Profiler (PyTorch)")
    print("=" * 60)

    csv_path = Path("data/logs/ip_queries.csv")
    
    # Generate synthetic data if log is empty/missing
    if not csv_path.exists() or csv_path.stat().st_size < 100:
        print("[*] Logs sparse. Generating tactical synthetic reconnaissance data...")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        countries = ["USA", "Germany", "China", "Russia", "UK", "Canada", "North Korea", "Brazil"]
        isps = ["Google", "Amazon", "DigitalOcean", "Linode", "Hetzner", "Anonymous VPN", "Local ISP"]
        rows = []
        for _ in range(500):
            country = np.random.choice(countries)
            isp = np.random.choice(isps)
            score = 10
            if country in ["China", "Russia", "North Korea"]: score += 40
            if isp in ["DigitalOcean", "Linode", "Hetzner"]: score += 30
            if "VPN" in isp: score += 50
            score = min(score + np.random.randint(-10, 10), 100)
            rows.append({"Country": country, "ISP": isp, "ThreatScore": score})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[+] Baseline data generated: {csv_path}")

    # Pipeline
    loader = TabularLoader().load(csv_path)
    loader.encode_categories()
    data = loader.split('ThreatScore', test_size=0.1)
    data = loader.normalize(data)

    # Convert to Tensors
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train']).view(-1, 1)

    # Model Definition
    class IPProfilerNet(nn.Module):
        def __init__(self, input_dim):
            super(IPProfilerNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        def forward(self, x):
            return self.network(x)

    model = IPProfilerNet(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("[*] Initiating PyTorch training loop...")
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/30], Loss: {loss.item():.4f}")

    # Save
    model_path = Path("data/models/ip_profiler.pth")
    torch.save(model.state_dict(), str(model_path))
    print(f"[+] IP Profiler model saved to: {model_path}")


# ═══════════════════════════════════════════════════════════════
#  TRAINER 8: Cyber Reasoning Instruction-Tuning (GPT-2)
# ═══════════════════════════════════════════════════════════════

def train_cyber_reasoner():
    """
    Fine-tune the conversational engine on cybersecurity instructions.
    Uses causal language modeling (GPT-2).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("\n[!] Error: Transformers/Torch not found. Skipping Cyber Reasoning training.")
        return

    import json
    from pathlib import Path

    print("\n" + "=" * 60)
    print("  🧠 Training Stage: Cyber Reasoning Instruction-Tuning")
    print("=" * 60)

    jsonl_path = Path("data/training/cyber_instructions.jsonl")
    if not jsonl_path.exists():
        print(f"[!] Error: Training data not found at {jsonl_path}")
        return

    # 1. Custom Dataset for JSONL (Prompt + Completion)
    class CyberDataset(Dataset):
        def __init__(self, path, tokenizer, max_length=256):
            self.tokenizer = tokenizer
            self.samples = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    # Simple prompt formatting
                    text = f"USER: {entry['prompt']}\nAI: {entry['completion']}<|endoftext|>"
                    self.samples.append(text)
            
            self.encodings = tokenizer(
                self.samples, padding=True, truncation=True, 
                max_length=max_length, return_tensors="pt"
            )

        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

    print("[*] Loading base model (GPT-2) and tokenizer...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # GPT-2 doesn't have a pad token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset = CyberDataset(jsonl_path, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    print(f"[*] Starting fine-tuning on {len(dataset)} instruction pairs...")
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids'] # Causal LM labels are the same as inputs
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/3, Loss: {total_loss/len(loader):.4f}")

    # Save using ModelManager for standardized registry
    from src.model_manager import ModelManager
    manager = ModelManager()
    manager.save_cyber_reasoner(model, tokenizer)
    print(f"\n✅ Cyber Reasoner fine-tuning complete and registered.")


def main():
    parser = argparse.ArgumentParser(description='🧠 Neuro Brain — Training Pipeline')
    parser.add_argument(
        '--task',
        choices=['numbers', 'images', 'text', 'ids', 'waf', 'forensics', 'ip', 'logic', 'all'],
        default='all',
        help='Which model to train: numbers, ids, waf, forensics, images, text, ip, logic, or all'
    )
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════╗
    ║        🧠 NEURO BRAIN — AI COACH v1.0            ║
    ╚══════════════════════════════════════════════════╝
    """)

    Path("data/models").mkdir(parents=True, exist_ok=True)

    # Core Trainers
    from src.models.intrusion_detector import train_ids_model
    from src.models.waf_ai import train_waf_model
    from src.models.memory_forensics import run_memory_forensics as train_memory_forensics

    if args.task in ['numbers', 'all']:
        train_number_predictor()

    if args.task in ['ids', 'all']:
        train_ids_model()

    if args.task in ['waf', 'all']:
        train_waf_model()

    if args.task in ['forensics', 'all']:
        train_memory_forensics()

    if args.task in ['images', 'all']:
        train_object_detector()

    if args.task in ['text', 'all']:
        train_text_processor()

    if args.task in ['ip', 'all']:
        train_ip_profiler()

    if args.task in ['logic', 'all']:
        train_cyber_reasoner()

    print("\n" + "=" * 60)
    print("  ✅ Training complete! SOC registry updated.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
