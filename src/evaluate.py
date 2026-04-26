"""
🧠 Neuro Brain — Evaluation Pipeline
Comprehensive model evaluation with all relevant metrics.

Metrics Included:
─────────────────────────────────────────────────────────────
  REGRESSION (Number Prediction):
    MSE   — Mean Squared Error (penalizes big errors)
    MAE   — Mean Absolute Error (average error in real units)
    RMSE  — Root Mean Squared Error (same units as target)
    R²    — How much variance the model explains (0-1, higher=better)

  CLASSIFICATION (Objects & Text):
    Accuracy   — What % of predictions are correct
    Precision  — Of predicted positives, how many are actually positive
    Recall     — Of actual positives, how many did we find
    F1 Score   — Balanced combination of precision and recall
    Confusion Matrix — Table showing exactly where errors occur
─────────────────────────────────────────────────────────────

Usage:
    python -m src.evaluate --task numbers
    python -m src.evaluate --task images
    python -m src.evaluate --task text
    python -m src.evaluate --task all
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure output directory exists
Path("data/models").mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  HELPER: Pretty metric printing
# ═══════════════════════════════════════════════════════════════

def print_metric(name, value, unit="", interpretation=""):
    """Print a metric with its interpretation."""
    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
    line = f"   {name:<22} {val_str:>10} {unit}"
    if interpretation:
        line += f"   ← {interpretation}"
    print(line)


def interpret_r2(r2):
    """Return a human-readable interpretation of R² score."""
    if r2 >= 0.90:   return "🟢 Excellent — model explains most variance"
    elif r2 >= 0.70: return "🟢 Good — strong predictive power"
    elif r2 >= 0.50: return "🟡 Moderate — captures some patterns"
    elif r2 >= 0.30: return "🟠 Weak — limited predictive power"
    else:            return "🔴 Poor — model is not useful"


def interpret_accuracy(acc):
    """Return interpretation of accuracy."""
    if acc >= 90:   return "🟢 Excellent"
    elif acc >= 80: return "🟢 Good"
    elif acc >= 70: return "🟡 Moderate"
    elif acc >= 60: return "🟠 Below average"
    else:           return "🔴 Poor — needs improvement"


def interpret_f1(f1):
    """Return interpretation of F1 score."""
    if f1 >= 0.90:   return "🟢 Excellent balance of precision & recall"
    elif f1 >= 0.80: return "🟢 Good"
    elif f1 >= 0.60: return "🟡 Moderate"
    elif f1 >= 0.40: return "🟠 Weak"
    else:            return "🔴 Poor — model struggles with this class"


# ═══════════════════════════════════════════════════════════════
#  EVALUATOR 1: Number Prediction (Regression)
# ═══════════════════════════════════════════════════════════════

def evaluate_number_predictor():
    """
    Evaluate the Number Predictor with regression metrics.

    METRICS EXPLAINED:
    ─────────────────────────────────────────────────────────────
    • MSE (Mean Squared Error)
      Formula: (1/n) × Σ(predicted - actual)²
      Interpretation: Average squared error. Lower = better.
      Why use: Standard loss function, penalizes large errors.
      Limitation: Units are squared (hard to interpret directly).

    • MAE (Mean Absolute Error)
      Formula: (1/n) × Σ|predicted - actual|
      Interpretation: "On average, predictions are off by $X."
      Why use: Easy to understand in real-world units.
      Example: MAE = $15,000 means average prediction error is $15K.

    • RMSE (Root Mean Squared Error)
      Formula: √MSE
      Interpretation: Like MAE but more sensitive to large errors.
      Same units as the target variable.

    • R² Score (Coefficient of Determination) ⭐
      Formula: 1 - (Σ(actual-predicted)² / Σ(actual-mean)²)
      Interpretation:
        R²=1.0  → Perfect predictions
        R²=0.0  → No better than always predicting the average
        R²<0.0  → Worse than guessing the average (bad model!)
      This is the BEST single metric for regression.

    • Residual Plot
      Scatter plot of (predicted - actual) vs predicted values.
      Ideal: random cloud centered at 0 (no pattern).
      Bad: funnel shape = errors grow with predictions.
      Bad: curve = model missing a non-linear pattern.
    ─────────────────────────────────────────────────────────────
    """
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("  🔢 EVALUATING: Number Predictor (Regression Metrics)")
    print("=" * 60)

    # ── 1. Load or generate test data ──────────────────────────
    print("\n📦 Preparing test data...")
    np.random.seed(42)
    n = 5000

    square_feet = np.random.randint(500, 5000, n).astype(float)
    bedrooms = np.random.randint(1, 6, n).astype(float)
    age = np.random.randint(0, 50, n).astype(float)
    distance = np.random.uniform(0.5, 30, n)

    X = np.column_stack([square_feet, bedrooms, age, distance])
    y = (square_feet * 150 + bedrooms * 20000 - age * 1000 -
         distance * 5000 + np.random.randn(n) * 10000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # ── 2. Load or train model ─────────────────────────────────
    model_path = Path("data/models/number_predictor.keras")
    if model_path.exists():
        print(f"   Loading model from {model_path}...")
        model = keras.models.load_model(str(model_path))
    else:
        print("   No saved model found. Training a new one...")
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(4,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train_s, y_train_s, epochs=50, batch_size=32,
                  validation_split=0.2, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(
                      patience=10, restore_best_weights=True)])
        model.save(str(model_path))

    # ── 3. Make predictions ────────────────────────────────────
    print("\n🔮 Making predictions on test set...")
    y_pred_s = model.predict(X_test_s, verbose=0).ravel()

    # Convert back to real scale
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_actual = y_test

    # ── 4. Compute ALL metrics ─────────────────────────────────
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    # Additional: percentage-based error
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100

    # ── 5. Display results ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📊 REGRESSION METRICS REPORT")
    print(f"{'─'*60}")
    print(f"   Test samples: {len(y_actual)}\n")

    print_metric("MSE", mse, "",
                 "avg squared error (lower=better)")
    print_metric("RMSE", rmse, "$",
                 "typical error magnitude")
    print_metric("MAE", mae, "$",
                 f"avg prediction off by ${mae:,.0f}")
    print_metric("MAPE", mape, "%",
                 f"avg {mape:.1f}% off from actual")
    print_metric("R² Score", r2, "",
                 interpret_r2(r2))

    print(f"\n{'─'*60}")
    print(f"  📖 HOW TO READ THESE RESULTS")
    print(f"{'─'*60}")
    print(f"""
   R² = {r2:.4f}
   → Your model explains {r2*100:.1f}% of the variance in prices.
   → {interpret_r2(r2)}

   MAE = ${mae:,.0f}
   → On average, predictions are off by ${mae:,.0f}.
   → For a $500,000 house, that's about {mae/500000*100:.1f}% error.

   RMSE = ${rmse:,.0f} vs MAE = ${mae:,.0f}
   → RMSE > MAE means some predictions have large errors.
   → Ratio: {rmse/mae:.2f}x (closer to 1.0 = errors are consistent).
""")

    # ── 6. Sample predictions ──────────────────────────────────
    print(f"  🎯 SAMPLE PREDICTIONS (first 10)")
    print(f"{'─'*60}")
    print(f"   {'Predicted':>12}  {'Actual':>12}  {'Error':>10}  {'Error%':>7}")
    print(f"   {'─'*12}  {'─'*12}  {'─'*10}  {'─'*7}")
    for pred, actual in zip(y_pred[:10], y_actual[:10]):
        error = pred - actual
        pct = abs(error) / abs(actual) * 100
        marker = " ✓" if pct < 10 else " ✗" if pct > 25 else ""
        print(f"   ${pred:>10,.0f}  ${actual:>10,.0f}  ${error:>+9,.0f}  {pct:>5.1f}%{marker}")

    # ── 7. Visualizations ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Predicted vs Actual
    axes[0].scatter(y_actual, y_pred, alpha=0.3, s=10, color='#4CAF50')
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                 label='Perfect prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=11)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=11)
    axes[0].set_title(f'Predicted vs Actual (R²={r2:.3f})', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Residual plot (errors)
    residuals = y_pred - y_actual
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color='#2196F3')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Price ($)', fontsize=11)
    axes[1].set_ylabel('Residual (Pred - Actual)', fontsize=11)
    axes[1].set_title('Residual Plot (should be random around 0)', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Error distribution
    axes[2].hist(residuals, bins=50, color='#FF9800', edgecolor='white', alpha=0.8)
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Prediction Error ($)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title(f'Error Distribution (MAE=${mae:,.0f})', fontsize=13)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('🧠 Number Predictor — Evaluation Report', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/models/number_predictor_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Evaluation plots saved to data/models/number_predictor_evaluation.png")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


# ═══════════════════════════════════════════════════════════════
#  EVALUATOR 2: Object Recognition (Classification)
# ═══════════════════════════════════════════════════════════════

def evaluate_object_detector():
    """
    Evaluate the Object Detector with classification metrics.

    METRICS EXPLAINED:
    ─────────────────────────────────────────────────────────────
    • Accuracy
      What % of ALL predictions are correct.
      Example: 85% → 85 out of 100 images classified correctly.
      ⚠️ Can be misleading with imbalanced classes.

    • Precision (per class)
      "Of images I CALLED 'cat', how many actually ARE cats?"
      High precision = few false positives (few wrong alarms).
      Example: Precision=0.95 for 'cat' → when model says cat,
               it's correct 95% of the time.

    • Recall (per class)
      "Of all ACTUAL cats, how many did I correctly find?"
      High recall = few false negatives (few missed cats).
      Example: Recall=0.80 for 'cat' → model finds 80% of cats,
               misses 20%.

    • F1 Score (per class)
      Harmonic mean of precision and recall → balanced view.
      F1 = 2 × (precision × recall) / (precision + recall)
      Best single metric when classes are imbalanced.

    • Confusion Matrix
      N×N grid showing predicted vs actual for every class.
      Diagonal = correct. Off-diagonal = specific error patterns.
      Example: high value at [cat, dog] means model often confuses
               cats for dogs → they need more distinguishing features.
    ─────────────────────────────────────────────────────────────
    """
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_recall_fscore_support
    )

    print("\n" + "=" * 60)
    print("  👁️ EVALUATING: Object Detector (Classification Metrics)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # ── 1. Load test data ──────────────────────────────────────
    print("\n📦 Loading CIFAR-10 test set...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    testset = torchvision.datasets.CIFAR10(
        root='data/raw', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    print(f"   Test set: {len(testset)} images, {len(CLASSES)} classes")

    # ── 2. Load or train model ─────────────────────────────────
    from src.models.object_detector import NeuroBrainCNN

    model = NeuroBrainCNN(num_classes=10).to(device)
    model_path = Path("data/models/object_detector_best.pth")
    alt_path = Path("data/models/object_detector.pth")

    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        print(f"   Loaded model from {model_path}")
    elif alt_path.exists():
        model.load_state_dict(torch.load(str(alt_path), map_location=device))
        print(f"   Loaded model from {alt_path}")
    else:
        print("   No saved model found. Training a quick model (5 epochs)...")
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='data/raw', train=True, download=True, transform=train_transform
        )
        train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(5):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            print(f"   Epoch {epoch+1}/5 complete")
        torch.save(model.state_dict(), str(alt_path))

    # ── 3. Collect all predictions ─────────────────────────────
    print("\n🔮 Running predictions on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── 4. Compute ALL metrics ─────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    cm = confusion_matrix(all_labels, all_preds)

    # ── 5. Display Overall Metrics ─────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📊 CLASSIFICATION METRICS REPORT")
    print(f"{'─'*60}")
    print(f"   Test samples: {len(all_labels)}\n")

    print_metric("Overall Accuracy", accuracy, "%", interpret_accuracy(accuracy))
    print_metric("Weighted Precision", precision_avg, "", "avg across all classes")
    print_metric("Weighted Recall", recall_avg, "", "avg across all classes")
    print_metric("Weighted F1 Score", f1_avg, "", interpret_f1(f1_avg))

    # ── 6. Per-Class Breakdown ─────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📋 PER-CLASS BREAKDOWN")
    print(f"{'─'*60}")
    print(f"   {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print(f"   {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for i, cls in enumerate(CLASSES):
        marker = " ⭐" if f1[i] >= 0.80 else " ⚠️" if f1[i] < 0.60 else ""
        print(f"   {cls:<12} {precision[i]:>10.3f} {recall[i]:>10.3f} "
              f"{f1[i]:>10.3f} {support[i]:>8d}{marker}")

    print(f"   {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    print(f"   {'WEIGHTED':<12} {precision_avg:>10.3f} {recall_avg:>10.3f} "
          f"{f1_avg:>10.3f} {len(all_labels):>8d}")

    # ── 7. Interpretation ──────────────────────────────────────
    best_class = CLASSES[np.argmax(f1)]
    worst_class = CLASSES[np.argmin(f1)]

    # Find most confused pair
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    max_confusion_idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    confused_actual = CLASSES[max_confusion_idx[0]]
    confused_pred = CLASSES[max_confusion_idx[1]]
    confused_count = cm_no_diag[max_confusion_idx]

    print(f"\n{'─'*60}")
    print(f"  📖 HOW TO READ THESE RESULTS")
    print(f"{'─'*60}")
    print(f"""
   Overall Accuracy: {accuracy:.1f}%
   → {accuracy:.0f} out of every 100 images are classified correctly.

   Best performing class: "{best_class}" (F1={f1[np.argmax(f1)]:.3f})
   → Model is most confident and accurate with this class.

   Worst performing class: "{worst_class}" (F1={f1[np.argmin(f1)]:.3f})
   → Consider: more training data, data augmentation, or
     checking if this class looks similar to others.

   Most confused pair: "{confused_actual}" ↔ "{confused_pred}"
     ({confused_count} images of "{confused_actual}" predicted as "{confused_pred}")
   → These classes look similar to the model. Consider:
     - More training images showing differences
     - Higher resolution input images
     - Fine-grained feature extraction
""")

    # ── 8. Confusion Matrix Visualization ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Confusion Matrix (counts)', fontsize=13)
    axes[0].set_xlabel('Predicted Class', fontsize=11)
    axes[0].set_ylabel('Actual Class', fontsize=11)
    axes[0].set_xticks(range(len(CLASSES)))
    axes[0].set_yticks(range(len(CLASSES)))
    axes[0].set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(CLASSES, fontsize=8)
    plt.colorbar(im, ax=axes[0])

    # Add text annotations
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                        color=color, fontsize=7)

    # Per-class F1 bar chart
    colors = ['#4CAF50' if f >= 0.80 else '#FF9800' if f >= 0.60 else '#f44336'
              for f in f1]
    bars = axes[1].bar(CLASSES, f1, color=colors, edgecolor='white')
    axes[1].set_title('F1 Score by Class', fontsize=13)
    axes[1].set_xlabel('Class', fontsize=11)
    axes[1].set_ylabel('F1 Score', fontsize=11)
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(y=0.80, color='green', linestyle='--', alpha=0.5, label='Good (0.80)')
    axes[1].axhline(y=0.60, color='orange', linestyle='--', alpha=0.5, label='Fair (0.60)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for bar, val in zip(bars, f1):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('🧠 Object Detector — Evaluation Report', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/models/object_detector_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"📈 Evaluation plots saved to data/models/object_detector_evaluation.png")

    return {'accuracy': accuracy, 'f1_weighted': f1_avg, 'precision': precision_avg,
            'recall': recall_avg}


# ═══════════════════════════════════════════════════════════════
#  EVALUATOR 3: Text Classification
# ═══════════════════════════════════════════════════════════════

def evaluate_text_processor():
    """
    Evaluate the Text Processor with NLP classification metrics.

    Same metrics as object detector (it's also classification),
    but applied to text sentiment analysis.

    ADDITIONAL METRIC:
    • Confidence Distribution
      Shows how confident the model is on average.
      Ideal: high confidence on correct predictions,
             low confidence on wrong predictions.
      Bad: model is highly confident but wrong → overconfident.
    """
    import torch
    from transformers import pipeline

    print("\n" + "=" * 60)
    print("  📝 EVALUATING: Text Processor (Sentiment Classification)")
    print("=" * 60)

    # ── 1. Prepare test data ───────────────────────────────────
    print("\n📦 Preparing test dataset...")

    test_data = [
        # (text, expected_label)
        ("This movie was absolutely amazing and wonderful!", "POSITIVE"),
        ("Terrible film. Worst I have ever seen.", "NEGATIVE"),
        ("A beautiful and heartwarming story!", "POSITIVE"),
        ("Boring, slow, and poorly written.", "NEGATIVE"),
        ("The performances were outstanding and moving.", "POSITIVE"),
        ("Complete waste of two hours of my life.", "NEGATIVE"),
        ("One of the best films of the decade!", "POSITIVE"),
        ("Painful to watch from beginning to end.", "NEGATIVE"),
        ("Incredible acting and stunning visuals!", "POSITIVE"),
        ("Predictable plot with zero originality.", "NEGATIVE"),
        ("A masterpiece that will stand the test of time.", "POSITIVE"),
        ("So bad it made me want to leave the theater.", "NEGATIVE"),
        ("Absolutely loved every minute of it!", "POSITIVE"),
        ("The script was lazy and the acting wooden.", "NEGATIVE"),
        ("A feel-good movie that warms the heart.", "POSITIVE"),
        ("Overhyped and deeply disappointing.", "NEGATIVE"),
        ("Brilliant direction and compelling story.", "POSITIVE"),
        ("Nothing but clichés and bad dialogue.", "NEGATIVE"),
        ("An emotional roller coaster in the best way!", "POSITIVE"),
        ("I regret paying money to see this disaster.", "NEGATIVE"),
    ]

    texts = [t[0] for t in test_data]
    expected = [t[1] for t in test_data]

    print(f"   Test samples: {len(texts)}")
    print(f"   Positive: {expected.count('POSITIVE')} | Negative: {expected.count('NEGATIVE')}")

    # ── 2. Load model and predict ──────────────────────────────
    print("\n🔮 Running sentiment analysis...")
    classifier = pipeline("sentiment-analysis")

    results = classifier(texts)
    predicted = [r['label'] for r in results]
    confidences = [r['score'] for r in results]

    # ── 3. Compute metrics ─────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix
    )

    accuracy = accuracy_score(expected, predicted) * 100
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, average=None, labels=['POSITIVE', 'NEGATIVE']
    )
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        expected, predicted, average='weighted'
    )
    cm = confusion_matrix(expected, predicted, labels=['POSITIVE', 'NEGATIVE'])

    # Confidence analysis
    correct_conf = [c for c, p, e in zip(confidences, predicted, expected) if p == e]
    wrong_conf = [c for c, p, e in zip(confidences, predicted, expected) if p != e]

    # ── 4. Display results ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📊 TEXT CLASSIFICATION METRICS REPORT")
    print(f"{'─'*60}")
    print(f"   Test samples: {len(texts)}\n")

    print_metric("Overall Accuracy", accuracy, "%", interpret_accuracy(accuracy))
    print_metric("Weighted Precision", precision_avg, "", "")
    print_metric("Weighted Recall", recall_avg, "", "")
    print_metric("Weighted F1 Score", f1_avg, "", interpret_f1(f1_avg))

    # Per-class
    labels_list = ['POSITIVE', 'NEGATIVE']
    print(f"\n   {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print(f"   {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for i, cls in enumerate(labels_list):
        print(f"   {cls:<12} {precision[i]:>10.3f} {recall[i]:>10.3f} "
              f"{f1[i]:>10.3f} {support[i]:>8d}")

    # Confidence analysis
    print(f"\n{'─'*60}")
    print(f"  🎯 CONFIDENCE ANALYSIS")
    print(f"{'─'*60}")
    if correct_conf:
        print(f"   Correct predictions:  avg confidence = {np.mean(correct_conf):.1%}")
    if wrong_conf:
        print(f"   Wrong predictions:    avg confidence = {np.mean(wrong_conf):.1%}")
        if np.mean(wrong_conf) > 0.80:
            print(f"   ⚠️  Model is overconfident on wrong predictions!")
    else:
        print(f"   ✅ No wrong predictions!")

    # ── 5. Individual results ──────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📝 INDIVIDUAL PREDICTIONS")
    print(f"{'─'*60}\n")

    for i, (text, exp, pred, conf) in enumerate(zip(texts, expected, predicted, confidences)):
        correct = "✅" if exp == pred else "❌"
        emoji = "😊" if pred == "POSITIVE" else "😞"
        print(f"   {correct} {emoji}  \"{text[:50]}{'...' if len(text)>50 else ''}\"")
        print(f"      Expected: {exp:<10} Got: {pred:<10} Confidence: {conf:.1%}")
        if exp != pred:
            print(f"      ⚠️  MISCLASSIFIED!")
        print()

    # ── 6. Confusion Matrix ────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  📊 CONFUSION MATRIX")
    print(f"{'─'*60}")
    print(f"""
                      PREDICTED
                  POS        NEG
   ACTUAL POS  [{cm[0][0]:>4}]     [{cm[0][1]:>4}]   ← {cm[0][0]} correct, {cm[0][1]} misclassified
         NEG  [{cm[1][0]:>4}]     [{cm[1][1]:>4}]   ← {cm[1][1]} correct, {cm[1][0]} misclassified
""")

    # ── 7. Interpretation ──────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  📖 HOW TO READ THESE RESULTS")
    print(f"{'─'*60}")
    print(f"""
   Accuracy: {accuracy:.1f}%
   → {accuracy:.0f} out of {len(texts)} reviews classified correctly.

   Precision for POSITIVE: {precision[0]:.3f}
   → Of reviews the model called positive, {precision[0]*100:.0f}% actually were.

   Recall for POSITIVE: {recall[0]:.3f}
   → Of all truly positive reviews, the model found {recall[0]*100:.0f}%.

   F1 for POSITIVE: {f1[0]:.3f}
   → {interpret_f1(f1[0])}
""")

    return {'accuracy': accuracy, 'f1_weighted': f1_avg, 'precision': precision_avg,
            'recall': recall_avg}


# ═══════════════════════════════════════════════════════════════
#  MAIN: Run evaluation from command line
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='🧠 Neuro Brain — Evaluation Pipeline')
    parser.add_argument(
        '--task',
        choices=['numbers', 'images', 'text', 'all'],
        default='all',
        help='Which model to evaluate'
    )
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════╗
    ║       🧠 NEURO BRAIN — EVALUATION PIPELINE      ║
    ╠══════════════════════════════════════════════════╣
    ║                                                  ║
    ║  Metrics computed:                               ║
    ║  ─────────────────────────────────────────────── ║
    ║  Regression:  MSE, RMSE, MAE, MAPE, R²          ║
    ║  Classification: Accuracy, Precision, Recall,    ║
    ║                  F1 Score, Confusion Matrix      ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)

    all_results = {}

    if args.task in ['numbers', 'all']:
        all_results['numbers'] = evaluate_number_predictor()

    if args.task in ['images', 'all']:
        all_results['images'] = evaluate_object_detector()

    if args.task in ['text', 'all']:
        all_results['text'] = evaluate_text_processor()

    # Summary
    print("\n" + "=" * 60)
    print("  📊 EVALUATION SUMMARY")
    print("=" * 60)

    for task, metrics in all_results.items():
        print(f"\n  {task.upper()}:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"    {key:<20} {val:.4f}")

    print("\n" + "=" * 60)
    print("  ✅ Evaluation complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
