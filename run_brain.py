"""
🧠 Neuro Brain — Run Everything Demo
All-in-one runner using PyTorch for all models.

Usage:
    python run_brain.py              # Run all demos
    python run_brain.py --numbers    # Number prediction only
    python run_brain.py --images     # Image classification only
    python run_brain.py --text       # Text analysis only
    python run_brain.py --ensemble   # Ensemble demo
"""

import argparse
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

# Ensure data dirs exist
Path("data/models").mkdir(parents=True, exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. NUMBER PREDICTOR — PyTorch FNN
# ═══════════════════════════════════════════════════════════════

class NumberPredictorNet(nn.Module):
    """
    Feedforward Neural Network for regression (PyTorch version).

    Architecture:
        Input(4) → Dense(128)+ReLU → Dense(64)+ReLU → Dense(32)+ReLU → Dense(1)

    Activation Choices:
        Hidden: ReLU — fast, avoids vanishing gradients
        Output: None (linear) — regression output is unbounded
    """
    def __init__(self, input_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),  # Linear output for regression
        )

    def forward(self, x):
        return self.network(x)


def run_number_predictor():
    """Train and evaluate the Number Predictor."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    print("\n" + "═" * 60)
    print("  🔢 NEURO BRAIN — Number Predictor (FNN)")
    print("═" * 60)

    # ── Generate Dataset ───────────────────────────────────────
    print("\n📦 Generating house price dataset...")
    np.random.seed(42)
    n = 5000

    sq_ft = np.random.randint(500, 5000, n).astype(np.float32)
    beds = np.random.randint(1, 6, n).astype(np.float32)
    age = np.random.randint(0, 50, n).astype(np.float32)
    dist = np.random.uniform(0.5, 30, n).astype(np.float32)

    X = np.column_stack([sq_ft, beds, age, dist])
    y = (sq_ft * 150 + beds * 20000 - age * 1000 -
         dist * 5000 + np.random.randn(n).astype(np.float32) * 10000)

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Split 80/20
    split = int(0.8 * n)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    # To tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    print(f"   Train: {len(X_train)} | Test: {len(X_test)} samples")
    print(f"   Features: sqft, bedrooms, age, distance")

    # ── Build & Train ──────────────────────────────────────────
    model = NumberPredictorNet(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    EPOCHS = 50
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    print(f"   Loss: MSE | Optimizer: Adam(lr=0.001) | Batch: 32\n")

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_t)
                val_loss = criterion(val_pred, y_test_t).item()
            print(f"   Epoch {epoch+1:>3}/{EPOCHS} │ Train Loss: {epoch_loss/len(train_loader):.6f} │ Val Loss: {val_loss:.6f}")

    train_time = time.time() - start

    # ── Evaluate ───────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy()

    # Convert back to actual prices
    pred_actual = scaler_y.inverse_transform(test_pred)
    y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    r2 = r2_score(y_actual, pred_actual)
    mae = mean_absolute_error(y_actual, pred_actual)

    print(f"\n{'─'*50}")
    print(f"  📊 RESULTS")
    print(f"{'─'*50}")
    print(f"   Training time:  {train_time:.1f}s")
    print(f"   R² Score:       {r2:.4f}  {'🟢 Excellent' if r2 > 0.9 else '🟡 Good' if r2 > 0.7 else '🟠 Fair'}")
    print(f"   MAE:            ${mae:,.0f}  (avg prediction error)")

    print(f"\n  🎯 SAMPLE PREDICTIONS")
    print(f"   {'Predicted':>12}  {'Actual':>12}  {'Error':>10}")
    print(f"   {'─'*12}  {'─'*12}  {'─'*10}")
    for p, a in zip(pred_actual[:8], y_actual[:8]):
        err = abs(p[0] - a[0])
        print(f"   ${p[0]:>10,.0f}  ${a[0]:>10,.0f}  ${err:>8,.0f}")

    # Save
    torch.save(model.state_dict(), "data/models/number_predictor.pth")
    print(f"\n💾 Model saved to data/models/number_predictor.pth")

    return model, r2, mae


# ═══════════════════════════════════════════════════════════════
#  2. OBJECT DETECTOR — CNN (CIFAR-10)
# ═══════════════════════════════════════════════════════════════

def run_object_detector():
    """Train and evaluate the Object Detector CNN."""
    import torchvision
    import torchvision.transforms as transforms
    from tqdm import tqdm

    print("\n" + "═" * 60)
    print("  👁️ NEURO BRAIN — Object Detector (CNN)")
    print("═" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n   Device: {device}")

    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # ── Load CIFAR-10 ──────────────────────────────────────────
    print("\n📦 Downloading CIFAR-10 dataset (first run only)...")
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

    trainset = torchvision.datasets.CIFAR10(root='data/raw', train=True,
                                             download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='data/raw', train=False,
                                            download=True, transform=test_transform)

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    print(f"   Train: {len(trainset)} images | Test: {len(testset)} images")

    # ── Build CNN ──────────────────────────────────────────────
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: edges, colors
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                # Block 2: shapes, textures
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
                # Block 3: object parts
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # ── Train ──────────────────────────────────────────────────
    EPOCHS = 10
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    print(f"   Loss: CrossEntropy | Optimizer: Adam(lr=0.001) | Batch: 64\n")

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{EPOCHS}", leave=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                           acc=f"{100.*correct/total:.1f}%")

    train_time = time.time() - start

    # ── Evaluate ───────────────────────────────────────────────
    print(f"\n📊 Evaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if pred[i] == label:
                    class_correct[label] += 1

    accuracy = 100. * correct / total

    print(f"\n{'─'*50}")
    print(f"  📊 RESULTS")
    print(f"{'─'*50}")
    print(f"   Training time:   {train_time:.1f}s")
    print(f"   Overall Accuracy: {accuracy:.2f}% {'🟢' if accuracy > 75 else '🟡' if accuracy > 60 else '🟠'}")
    print(f"\n  Per-class accuracy:")
    for i, cls in enumerate(CLASSES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            bar = "█" * int(acc / 5)
            print(f"   {cls:>12}: {acc:>5.1f}% {bar}")

    # Save
    torch.save(model.state_dict(), "data/models/object_detector.pth")
    print(f"\n💾 Model saved to data/models/object_detector.pth")

    return model, accuracy


# ═══════════════════════════════════════════════════════════════
#  3. TEXT PROCESSOR — Transformer Sentiment Analysis
# ═══════════════════════════════════════════════════════════════

def run_text_processor():
    """Run the Text Processor for sentiment analysis."""
    print("\n" + "═" * 60)
    print("  📝 NEURO BRAIN — Text Processor (Transformer)")
    print("═" * 60)

    try:
        from transformers import pipeline
    except ImportError:
        print("\n   ⚠️ transformers not installed. Skipping text demo.")
        print("   Install with: python -m pip install transformers")
        return None, 0

    print("\n📥 Loading pre-trained sentiment model...")
    print("   (First run downloads ~260MB model)\n")

    try:
        classifier = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        print(f"   ⚠️ Error loading model: {e}")
        print("   Trying default model...")
        try:
            classifier = pipeline("sentiment-analysis")
        except Exception as e2:
            print(f"   ❌ Cannot load text model: {e2}")
            return None, 0

    # ── Run Analysis ───────────────────────────────────────────
    test_texts = [
        ("I absolutely love building neural networks!", "POSITIVE"),
        ("This is the worst code I have ever seen.", "NEGATIVE"),
        ("The weather is nice today.", "POSITIVE"),
        ("I am so frustrated with these bugs.", "NEGATIVE"),
        ("Amazing progress on the AI project!", "POSITIVE"),
        ("This product completely failed my expectations.", "NEGATIVE"),
        ("The documentation is clear and helpful.", "POSITIVE"),
        ("Nothing works and support is terrible.", "NEGATIVE"),
        ("What a beautiful and elegant solution!", "POSITIVE"),
        ("Slow, buggy, and unreliable software.", "NEGATIVE"),
    ]

    print(f"  📝 SENTIMENT ANALYSIS RESULTS")
    print(f"{'─'*60}\n")

    correct = 0
    total = len(test_texts)

    for text, expected in test_texts:
        result = classifier(text)[0]
        label = result['label']
        conf = result['score']
        is_correct = label == expected

        if is_correct:
            correct += 1

        emoji = "😊" if label == "POSITIVE" else "😞"
        mark = "✅" if is_correct else "❌"

        print(f"   {mark} {emoji} [{conf:.0%}] \"{text[:55]}\"")

    accuracy = 100 * correct / total

    print(f"\n{'─'*50}")
    print(f"  📊 RESULTS")
    print(f"{'─'*50}")
    print(f"   Accuracy: {accuracy:.0f}% ({correct}/{total} correct)")
    print(f"   Model: DistilBERT (66M parameters)")
    print(f"   {'🟢 Excellent' if accuracy > 90 else '🟡 Good' if accuracy > 70 else '🟠 Fair'}")

    return classifier, accuracy


# ═══════════════════════════════════════════════════════════════
#  4. ENSEMBLE DEMO
# ═══════════════════════════════════════════════════════════════

def run_ensemble_demo():
    """Demo: combine multiple CNN models."""
    print("\n" + "═" * 60)
    print("  🔗 NEURO BRAIN — Ensemble Demo")
    print("═" * 60)

    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Build simple CNN (reusable definition)
    class MiniCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
                nn.Linear(64, 10),
            )
        def forward(self, x):
            return self.net(x)

    # Create 3 models with different random weights (simulates different training)
    print("\n   Creating 3 models for ensemble...")
    models = [MiniCNN() for _ in range(3)]

    # Dummy input (4 random images)
    dummy = torch.randn(4, 3, 32, 32)

    # Individual predictions
    print(f"\n   Individual model predictions:")
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            out = model(dummy)
            preds = out.argmax(dim=1)
        print(f"   Model {i+1}: {[CLASSES[p] for p in preds.tolist()]}")

    # Soft voting ensemble
    print(f"\n   🔗 Soft Voting Ensemble (weighted average):")
    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(dummy)
            probs = torch.softmax(out, dim=1)
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    ensemble_preds = avg_probs.argmax(dim=1)
    ensemble_confs = avg_probs.max(dim=1)[0]

    for i in range(4):
        print(f"   Image {i+1}: {CLASSES[ensemble_preds[i]]} ({ensemble_confs[i]:.1%})")

    print(f"\n   ✅ Ensemble reduces individual model errors by averaging!")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🧠 Neuro Brain — Run All Demos")
    parser.add_argument('--numbers', action='store_true', help='Run number predictor')
    parser.add_argument('--images', action='store_true', help='Run object detector')
    parser.add_argument('--text', action='store_true', help='Run text processor')
    parser.add_argument('--ensemble', action='store_true', help='Run ensemble demo')
    args = parser.parse_args()

    # If no flags, run everything
    run_all = not (args.numbers or args.images or args.text or args.ensemble)

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║           🧠  N E U R O   B R A I N  🧠             ║
    ║                                                      ║
    ║       Personal AI — Neural Network Platform          ║
    ║                                                      ║
    ╠══════════════════════════════════════════════════════╣
    ║  Models:                                             ║
    ║   🔢 Number Predictor  — FNN regression              ║
    ║   👁️ Object Detector   — CNN image classification    ║
    ║   📝 Text Processor    — Transformer sentiment       ║
    ║   🔗 Ensemble          — Multi-model combination     ║
    ╚══════════════════════════════════════════════════════╝
    """)

    results = {}
    total_start = time.time()

    # ── Run selected demos ─────────────────────────────────────
    if run_all or args.numbers:
        _, r2, mae = run_number_predictor()
        results['numbers'] = {'R²': f"{r2:.4f}", 'MAE': f"${mae:,.0f}"}

    if run_all or args.images:
        _, acc = run_object_detector()
        results['images'] = {'Accuracy': f"{acc:.2f}%"}

    if run_all or args.text:
        _, acc = run_text_processor()
        if acc:
            results['text'] = {'Accuracy': f"{acc:.0f}%"}

    if run_all or args.ensemble:
        run_ensemble_demo()
        results['ensemble'] = {'Status': 'Complete'}

    # ── Final Summary ──────────────────────────────────────────
    total_time = time.time() - total_start

    print("\n" + "═" * 60)
    print("  🧠 NEURO BRAIN — FINAL SUMMARY")
    print("═" * 60)

    for task, metrics in results.items():
        print(f"\n   {task.upper()}:")
        for key, val in metrics.items():
            print(f"     {key}: {val}")

    print(f"\n   Total runtime: {total_time:.1f}s")
    print(f"   Models saved to: data/models/")
    print("\n" + "═" * 60)
    print("  ✅ All systems operational! 🧠")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
