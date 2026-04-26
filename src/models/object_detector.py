"""
🧠 Neuro Brain — Object Detector
Convolutional Neural Network (CNN) for image classification.

Architecture Overview:
    Input(32×32×3) → [Conv+BN+ReLU+Pool]×3 → Flatten → Dense(256)+ReLU → Dense(10)+Softmax

Activation Choices:
    • Conv blocks: ReLU — standard for vision. Lets the network learn to "fire" when
      it detects a visual feature (edge, corner, texture) and stay silent otherwise.
    • Dense hidden: ReLU — combines spatial features into high-level concepts.
    • Output: Softmax (implicit in CrossEntropyLoss) — converts 10 scores into
      probabilities ("85% bird, 10% airplane, 5% other").

Why CNN for images (not FNN)?
    A Feedforward network would treat each pixel independently — it wouldn't know
    that adjacent pixels form edges. CNNs use sliding filters that naturally detect
    spatial patterns: edges → shapes → objects, through their hierarchical structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class NeuroBrainCNN(nn.Module):
    """
    Convolutional Neural Network for image classification.

    Architecture (for 32×32 RGB images):
    ─────────────────────────────────────────────────────────────
    Layer                     Output Shape     Activation   Why
    ─────────────────────────────────────────────────────────────
    INPUT                     [3, 32, 32]      —
    Conv2d(3→32, 3×3)        [32, 32, 32]     —            Detect low-level features
    BatchNorm2d(32)           [32, 32, 32]     —            Stabilize training
    ReLU                      [32, 32, 32]     ReLU         Non-linearity for feature detection
    MaxPool2d(2×2)            [32, 16, 16]     —            Downsample, focus on important features

    Conv2d(32→64, 3×3)       [64, 16, 16]     —            Detect mid-level features
    BatchNorm2d(64)           [64, 16, 16]     —            Stabilize
    ReLU                      [64, 16, 16]     ReLU         Non-linearity
    MaxPool2d(2×2)            [64, 8, 8]       —            Downsample

    Conv2d(64→128, 3×3)      [128, 8, 8]      —            Detect high-level features
    BatchNorm2d(128)          [128, 8, 8]      —            Stabilize
    ReLU                      [128, 8, 8]      ReLU         Non-linearity
    MaxPool2d(2×2)            [128, 4, 4]      —            Downsample

    Flatten                   [2048]           —            3D → 1D for dense layers
    Dense(2048→256)           [256]            ReLU         Combine all visual features
    Dropout(0.5)              [256]            —            Prevent overfitting
    Dense(256→10)             [10]             Softmax*     Class probabilities
    ─────────────────────────────────────────────────────────────
    * Softmax is applied by CrossEntropyLoss during training
    """

    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.num_classes = num_classes

        # ══════════════════════════════════════════════════════
        # CONVOLUTIONAL BLOCKS — Feature Extraction
        # These layers learn WHAT visual patterns exist in images
        # ══════════════════════════════════════════════════════

        # ── Block 1: Edge & Color Detection ───────────────────
        # Conv2d: Slides 32 different 3×3 filters across the image.
        # Each filter learns to detect one specific pattern (horizontal edge,
        # red blob, diagonal line, etc.)
        #
        # BatchNorm: Normalizes outputs so values don't explode or vanish
        # as they pass through many layers. Makes training faster and stable.
        #
        # ReLU: f(x) = max(0, x)
        # WHY: When a filter detects its pattern, the output is positive → ReLU
        # passes it through. When the pattern is absent, output is negative →
        # ReLU outputs 0 (silence). This is exactly what we want: "fire when
        # pattern found, stay silent otherwise."
        #
        # MaxPool(2): Takes each 2×2 region and keeps only the maximum value.
        # Effect: Image shrinks from 32×32 → 16×16, but keeps the strongest
        # feature activations. This makes the network position-invariant (a cat
        # in the top-left is the same as a cat in the bottom-right).
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # 3→32 filters
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),          # Activation: ReLU
            nn.MaxPool2d(kernel_size=2),    # 32×32 → 16×16
        )

        # ── Block 2: Shape & Texture Detection ────────────────
        # Now has 64 filters that combine Block 1's edge/color features
        # into higher-level patterns: corners, curves, textures.
        # Same activation logic: ReLU fires when pattern is detected.
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32→64 filters
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),          # Activation: ReLU
            nn.MaxPool2d(kernel_size=2),    # 16×16 → 8×8
        )

        # ── Block 3: Object Part Detection ────────────────────
        # 128 filters combine textures/shapes into object parts:
        # wheels, eyes, wings, windows — depending on the dataset.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64→128 filters
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),          # Activation: ReLU
            nn.MaxPool2d(kernel_size=2),    # 8×8 → 4×4
        )

        # ══════════════════════════════════════════════════════
        # CLASSIFICATION HEAD — Decision Making
        # Takes all visual features and decides "what is this?"
        # ══════════════════════════════════════════════════════

        # Dense(2048→256) + ReLU:
        # Flatten takes the 128 feature maps of 4×4 = 2048 values.
        # Dense combines ALL of them (a dog needs eyes AND ears AND fur).
        # ReLU adds non-linearity to the combination.
        #
        # Dropout(0.5): During training, randomly zeros 50% of neurons.
        # This is aggressive but necessary: it forces the network to NOT
        # rely on any single visual cue (e.g., can't just look for "blue" to
        # detect airplane — must learn complete object shapes).
        #
        # Dense(256→10) — Output layer:
        # 10 neurons, one per class. The raw outputs are called "logits."
        # Softmax (applied by CrossEntropyLoss) converts logits to probabilities:
        #   logits [2.1, 0.5, -1.0, ...] → softmax → [0.65, 0.13, 0.03, ...]
        # WHY Softmax: It ensures all probabilities sum to 1.0 and the
        # network's "confidence" is represented as a valid distribution.
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # [128, 4, 4] → [2048]
            nn.Linear(128 * 4 * 4, 256),        # 2048 → 256
            nn.ReLU(inplace=True),              # Activation: ReLU
            nn.Dropout(0.5),                    # 50% dropout
            nn.Linear(256, num_classes),         # 256 → 10 class scores
            # Softmax is applied by nn.CrossEntropyLoss (not here!)
        )

    def forward(self, x):
        """
        Forward pass: data flows through all layers.
        Input:  [batch, 3, 32, 32]  — batch of RGB images
        Output: [batch, 10]         — class scores (logits)
        """
        x = self.block1(x)    # [B,3,32,32]  → [B,32,16,16]
        x = self.block2(x)    # [B,32,16,16] → [B,64,8,8]
        x = self.block3(x)    # [B,64,8,8]   → [B,128,4,4]
        x = self.classifier(x)  # [B,128,4,4] → [B,10]
        return x

    def explain_architecture(self):
        """Print a visual explanation of the architecture."""
        print("""
╔════════════════════════════════════════════════════════════════╗
║             🧠 OBJECT DETECTOR — CNN ARCHITECTURE             ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║   INPUT IMAGE: [3, 32, 32] — RGB 32×32 pixels                ║
║       │                                                        ║
║       ▼                                                        ║
║   ┌────────────────────────────────────────────────────┐       ║
║   │ BLOCK 1: Edge & Color Detection                    │       ║
║   │  Conv2d(3→32, 3×3)  — 32 edge/color filters       │       ║
║   │  BatchNorm2d(32)    — stabilize values             │       ║
║   │  ReLU               — fire when pattern found      │       ║
║   │  MaxPool(2×2)       — 32×32 → 16×16 (downsample)  │       ║
║   └──────────────────────────┬─────────────────────────┘       ║
║                              ▼                                 ║
║   ┌────────────────────────────────────────────────────┐       ║
║   │ BLOCK 2: Shape & Texture Detection                 │       ║
║   │  Conv2d(32→64, 3×3) — 64 shape/texture filters    │       ║
║   │  BatchNorm2d(64)    — stabilize                    │       ║
║   │  ReLU               — non-linearity                │       ║
║   │  MaxPool(2×2)       — 16×16 → 8×8                 │       ║
║   └──────────────────────────┬─────────────────────────┘       ║
║                              ▼                                 ║
║   ┌────────────────────────────────────────────────────┐       ║
║   │ BLOCK 3: Object Part Detection                     │       ║
║   │  Conv2d(64→128, 3×3) — 128 high-level filters     │       ║
║   │  BatchNorm2d(128)     — stabilize                  │       ║
║   │  ReLU                 — non-linearity              │       ║
║   │  MaxPool(2×2)         — 8×8 → 4×4                 │       ║
║   └──────────────────────────┬─────────────────────────┘       ║
║                              ▼                                 ║
║   ┌────────────────────────────────────────────────────┐       ║
║   │ CLASSIFIER                                         │       ║
║   │  Flatten         — [128,4,4] → [2048]              │       ║
║   │  Dense(2048→256) — combine all features            │       ║
║   │  ReLU            — non-linear combination          │       ║
║   │  Dropout(0.5)    — prevent overfitting             │       ║
║   │  Dense(256→10)   — one score per class             │       ║
║   │  Softmax*        — convert to probabilities        │       ║
║   └────────────────────────────────────────────────────┘       ║
║                                                                ║
║   OUTPUT: [0.02, 0.01, 0.85, 0.03, ...]                      ║
║           airplane car  BIRD  cat   ...                        ║
║                                                                ║
║   * Softmax is applied by CrossEntropyLoss during training     ║
╚════════════════════════════════════════════════════════════════╝
""")


class ObjectDetector:
    """Wrapper class for training and using the CNN model."""

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, num_classes=10, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuroBrainCNN(num_classes=num_classes).to(self.device)

        # CrossEntropyLoss = Softmax + NLLLoss combined
        # It applies softmax internally, which is why our model's last layer
        # outputs raw logits (not softmax). This is more numerically stable.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self, data_dir='./data/raw', batch_size=64):
        """Load CIFAR-10 dataset with augmentation."""
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )

        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        print(f"📦 Loaded {len(trainset)} train / {len(testset)} test images")

    def train(self, epochs=10):
        """Train the model."""
        print(f"🧠 Training Object Detector for {epochs} epochs on {self.device}...\n")
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{running_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })

            print(f"  → Epoch {epoch+1} | Loss: {running_loss/len(self.trainloader):.4f} "
                  f"| Accuracy: {100.*correct/total:.2f}%")

    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def predict_single(self, image_tensor):
        """Predict the class of a single image."""
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = probabilities.max(1)

            print(f"🔍 Prediction: {self.CLASSES[predicted_class.item()]}")
            print(f"   Confidence: {confidence.item():.1%}")
            print(f"   All probabilities:")
            for i, (cls, prob) in enumerate(zip(self.CLASSES, probabilities[0])):
                bar = "█" * int(prob.item() * 30)
                print(f"     {cls:>12}: {prob.item():.1%} {bar}")
            return predicted_class.item(), confidence.item()

    def save(self, path="data/models/object_detector.pth"):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")

    def load(self, path="data/models/object_detector.pth"):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"📂 Model loaded from {path}")


# ── Quick Demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🧠 Object Detector — Full Demo")
    print("=" * 55)

    model = NeuroBrainCNN()
    model.explain_architecture()

    # Show total parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parameters: {total:,} total, {trainable:,} trainable\n")

    detector = ObjectDetector()
    detector.load_data()
    detector.train(epochs=5)
    detector.evaluate()
