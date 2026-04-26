"""
🧠 Neuro Brain — Ensemble & Multi-Network Systems
Combine multiple neural networks for higher accuracy and multi-task AI.

Three Strategies:
─────────────────────────────────────────────────────────────
1. VOTING ENSEMBLE:  Run same task on multiple models, take majority vote.
2. MULTI-MODAL:      Combine different input types (image+text → decision).
3. STACKING:         Feed model outputs into a meta-model for final prediction.
─────────────────────────────────────────────────────────────

Usage:
    python -m src.models.ensemble
"""

import numpy as np
import torch
import torch.nn as nn
from collections import Counter


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 1: Voting Ensemble
# ═══════════════════════════════════════════════════════════════

class VotingEnsemble:
    """
    Combine predictions from multiple models using voting.

    How it works:
    ─────────────────────────────────────────────────────────
    Model A says → "cat"  (confidence 0.85)
    Model B says → "cat"  (confidence 0.70)
    Model C says → "dog"  (confidence 0.60)

    Hard voting: majority wins → "cat" (2 out of 3)
    Soft voting: average probabilities → best avg → "cat"
    ─────────────────────────────────────────────────────────

    WHY this works:
    Different models make different mistakes. Model A might
    struggle with birds but excel at cats. Model B might be
    the opposite. Together, they cover each other's weaknesses.
    """

    def __init__(self, models=None, weights=None, voting='soft'):
        """
        Args:
            models:  List of trained PyTorch models
            weights: Optional list of floats (trust per model).
                     [0.5, 0.3, 0.2] = model 1 trusted most.
            voting:  'hard' (majority class) or 'soft' (avg probabilities)
        """
        self.models = models or []
        self.weights = weights
        self.voting = voting

        if weights and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

    def predict(self, x):
        """
        Make a prediction using all models.

        Returns the class with the highest combined score.
        """
        if not self.models:
            raise ValueError("No models in ensemble! Call add_model() first.")

        all_probs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs)

        if self.voting == 'soft':
            return self._soft_vote(all_probs)
        else:
            return self._hard_vote(all_probs)

    def _soft_vote(self, all_probs):
        """
        SOFT VOTING: Average the probability distributions, then pick the best.

        Model A: [0.80 cat, 0.15 dog, 0.05 bird]
        Model B: [0.60 cat, 0.30 dog, 0.10 bird]
        Model C: [0.40 cat, 0.50 dog, 0.10 bird]
        ─────────────────────────────────────────
        Average: [0.60 cat, 0.32 dog, 0.08 bird] → CAT ✅
        """
        weights = self.weights or [1.0] * len(all_probs)
        total_weight = sum(weights)

        weighted_sum = torch.zeros_like(all_probs[0])
        for probs, weight in zip(all_probs, weights):
            weighted_sum += probs * (weight / total_weight)

        final_class = weighted_sum.argmax(dim=1)
        final_confidence = weighted_sum.max(dim=1)[0]
        return final_class, final_confidence, weighted_sum

    def _hard_vote(self, all_probs):
        """
        HARD VOTING: Each model votes for one class, majority wins.

        Model A votes → "cat"
        Model B votes → "cat"
        Model C votes → "dog"
        Majority → "cat" (2/3 votes)
        """
        votes = []
        for probs in all_probs:
            votes.append(probs.argmax(dim=1))

        # Stack votes and find majority for each sample
        vote_stack = torch.stack(votes, dim=0)  # [n_models, batch]
        batch_size = vote_stack.shape[1]

        final_classes = []
        for i in range(batch_size):
            sample_votes = vote_stack[:, i].tolist()
            majority = Counter(sample_votes).most_common(1)[0][0]
            final_classes.append(majority)

        final_class = torch.tensor(final_classes)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        final_conf = avg_probs.max(dim=1)[0]
        return final_class, final_conf, avg_probs


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 2: Multi-Modal Fusion
# ═══════════════════════════════════════════════════════════════

class MultiModalFusion(nn.Module):
    """
    Combine image features and text features into a single decision.

    Architecture:
    ─────────────────────────────────────────────────────────
    Image → CNN → [256-dim vector] ─┐
                                     ├→ Fuse → Dense → Prediction
    Text → Transformer → [256-dim] ─┘

    Fusion strategies:
    1. Concatenate: [img_feats, text_feats] → Dense [simple, effective]
    2. Add:         img_feats + text_feats  [if same dimension]
    3. Attention:   cross-attention between modalities [advanced]
    ─────────────────────────────────────────────────────────

    Use cases:
    • Social media: analyze image + caption together
    • E-commerce: product image + description → category
    • Medical: scan image + patient notes → diagnosis
    """

    def __init__(self, image_feat_dim=256, text_feat_dim=256,
                 num_classes=10, fusion='concat'):
        super().__init__()
        self.fusion = fusion

        # Image feature extractor (from CNN's penultimate layer)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Text feature extractor (from Transformer's CLS token)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Fusion classifier
        if fusion == 'concat':
            fused_dim = 256  # 128 + 128
        elif fusion == 'add':
            fused_dim = 128  # same dimension
        else:
            fused_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_features, text_features):
        """
        Forward pass: combine image and text features.

        Args:
            image_features: [batch, image_feat_dim] from CNN
            text_features:  [batch, text_feat_dim] from Transformer
        """
        img_encoded = self.image_encoder(image_features)   # [batch, 128]
        txt_encoded = self.text_encoder(text_features)     # [batch, 128]

        # Fuse the two modalities
        if self.fusion == 'concat':
            # Concatenate: [img_128, txt_128] → [256]
            fused = torch.cat([img_encoded, txt_encoded], dim=1)
        elif self.fusion == 'add':
            # Element-wise addition (requires same dimension)
            fused = img_encoded + txt_encoded
        else:
            fused = torch.cat([img_encoded, txt_encoded], dim=1)

        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 3: Stacking
# ═══════════════════════════════════════════════════════════════

class StackingEnsemble:
    """
    Stacking: Feed model outputs into a meta-model for final prediction.

    Architecture:
    ─────────────────────────────────────────────────────────
    Input → Model A → predictions_A ─┐
    Input → Model B → predictions_B ─┼→ Meta-Model → Final Prediction
    Input → Model C → predictions_C ─┘

    The meta-model learns WHICH base model to trust for which
    input patterns. Example: it might learn that Model A is
    best for outdoor scenes, Model B for indoor scenes.
    ─────────────────────────────────────────────────────────

    WHY stacking works better than simple voting:
    Voting treats all models equally. Stacking learns that
    some models are better at certain types of inputs.
    """

    def __init__(self, base_models, num_classes=10):
        self.base_models = base_models
        n_models = len(base_models)

        # Meta-model: takes base model outputs, produces final prediction
        self.meta_model = nn.Sequential(
            nn.Linear(n_models * num_classes, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def get_base_predictions(self, x):
        """Get predictions from all base models."""
        all_preds = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                all_preds.append(probs)
        # Concatenate: [batch, n_models * n_classes]
        return torch.cat(all_preds, dim=1)

    def train_meta(self, train_loader, epochs=20):
        """Train the meta-model on base model outputs."""
        print("🧠 Training stacking meta-model...")
        self.meta_model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                # Get base model predictions (no gradient needed)
                meta_input = self.get_base_predictions(inputs)

                # Train meta-model
                self.optimizer.zero_grad()
                output = self.meta_model(meta_input)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            if (epoch + 1) % 5 == 0:
                acc = 100. * correct / total
                print(f"   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.1f}%")

    def predict(self, x):
        """Make prediction using stacked ensemble."""
        meta_input = self.get_base_predictions(x)
        self.meta_model.eval()
        with torch.no_grad():
            output = self.meta_model(meta_input)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)
        return predicted, confidence, probs


# ═══════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  🧠 Ensemble & Multi-Network — Demo")
    print("=" * 55)

    # ── Demo 1: Voting Ensemble ────────────────────────────────
    print("\n📊 Demo 1: Voting Ensemble")
    print("-" * 45)

    from src.models.object_detector import NeuroBrainCNN

    # Create 3 different models (in practice, train each differently)
    model_a = NeuroBrainCNN(num_classes=10)
    model_b = NeuroBrainCNN(num_classes=10)
    model_c = NeuroBrainCNN(num_classes=10)

    # Soft voting ensemble
    ensemble = VotingEnsemble(
        models=[model_a, model_b, model_c],
        weights=[0.5, 0.3, 0.2],  # Trust model A most
        voting='soft'
    )

    # Test with dummy image batch
    dummy_images = torch.randn(4, 3, 32, 32)  # 4 images
    classes, confidences, probs = ensemble.predict(dummy_images)

    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(4):
        print(f"   Image {i+1}: {LABELS[classes[i]]} ({confidences[i]:.1%})")

    # ── Demo 2: Multi-Modal Fusion ────────────────────────────
    print(f"\n📊 Demo 2: Multi-Modal Fusion (Image + Text)")
    print("-" * 45)

    fusion = MultiModalFusion(
        image_feat_dim=256,
        text_feat_dim=256,
        num_classes=5,
        fusion='concat'
    )

    # Simulated features
    img_feats = torch.randn(2, 256)   # 2 image feature vectors
    txt_feats = torch.randn(2, 256)   # 2 text feature vectors

    output = fusion(img_feats, txt_feats)
    pred = torch.softmax(output, dim=1)

    categories = ['nature', 'food', 'tech', 'sports', 'travel']
    for i in range(2):
        best = pred[i].argmax().item()
        print(f"   Sample {i+1}: {categories[best]} ({pred[i][best]:.1%})")

    total = sum(p.numel() for p in fusion.parameters())
    print(f"\n   Fusion model parameters: {total:,}")

    # ── Demo 3: Stacking ──────────────────────────────────────
    print(f"\n📊 Demo 3: Stacking Ensemble")
    print("-" * 45)

    stack = StackingEnsemble(
        base_models=[model_a, model_b, model_c],
        num_classes=10
    )

    dummy_input = torch.randn(4, 3, 32, 32)
    preds, confs, all_probs = stack.predict(dummy_input)

    for i in range(4):
        print(f"   Image {i+1}: {LABELS[preds[i]]} ({confs[i]:.1%})")

    meta_params = sum(p.numel() for p in stack.meta_model.parameters())
    print(f"\n   Meta-model parameters: {meta_params:,}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n\n{'='*55}")
    print(f"  📖 COMPARISON")
    print(f"{'='*55}")
    print(f"""
   ┌──────────────┬──────────────────────────────────────┐
   │ Strategy     │ Best When                            │
   ├──────────────┼──────────────────────────────────────┤
   │ Voting       │ You have multiple trained models     │
   │              │ for the same task                    │
   ├──────────────┼──────────────────────────────────────┤
   │ Multi-Modal  │ You have different input types       │
   │              │ (image + text, audio + video)        │
   ├──────────────┼──────────────────────────────────────┤
   │ Stacking     │ You want a smart combiner that       │
   │              │ learns which model to trust when     │
   └──────────────┴──────────────────────────────────────┘
    """)
