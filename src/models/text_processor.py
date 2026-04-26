"""
🧠 Neuro Brain — Text Processor
Transformer-based NLP using Hugging Face for text analysis and generation.

Architecture Overview (DistilBERT — used for classification):
    Tokens → Embedding(768) → [Self-Attention + FF(GELU)]×6 → Dense+Softmax

Activation Choices:
    • Feed-Forward hidden: GELU — smooth non-linearity that softly gates values.
      Unlike ReLU's hard cutoff at 0, GELU "partially" allows small negatives
      through. This preserves subtle language signals that ReLU would kill.
    • Classification output: Softmax — probability distribution over classes.
    • Generation output: Softmax over vocabulary — probability of each next word.

Why Transformer for text (not CNN/RNN)?
    • CNN: sees fixed-size windows, misses long-range meaning ("The movie I saw
      last weekend with my friends from college was terrible" — CNN might not
      connect "movie" to "terrible" across 10+ words).
    • RNN/LSTM: reads one word at a time sequentially (slow), struggles with
      very long sentences.
    • Transformer: self-attention lets EVERY word look at EVERY other word
      simultaneously. "terrible" directly attends to "movie" regardless of distance.

This file wraps pre-trained models. For custom training, see the guide:
    NEURAL_NETWORK_GUIDE.md
"""

import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class TransformerClassifier(nn.Module):
    """
    Custom Transformer-based text classifier built from scratch (for learning).
    Shows the internal architecture of what Hugging Face models do under the hood.

    Architecture:
    ─────────────────────────────────────────────────────────────
    Layer                      Size          Activation   Purpose
    ─────────────────────────────────────────────────────────────
    Embedding                  vocab→768     —            Words → dense vectors
    Positional Encoding        768           —            Add position information
    ┌─ Transformer Block (×6) ────────────────────────────────┐
    │  Multi-Head Attention    768           Softmax*     Each word attends to all others
    │  Layer Norm              768           —            Stabilize values
    │  Feed-Forward:                                     Process attention output
    │    Linear(768→3072)      3072          GELU         Expand + non-linearity
    │    Linear(3072→768)      768           —            Compress back
    │  Layer Norm              768           —            Stabilize
    └─────────────────────────────────────────────────────────┘
    Classification Head:
      Linear(768→256)          256           ReLU         Combine representations
      Dropout(0.3)             256           —            Prevent overfitting
      Linear(256→num_classes)  num_classes   Softmax      Class probabilities
    ─────────────────────────────────────────────────────────────

    * Softmax in attention: computes HOW MUCH each word should pay attention
      to every other word. E.g., in "The cat sat on the mat", "sat" might
      attend strongly to "cat" (who sat?) and "mat" (where?).
    """

    def __init__(self, vocab_size=30522, embed_dim=256, num_heads=4,
                 num_layers=2, num_classes=2, max_length=128):
        super().__init__()

        # ── EMBEDDING LAYER ───────────────────────────────────
        # Converts integer token IDs into dense vectors.
        # "cat" (token 2368) → [0.12, -0.45, 0.78, ...] (256-dim vector)
        # WHY: Neural networks need continuous numbers, not discrete IDs.
        # The embedding is LEARNED — similar words get similar vectors.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # ── POSITIONAL ENCODING ───────────────────────────────
        # Adds position information. Without this, "dog bites man" and
        # "man bites dog" would look identical to the model (same words!).
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        # ── TRANSFORMER ENCODER BLOCKS ────────────────────────
        # Each block: Self-Attention + Feed-Forward with GELU
        #
        # Self-Attention mechanism:
        #   For each word, compute attention scores with ALL other words.
        #   "I love my cat" → "cat" attends to "my" (whose cat?) and "love" (sentiment)
        #   Uses Softmax to normalize scores into probabilities (sum to 1).
        #
        # Feed-Forward network inside each block:
        #   Linear(embed_dim → 4*embed_dim) + GELU + Linear(4*embed_dim → embed_dim)
        #
        # WHY GELU (not ReLU)?
        #   GELU(x) = x * Φ(x), where Φ is the Gaussian CDF
        #   It's smooth: values near 0 are partially passed through
        #   ReLU has a hard cutoff: any x<0 → 0 (information lost)
        #   For language, subtle negative signals can carry meaning
        #   (e.g., negation: "not good" has important negative activations)
        #   GELU preserves this nuance → better language understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,   # Expand 4× then compress back
            dropout=0.1,
            activation='gelu',               # ← GELU activation
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ── CLASSIFICATION HEAD ───────────────────────────────
        # Takes the [CLS] token representation and maps to class probabilities.
        #
        # Linear(256→128) + ReLU:
        #   Combines all the contextual information into a decision space.
        #   ReLU here is fine (simpler than GELU, and this is a small head).
        #
        # Linear(128→num_classes):
        #   Final layer outputs one score per class.
        #   Softmax is applied by CrossEntropyLoss (not here, for numerical stability).
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),   # 256 → 128
            nn.ReLU(),                               # Activation: ReLU
            nn.Dropout(0.3),                         # Prevent overfitting
            nn.Linear(embed_dim // 2, num_classes),  # 128 → num_classes
            # Softmax applied by loss function
        )

        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the Transformer.

        Args:
            input_ids: [batch, seq_len] — integer token IDs
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embed tokens + positions
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Create mask for padding tokens (Transformer needs inverted mask)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Pass through Transformer blocks
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Use [CLS] token (first token) for classification
        cls_output = x[:, 0, :]  # [batch, embed_dim]

        # Classify
        logits = self.classifier(cls_output)  # [batch, num_classes]
        return logits

    @staticmethod
    def explain_architecture():
        """Print a visual explanation of the Transformer architecture."""
        print("""
╔════════════════════════════════════════════════════════════════╗
║            🧠 TEXT PROCESSOR — TRANSFORMER ARCHITECTURE       ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║   INPUT: "I love building neural networks"                    ║
║       │                                                        ║
║       ▼  Tokenizer                                            ║
║   [101, 1045, 2293, 2311, 7842, 5765, 102]                   ║
║       │                                                        ║
║       ▼                                                        ║
║   ┌────────────────────────────────────────────────────┐       ║
║   │ EMBEDDING + POSITION                               │       ║
║   │  Token Embedding:    word → 256-dim vector         │       ║
║   │  Position Embedding: position → 256-dim vector     │       ║
║   │  Combined:           token + position              │       ║
║   │  No activation (linear projection)                 │       ║
║   └──────────────────────────┬─────────────────────────┘       ║
║                              ▼                                 ║
║   ┌────────────────────────────────────────────────────┐       ║
║   │ TRANSFORMER BLOCK (×N layers)                      │       ║
║   │                                                    │       ║
║   │  ┌──────────────────────────────────────────────┐  │       ║
║   │  │ MULTI-HEAD SELF-ATTENTION                    │  │       ║
║   │  │  For each word, compute attention to ALL     │  │       ║
║   │  │  other words using Softmax                   │  │       ║
║   │  │  "love" ←→ "networks" (high attention)       │  │       ║
║   │  │  "I" ←→ "love" (high attention)              │  │       ║
║   │  │  Activation: Softmax (attention weights)     │  │       ║
║   │  └──────────────────────────────────────────────┘  │       ║
║   │                    │                               │       ║
║   │                    ▼                               │       ║
║   │  ┌──────────────────────────────────────────────┐  │       ║
║   │  │ FEED-FORWARD NETWORK                        │  │       ║
║   │  │  Linear(256 → 1024) + GELU                  │  │       ║
║   │  │  Linear(1024 → 256)                         │  │       ║
║   │  │                                             │  │       ║
║   │  │  WHY GELU (not ReLU)?                       │  │       ║
║   │  │  • ReLU: x<0 → 0    (hard cutoff)           │  │       ║
║   │  │  • GELU: x<0 → soft  (gradual transition)   │  │       ║
║   │  │  Language has subtle negative signals that   │  │       ║
║   │  │  GELU preserves but ReLU would destroy.     │  │       ║
║   │  └──────────────────────────────────────────────┘  │       ║
║   └──────────────────────────┬─────────────────────────┘       ║
║                              ▼                                 ║
║   ┌────────────────────────────────────────────────────┐       ║
║   │ CLASSIFICATION HEAD                                │       ║
║   │  Take [CLS] token representation                  │       ║
║   │  Dense(256→128) + ReLU  — combine features        │       ║
║   │  Dropout(0.3)           — regularization           │       ║
║   │  Dense(128→2)           — class scores             │       ║
║   │  Softmax                — probabilities            │       ║
║   └────────────────────────────────────────────────────┘       ║
║                                                                ║
║   OUTPUT: [0.95, 0.05] → POSITIVE (95% confidence)           ║
╚════════════════════════════════════════════════════════════════╝
""")


class TextProcessor:
    """
    Text processing using pre-trained Transformer models (Hugging Face).
    Uses production-ready models for real-world NLP tasks.
    """

    def __init__(self):
        self._sentiment = None
        self._generator = None
        self._summarizer = None
        self._ner = None

    @property
    def sentiment_analyzer(self):
        """Lazy-load sentiment analysis pipeline."""
        if self._sentiment is None:
            print("📥 Loading sentiment analysis model...")
            self._sentiment = pipeline("sentiment-analysis")
        return self._sentiment

    @property
    def text_generator(self):
        """Lazy-load text generation pipeline."""
        if self._generator is None:
            print("📥 Loading text generation model (GPT-2)...")
            self._generator = pipeline("text-generation", model="gpt2")
        return self._generator

    @property
    def summarizer(self):
        """Lazy-load summarization pipeline."""
        if self._summarizer is None:
            print("📥 Loading summarization model...")
            self._summarizer = pipeline("summarization")
        return self._summarizer

    @property
    def ner_detector(self):
        """Lazy-load Named Entity Recognition pipeline."""
        if self._ner is None:
            print("📥 Loading NER model...")
            self._ner = pipeline("ner", grouped_entities=True)
        return self._ner

    def analyze_sentiment(self, text):
        """Analyze the sentiment of input text."""
        result = self.sentiment_analyzer(text)[0]
        return {
            "text": text,
            "label": result["label"],
            "confidence": round(result["score"] * 100, 2)
        }

    def generate_text(self, prompt, max_length=100, num_results=1):
        """Generate text based on a prompt."""
        results = self.text_generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_results,
            truncation=True
        )
        return [r["generated_text"] for r in results]

    def summarize(self, text, max_length=130, min_length=30):
        """Summarize a long piece of text."""
        result = self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )
        return result[0]["summary_text"]

    def detect_entities(self, text):
        """Detect named entities in text."""
        entities = self.ner_detector(text)
        return [
            {
                "entity": e["entity_group"],
                "word": e["word"],
                "confidence": round(e["score"] * 100, 2)
            }
            for e in entities
        ]


# ── Quick Demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🧠 Text Processor — Full Demo")
    print("=" * 55)

    # Show the architecture explanation
    TransformerClassifier.explain_architecture()

    # Demo custom Transformer
    print("\n📐 Custom TransformerClassifier (for learning):")
    print("-" * 50)
    model = TransformerClassifier(num_classes=2)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total:,}")
    print(f"   Layers: Embedding → Self-Attention(×2) → Dense+ReLU → Dense")
    print(f"   Activations: GELU (feed-forward), Softmax (attention), ReLU (classifier)")

    # Test with dummy input
    dummy_ids = torch.randint(0, 1000, (2, 32))   # 2 sentences, 32 tokens
    dummy_mask = torch.ones(2, 32)
    output = model(dummy_ids, dummy_mask)
    probs = torch.softmax(output, dim=1)
    print(f"\n   Dummy input shape:  {dummy_ids.shape}")
    print(f"   Output logits:      {output.shape}")
    print(f"   Probabilities:      {probs[0].detach().numpy()}")

    # Demo pre-trained model (Hugging Face)
    print(f"\n\n📝 Pre-trained Model (Hugging Face):")
    print("-" * 50)
    processor = TextProcessor()

    texts = [
        "I love building neural networks!",
        "This code has too many bugs.",
        "The neuro brain project is amazing!",
    ]

    print("\n  Sentiment Analysis:")
    for text in texts:
        result = processor.analyze_sentiment(text)
        emoji = "😊" if result["label"] == "POSITIVE" else "😞"
        print(f"    {emoji} \"{result['text']}\"")
        print(f"       → {result['label']} ({result['confidence']}%)\n")
