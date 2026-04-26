"""
🧠 Neuro Brain — Configuration & Hyperparameters
Central configuration for all neural network models.
"""

import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Training Hyperparameters ───────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# ── Number Predictor (FNN) ────────────────────────────────────
NUMBER_PREDICTOR = {
    "hidden_layers": [64, 32],
    "activation": "relu",
    "optimizer": "adam",
    "loss": "mse",
}

# ── Object Detector (CNN) ─────────────────────────────────────
OBJECT_DETECTOR = {
    "input_size": (32, 32),
    "num_classes": 10,
    "conv_channels": [32, 64, 128],
    "dropout": 0.5,
}

# ── Text Processor (Transformer) ──────────────────────────────
TEXT_PROCESSOR = {
    "model_name": "distilbert-base-uncased",
    "max_length": 512,
    "task": "sentiment-analysis",
}

# ── Device Configuration ──────────────────────────────────────
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Neuro Brain initialized | Device: {DEVICE}")
