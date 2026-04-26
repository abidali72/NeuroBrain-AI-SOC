"""
🧠 Neuro Brain — Model Save, Load & Export Utilities
Unified module for persisting and loading all three neural networks.

Supports:
  • TensorFlow .keras format (Number Predictor)
  • PyTorch .pth state_dict format (Object Detector)
  • Hugging Face save_pretrained (Text Processor)
  • ONNX export for cross-platform deployment
  • TFLite export for mobile deployment

Usage:
    from src.model_manager import ModelManager
    manager = ModelManager()

    # Save after training
    manager.save_number_predictor(model)
    manager.save_object_detector(model)

    # Load for inference
    model = manager.load_number_predictor()
    model = manager.load_object_detector()

    # Export for apps
    manager.export_to_onnx(model, "number_predictor")
    manager.export_to_tflite(tf_model, "number_predictor")
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime


class ModelManager:
    """
    Central manager for saving, loading, and exporting neural network models.

    Save Formats Explained:
    ─────────────────────────────────────────────────────────────
    FORMAT           USE CASE                    SIZE     SPEED
    ─────────────────────────────────────────────────────────────
    .keras           TensorFlow models           Medium   Fast
    .pth             PyTorch weights only         Small    Fast
    .pt              PyTorch full model           Medium   Fast
    .onnx            Cross-platform deployment    Medium   Fastest
    .tflite          Android/iOS mobile apps      Small    Fast
    save_pretrained  Hugging Face Transformers    Large    Medium
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self, models_dir="data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self):
        """Load or create model registry."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}}

    def _save_registry(self):
        """Persist registry to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _register(self, name, path, framework, format_type, metadata=None):
        """Register a saved model in the registry."""
        self.registry["models"][name] = {
            "path": str(path),
            "framework": framework,
            "format": format_type,
            "saved_at": datetime.now().isoformat(),
            "file_size_mb": round(self._get_size_mb(path), 2),
            **(metadata or {})
        }
        self._save_registry()

    def _get_size_mb(self, path):
        """Get file/directory size in MB."""
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        elif path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total / (1024 * 1024)
        return 0

    # ════════════════════════════════════════════════════════════
    #  TENSORFLOW / KERAS  — Save & Load
    # ════════════════════════════════════════════════════════════

    def save_number_predictor(self, model, name="number_predictor"):
        """
        Save a TensorFlow/Keras model.

        Three save formats available:
        ─────────────────────────────────────────────────────────
        1. .keras (recommended) — Full model: architecture + weights + optimizer
           Use when: you want to resume training later or deploy with TF.
           model.save("model.keras")

        2. Weights only (.weights.h5) — Just the learned parameters
           Use when: you have the model code and only need the weights.
           model.save_weights("weights.weights.h5")

        3. SavedModel directory — TensorFlow's native format
           Use when: deploying with TensorFlow Serving or TF.js.
           model.save("saved_model/")
        ─────────────────────────────────────────────────────────
        """
        # Method 1: Full model (.keras format — RECOMMENDED)
        keras_path = self.models_dir / f"{name}.keras"
        model.save(str(keras_path))
        print(f"💾 Saved Keras model  → {keras_path}")

        # Method 2: Weights only (lighter, need model code to reload)
        weights_path = self.models_dir / f"{name}.weights.h5"
        model.save_weights(str(weights_path))
        print(f"💾 Saved weights      → {weights_path}")

        self._register(name, keras_path, "tensorflow", "keras", {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "parameters": model.count_params(),
        })

        print(f"   File size: {self._get_size_mb(keras_path):.2f} MB")
        print(f"   Parameters: {model.count_params():,}")
        return keras_path

    def load_number_predictor(self, name="number_predictor"):
        """
        Load a saved TensorFlow/Keras model.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow not installed. Cannot load Number Predictor.")

        keras_path = self.models_dir / f"{name}.keras"

        if not keras_path.exists():
            raise FileNotFoundError(
                f"No saved model at {keras_path}. Train the model first with:\n"
                f"  python -m src.train --task numbers"
            )

        start = time.time()
        model = keras.models.load_model(str(keras_path))
        load_time = time.time() - start

        print(f"📂 Loaded Keras model ← {keras_path}")
        print(f"   Load time: {load_time*1000:.0f}ms")
        print(f"   Parameters: {model.count_params():,}")
        return model

    # ════════════════════════════════════════════════════════════
    #  PYTORCH — Save & Load
    # ════════════════════════════════════════════════════════════

    def save_object_detector(self, model, name="object_detector"):
        """
        Save a PyTorch model.

        Two save strategies:
        ─────────────────────────────────────────────────────────
        1. state_dict only (RECOMMENDED by PyTorch)
           Saves learned weights/biases, NOT the model architecture.
           + Smaller files, more flexible, version-safe.
           - Requires model class code to rebuild architecture.
           torch.save(model.state_dict(), "weights.pth")

        2. Full model (pickle)
           Saves everything: architecture + weights + optimizer.
           + One file, no code needed to load.
           - Fragile: breaks if you rename classes or move files.
           torch.save(model, "full_model.pt")
        ─────────────────────────────────────────────────────────
        """
        import torch

        # Method 1: State dict (RECOMMENDED)
        weights_path = self.models_dir / f"{name}.pth"
        torch.save(model.state_dict(), str(weights_path))
        print(f"💾 Saved PyTorch weights → {weights_path}")

        # Method 2: Full model (backup)
        full_path = self.models_dir / f"{name}_full.pt"
        torch.save(model, str(full_path))
        print(f"💾 Saved full model      → {full_path}")

        total_params = sum(p.numel() for p in model.parameters())
        self._register(name, weights_path, "pytorch", "state_dict", {
            "parameters": total_params,
        })

        print(f"   File size: {self._get_size_mb(weights_path):.2f} MB")
        print(f"   Parameters: {total_params:,}")
        return weights_path

    def load_object_detector(self, name="object_detector", device=None):
        """
        Load a saved PyTorch model.

        Since we saved state_dict, we need to:
        1. Create the model architecture (from our class)
        2. Load the saved weights into it
        """
        import torch
        from src.models.object_detector import NeuroBrainCNN

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        weights_path = self.models_dir / f"{name}.pth"
        best_path = self.models_dir / f"{name}_best.pth"

        load_path = best_path if best_path.exists() else weights_path

        if not load_path.exists():
            raise FileNotFoundError(
                f"No saved model at {load_path}. Train the model first with:\n"
                f"  python -m src.train --task images"
            )

        # Step 1: Rebuild architecture
        model = NeuroBrainCNN(num_classes=10).to(device)

        # Step 2: Load weights into architecture
        start = time.time()
        model.load_state_dict(torch.load(str(load_path), map_location=device))
        load_time = time.time() - start

        model.eval()  # Set to inference mode (disables dropout)

        print(f"📂 Loaded PyTorch model ← {load_path}")
        print(f"   Load time: {load_time*1000:.0f}ms")
        print(f"   Device: {device}")
        return model

    # ════════════════════════════════════════════════════════════
    #  HUGGING FACE — Save & Load
    # ════════════════════════════════════════════════════════════

    def save_text_processor(self, model, tokenizer, name="text_processor"):
        """
        Save a Hugging Face Transformer model.

        save_pretrained() creates a directory with:
        ─────────────────────────────────────────────────────────
        text_processor/
        ├── config.json          ← Model architecture config
        ├── model.safetensors    ← Weights (safe binary format)
        ├── tokenizer.json       ← Tokenizer vocabulary
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── vocab.txt            ← Word vocabulary
        ─────────────────────────────────────────────────────────
        """
        save_dir = self.models_dir / name
        save_dir.mkdir(exist_ok=True)

        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

        self._register(name, save_dir, "huggingface", "save_pretrained")

        print(f"💾 Saved Transformer   → {save_dir}/")
        print(f"   Directory size: {self._get_size_mb(save_dir):.2f} MB")
        return save_dir

        return model, tokenizer

    # ════════════════════════════════════════════════════════════
    #  CYBERSECURITY MODELS — Save & Load
    # ════════════════════════════════════════════════════════════

    def save_intrusion_detector(self, model, name="intrusion_detector"):
        """Save NIDS model weights."""
        import torch
        weights_path = self.models_dir / f"{name}.pth"
        torch.save(model.state_dict(), str(weights_path))
        print(f"💾 Saved IDS weights  → {weights_path}")
        return weights_path

    def load_intrusion_detector(self, name="intrusion_detector", device=None):
        """Load NIDS model."""
        import torch
        from src.models.intrusion_detector import IntrusionDetectionNet
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = IntrusionDetectionNet(input_dim=12).to(device)
        path = self.models_dir / f"{name}.pth"
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.eval()
        print(f"📂 Loaded IDS weights ← {path}")
        return model

    def save_waf_model(self, model, vectorizer, name="waf_ai"):
        """Save WAF model weights and TF-IDF vectorizer."""
        import torch
        import joblib
        weights_path = self.models_dir / f"{name}.pth"
        vec_path = self.models_dir / f"waf_vectorizer.pkl"
        torch.save(model.state_dict(), str(weights_path))
        joblib.dump(vectorizer, str(vec_path))
        print(f"💾 Saved WAF weights  → {weights_path}")
        print(f"💾 Saved WAF Vectorizer → {vec_path}")
        return weights_path

    def load_waf_model(self, name="waf_ai", device=None):
        """Load WAF model and its vectorizer."""
        import torch
        import joblib
        from src.models.waf_ai import WAFNeuralNet
        
        vec_path = self.models_dir / "waf_vectorizer.pkl"
        vectorizer = joblib.load(str(vec_path))
        input_dim = len(vectorizer.get_feature_names_out())
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model = WAFNeuralNet(input_dim=input_dim).to(device)
        weights_path = self.models_dir / f"{name}.pth"
        model.load_state_dict(torch.load(str(weights_path), map_location=device))
        model.eval()
        print(f"📂 Loaded WAF weights ← {weights_path}")
        return model, vectorizer

    def save_memory_forensics(self, model, name="memory_forensics"):
        """Save Memory Forensics CNN weights."""
        import torch
        weights_path = self.models_dir / f"{name}.pth"
        torch.save(model.state_dict(), str(weights_path))
        print(f"💾 Saved Forensics   → {weights_path}")
        return weights_path

    def load_memory_forensics(self, name="memory_forensics", device=None):
        """Load Memory Forensics CNN."""
        import torch
        from src.models.memory_forensics import MemoryForensicsCNN
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        model = MemoryForensicsCNN().to(device)
        path = self.models_dir / f"{name}.pth"
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.eval()
        print(f"📂 Loaded Forensics  ← {path}")
        return model

    def save_cyber_reasoner(self, model, tokenizer, name="cyber_reasoner"):
        """
        Save the fine-tuned GPT-2 model and tokenizer.
        """
        save_dir = self.models_dir / name
        save_dir.mkdir(exist_ok=True)
        
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        
        self._register(name, save_dir, "huggingface", "gpt2_finetuned")
        print(f"💾 Saved Cyber Reasoner → {save_dir}/")
        return save_dir

    def load_cyber_reasoner(self, name="cyber_reasoner", device=None):
        """
        Load the fine-tuned GPT-2 model and tokenizer.
        """
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch
        
        save_dir = self.models_dir / name
        if not save_dir.exists():
            raise FileNotFoundError(f"No saved model found at {save_dir}")
            
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        tokenizer = GPT2Tokenizer.from_pretrained(str(save_dir))
        model = GPT2LMHeadModel.from_pretrained(str(save_dir)).to(device)
        model.eval()
        
        print(f"📂 Loaded Cyber Reasoner ← {save_dir}/")
        return model, tokenizer

    # ════════════════════════════════════════════════════════════
    #  EXPORT: ONNX (Cross-Platform Deployment)
    # ════════════════════════════════════════════════════════════

    def export_to_onnx(self, pytorch_model, name, input_shape=(1, 3, 32, 32)):
        """
        Export a PyTorch model to ONNX format.

        ONNX (Open Neural Network Exchange):
        ─────────────────────────────────────────────────────────
        • Universal format readable by any framework
        • Runs on: C++, C#, Java, JavaScript, Python, Rust
        • Optimized with ONNX Runtime for fast inference
        • Perfect for: Desktop apps, web apps, game engines
        ─────────────────────────────────────────────────────────
        """
        import torch

        pytorch_model.eval()
        onnx_path = self.models_dir / f"{name}.onnx"
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            pytorch_model,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13
        )

        self._register(f"{name}_onnx", onnx_path, "onnx", "onnx")
        print(f"💾 Exported ONNX model → {onnx_path}")
        print(f"   File size: {self._get_size_mb(onnx_path):.2f} MB")
        return onnx_path

    # ════════════════════════════════════════════════════════════
    #  EXPORT: TFLite (Mobile Deployment)
    # ════════════════════════════════════════════════════════════

    def export_to_tflite(self, tf_model, name):
        """
        Export a TensorFlow model to TFLite format.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Cannot export to TFLite.")

        tflite_path = self.models_dir / f"{name}.tflite"

        # Convert with optimizations
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize for speed
        tflite_model = converter.convert()

        with open(str(tflite_path), 'wb') as f:
            f.write(tflite_model)

        self._register(f"{name}_tflite", tflite_path, "tflite", "tflite")
        print(f"💾 Exported TFLite     → {tflite_path}")
        print(f"   File size: {self._get_size_mb(tflite_path):.2f} MB")
        return tflite_path

    # ════════════════════════════════════════════════════════════
    #  UTILITIES
    # ════════════════════════════════════════════════════════════

    def list_models(self):
        """List all saved models."""
        print(f"\n{'─'*60}")
        print(f"  📦 SAVED MODELS")
        print(f"{'─'*60}")

        if not self.registry["models"]:
            print("   No saved models found.")
            print(f"   Train a model with: python -m src.train --task all")
            return

        for name, info in self.registry["models"].items():
            print(f"\n   📁 {name}")
            print(f"      Framework: {info.get('framework', 'unknown')}")
            print(f"      Format:    {info.get('format', 'unknown')}")
            print(f"      Path:      {info.get('path', 'unknown')}")
            print(f"      Size:      {info.get('file_size_mb', 0):.2f} MB")
            print(f"      Saved:     {info.get('saved_at', 'unknown')}")
            if 'parameters' in info:
                print(f"      Params:    {info['parameters']:,}")

        print(f"\n{'─'*60}")


# ═══════════════════════════════════════════════════════════════
#  QUICK INFERENCE: Load and Predict in One Call
# ═══════════════════════════════════════════════════════════════

class QuickPredictor:
    """
    One-line prediction from saved models.
    Designed for integration into applications.

    Usage:
        predictor = QuickPredictor()

        # Number prediction
        price = predictor.predict_number([1500, 3, 10, 5.0])

        # Image classification
        label = predictor.predict_image("photo.jpg")

        # Sentiment analysis
        sentiment = predictor.predict_text("This product is great!")
    """

    def __init__(self, models_dir="data/models"):
        self.manager = ModelManager(models_dir)
        self._number_model = None
        self._object_model = None
        self._text_pipeline = None

    def predict_number(self, features):
        """Predict a number from input features."""
        import numpy as np

        if self._number_model is None:
            try:
                self._number_model = self.manager.load_number_predictor()
            except Exception as e:
                print(f"[!] Error loading Number Predictor: {e}")
                return 0.0

        X = np.array(features).reshape(1, -1)
        # Handle case where model is Keras vs PyTorch
        if hasattr(self._number_model, 'predict'): # Keras
            prediction = self._number_model.predict(X, verbose=0)
            result = float(prediction[0][0])
        else: # PyTorch
            import torch
            with torch.no_grad():
                tensor = torch.FloatTensor(X)
                output = self._number_model(tensor)
                result = output.item()
                
        print(f"🔢 Prediction: {result:.2f}")
        return result

    def predict_image(self, image_path, top_k=3):
        """Classify an image and return top-k predictions."""
        import torch
        import cv2
        import numpy as np
        from torchvision import transforms

        if self._object_model is None:
            self._object_model = self.manager.load_object_detector()

        CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

        # Load and preprocess image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ])
        tensor = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = self._object_model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            top_probs, top_indices = probs.topk(top_k)

        results = []
        print(f"👁️ Image: {image_path}")
        for prob, idx in zip(top_probs, top_indices):
            label = CLASSES[idx]
            confidence = prob.item()
            results.append({"label": label, "confidence": confidence})
            print(f"   {label}: {confidence:.1%}")

        return results

    def predict_text(self, text):
        """Analyze sentiment of text."""
        try:
            from transformers import pipeline
            if self._text_pipeline is None:
                self._text_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

            result = self._text_pipeline(text)[0]
            sentiment = result['label']
            confidence = result['score']

            emoji = "😊" if sentiment == "POSITIVE" else "😞"
            print(f"📝 {emoji} {sentiment} ({confidence:.1%})")
            return {"label": sentiment, "confidence": confidence}
        except Exception as e:
            print(f"[!] Sentiment analysis error: {e}")
            return {"label": "UNKNOWN", "confidence": 0.0}

    # ════════════════════════════════════════════════════════════
    #  CYBERSECURITY PREDICTIONS
    # ════════════════════════════════════════════════════════════

    def predict_intrusion(self, features):
        """Detect network intrusion from telemetry."""
        import torch
        if self._object_model is None: # Reusing Slot or use unique ones
            self._ids_model = self.manager.load_intrusion_detector()
        
        # In QuickPredictor, let's use dedicated private vars
        if not hasattr(self, '_ids_model') or self._ids_model is None:
            self._ids_model = self.manager.load_intrusion_detector()
            
        tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            output = self._ids_model(tensor)
            prob = output.item()
            
        label = "Malicious" if prob > 0.5 else "Benign"
        return {"label": label, "confidence": prob if prob > 0.5 else 1 - prob, "score": prob}

    def predict_waf(self, query):
        """Detect SQLi/XSS in HTTP query."""
        import torch
        if not hasattr(self, '_waf_model') or self._waf_model is None:
            self._waf_model, self._waf_vec = self.manager.load_waf_model()
            
        X_tfidf = self._waf_vec.transform([query]).toarray()
        tensor = torch.FloatTensor(X_tfidf)
        
        with torch.no_grad():
            outputs = self._waf_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred = torch.max(probs, 0)
            
        labels = ['Benign', 'SQLi', 'XSS']
        return {"label": labels[pred.item()], "confidence": conf.item()}

    def predict_forensics(self, memory_page):
        """Detect shellcode in RAM segment."""
        import torch
        if not hasattr(self, '_forensics_model') or self._forensics_model is None:
            self._forensics_model = self.manager.load_memory_forensics()
            
        # Expecting memory_page as 64x64 numpy array or flattened 4096 list
        img = torch.FloatTensor(memory_page).reshape(1, 1, 64, 64)
        with torch.no_grad():
            output = self._forensics_model(img)
            prob = output.item()
            
        label = "Malicious" if prob > 0.5 else "Benign"
        return {"label": label, "confidence": prob if prob > 0.5 else 1 - prob, "score": prob}


# ── Demo & CLI ─────────────────────────────────────────────────
if __name__ == "__main__":
    manager = ModelManager()
    manager.list_models()

    print("\n📖 Save/Load examples:")
    print("""
    from src.model_manager import ModelManager, QuickPredictor

    # === SAVE after training ===
    manager = ModelManager()
    manager.save_number_predictor(model)
    manager.save_object_detector(model)
    manager.save_text_processor(model, tokenizer)

    # === LOAD for inference ===
    model = manager.load_number_predictor()
    model = manager.load_object_detector()
    model, tokenizer = manager.load_text_processor()

    # === EXPORT for apps ===
    manager.export_to_onnx(pytorch_model, "object_detector")
    manager.export_to_tflite(tf_model, "number_predictor")

    # === QUICK PREDICT (one-liner) ===
    predictor = QuickPredictor()
    predictor.predict_number([1500, 3, 10, 5.0])
    predictor.predict_text("This is amazing!")
    """)
