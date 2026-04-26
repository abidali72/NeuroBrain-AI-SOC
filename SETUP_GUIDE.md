# 🧠 Neuro Brain — Complete Python Setup Guide

> A comprehensive guide to building your personal AI neuro brain from scratch.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Python Installation](#2-python-installation)
3. [Virtual Environment Setup](#3-virtual-environment-setup)
4. [Core Libraries](#4-core-libraries)
5. [Installation Commands](#5-installation-commands)
6. [Project Structure](#6-project-structure)
7. [Verify Installation](#7-verify-installation)
8. [Quick Start Examples](#8-quick-start-examples)
9. [GPU Setup (Optional)](#9-gpu-setup-optional)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement       | Minimum          | Recommended       |
|-------------------|------------------|--------------------|
| **OS**            | Windows 10       | Windows 11         |
| **Python**        | 3.9              | 3.10 or 3.11       |
| **RAM**           | 8 GB             | 16 GB+             |
| **Disk Space**    | 10 GB free       | 20 GB+ free        |
| **GPU (optional)**| NVIDIA GTX 1060  | NVIDIA RTX 3060+   |

---

## 2. Python Installation

### Step 1: Download Python
- Go to [python.org/downloads](https://www.python.org/downloads/)
- Download **Python 3.11.x** (recommended)

### Step 2: Install Python
```
⚠️ IMPORTANT: Check "Add Python to PATH" during installation!
```
- Run the installer
- ✅ Check **"Add python.exe to PATH"**
- Click **"Install Now"**

### Step 3: Verify Installation
Open **PowerShell** or **Command Prompt** and run:
```powershell
python --version
pip --version
```
Expected output:
```
Python 3.11.x
pip 24.x.x
```

---

## 3. Virtual Environment Setup

A virtual environment isolates your project's dependencies from other Python projects.

### Create the project folder and virtual environment:
```powershell
# Navigate to your brain project
cd C:\Users\HP\OneDrive\Desktop\brain

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

> **Note:** If you get an execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Verify activation:
Your terminal prompt should now show `(venv)` at the beginning:
```
(venv) C:\Users\HP\OneDrive\Desktop\brain>
```

### Deactivate (when done working):
```powershell
deactivate
```

---

## 4. Core Libraries

Here's every library you'll need, organized by purpose:

### 🔧 Foundation Libraries

| Library     | Version  | Purpose                                      |
|-------------|----------|----------------------------------------------|
| **NumPy**   | ≥1.24    | Array operations, mathematical computing     |
| **Pandas**  | ≥2.0     | Data loading, manipulation, and analysis     |
| **Matplotlib** | ≥3.7  | Plotting, visualizations, training graphs    |
| **Seaborn** | ≥0.12   | Statistical data visualization               |
| **Scikit-learn** | ≥1.3 | Data preprocessing, metrics, classic ML     |

### 🧠 Deep Learning Frameworks

| Library        | Version  | Purpose                                    |
|----------------|----------|--------------------------------------------|
| **TensorFlow** | ≥2.15   | Neural network building & training (Google)|
| **Keras**      | (bundled)| High-level API for TensorFlow              |
| **PyTorch**    | ≥2.1    | Neural network building & training (Meta)  |
| **Torchvision**| ≥0.16   | Image datasets, models, transforms         |
| **Torchaudio** | ≥2.1    | Audio processing with PyTorch              |

### 👁️ Computer Vision

| Library    | Version | Purpose                                       |
|------------|---------|-----------------------------------------------|
| **OpenCV** | ≥4.8   | Image/video processing, camera input          |
| **Pillow** | ≥10.0  | Image file loading and manipulation           |

### 📝 Natural Language Processing

| Library          | Version | Purpose                                  |
|------------------|---------|------------------------------------------|
| **Transformers** | ≥4.35  | Pre-trained NLP models (BERT, GPT, etc.) |
| **Tokenizers**   | ≥0.15  | Fast text tokenization                   |
| **NLTK**         | ≥3.8   | Classic NLP tools and datasets           |

### 🛠️ Utilities

| Library      | Version | Purpose                                     |
|--------------|---------|---------------------------------------------|
| **Jupyter**  | ≥1.0   | Interactive notebooks for experimentation   |
| **tqdm**     | ≥4.66  | Progress bars for training loops            |
| **tensorboard** | ≥2.15 | Training visualization dashboard          |
| **python-dotenv** | ≥1.0 | Environment variable management           |

---

## 5. Installation Commands

### Option A: Install Everything at Once (Recommended)

First, make sure your virtual environment is activated, then run:

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Foundation
pip install numpy pandas matplotlib seaborn scikit-learn

# TensorFlow (includes Keras)
pip install tensorflow

# PyTorch (CPU version — see GPU section below for CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Computer Vision
pip install opencv-python Pillow

# NLP
pip install transformers tokenizers nltk

# Utilities
pip install jupyter tqdm tensorboard python-dotenv
```

### Option B: Install from requirements.txt

Create or use the provided `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Option C: Install Only What You Need

| Task               | Command                                                             |
|--------------------|---------------------------------------------------------------------|
| **Number Prediction** | `pip install numpy pandas matplotlib scikit-learn tensorflow`     |
| **Object Recognition**| `pip install numpy matplotlib opencv-python torch torchvision Pillow` |
| **Text Processing**   | `pip install numpy pandas transformers tokenizers nltk torch`    |

---

## 6. Project Structure

```
brain/
├── venv/                       # Virtual environment (auto-generated)
├── data/                       # Datasets
│   ├── raw/                    # Original unprocessed data
│   ├── processed/              # Cleaned and prepared data
│   └── models/                 # Saved trained models
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── 01_number_prediction.ipynb
│   ├── 02_object_recognition.ipynb
│   └── 03_text_processing.ipynb
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration and hyperparameters
│   ├── data_loader.py          # Data loading utilities
│   ├── models/                 # Neural network architectures
│   │   ├── __init__.py
│   │   ├── number_predictor.py # FNN/LSTM for number prediction
│   │   ├── object_detector.py  # CNN for object recognition
│   │   └── text_processor.py   # Transformer for text processing
│   ├── train.py                # Training pipeline
│   ├── predict.py              # Inference/prediction pipeline
│   └── utils.py                # Helper functions
├── tests/                      # Unit tests
│   └── test_models.py
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys, etc.)
├── .gitignore                  # Git ignore rules
├── SETUP_GUIDE.md              # This guide
└── README.md                   # Project overview
```

---

## 7. Verify Installation

After installing all libraries, run this verification script to confirm everything works:

```python
# Save as: verify_setup.py
# Run with: python verify_setup.py

import sys

def check(name, import_name=None):
    try:
        mod = __import__(import_name or name)
        ver = getattr(mod, '__version__', '✓')
        print(f"  ✅ {name:.<25} {ver}")
        return True
    except ImportError:
        print(f"  ❌ {name:.<25} NOT INSTALLED")
        return False

print(f"\n🧠 Neuro Brain — Environment Check")
print(f"{'='*45}")
print(f"  Python ................. {sys.version.split()[0]}\n")

all_ok = True

print("📦 Foundation:")
all_ok &= check("numpy")
all_ok &= check("pandas")
all_ok &= check("matplotlib")
all_ok &= check("seaborn")
all_ok &= check("scikit-learn", "sklearn")

print("\n🧠 Deep Learning:")
all_ok &= check("tensorflow")
all_ok &= check("torch")
all_ok &= check("torchvision")

print("\n👁️ Computer Vision:")
all_ok &= check("opencv-python", "cv2")
all_ok &= check("Pillow", "PIL")

print("\n📝 NLP:")
all_ok &= check("transformers")
all_ok &= check("nltk")

print("\n🛠️ Utilities:")
all_ok &= check("jupyter")
all_ok &= check("tqdm")
all_ok &= check("tensorboard")

print(f"\n{'='*45}")
if all_ok:
    print("🎉 All libraries installed successfully!")
else:
    print("⚠️  Some libraries are missing. Install them with pip.")
print()
```

---

## 8. Quick Start Examples

### 🔢 Example 1: Number Prediction (FNN with TensorFlow)

```python
# src/models/number_predictor.py
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Generate sample data: predict y = 2x + 1
X_train = np.random.rand(1000, 1) * 100
y_train = 2 * X_train + 1 + np.random.randn(1000, 1) * 2

# Build the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=1)

# Predict
test_input = np.array([[25.0]])
prediction = model.predict(test_input)
print(f"\nInput: 25.0 → Predicted: {prediction[0][0]:.2f} (Expected: 51.0)")

# Save the model
model.save('data/models/number_predictor.keras')
```

---

### 👁️ Example 2: Object Recognition (CNN with PyTorch)

```python
# src/models/object_detector.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset (10 object classes)
trainset = torchvision.datasets.CIFAR10(
    root='./data/raw', train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Define CNN architecture
class NeuroBrainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuroBrainCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/10 — Loss: {running_loss/len(trainloader):.4f}")

# Save the model
torch.save(model.state_dict(), 'data/models/object_detector.pth')
print("✅ Model saved!")
```

---

### 📝 Example 3: Text Processing (Transformer with Hugging Face)

```python
# src/models/text_processor.py
from transformers import pipeline

# Sentiment Analysis — uses a pre-trained model
classifier = pipeline("sentiment-analysis")

texts = [
    "I love building neural networks!",
    "This code has too many bugs.",
    "The neuro brain project is amazing!",
    "I'm frustrated with the training loss."
]

print("📝 Sentiment Analysis Results:\n")
for text in texts:
    result = classifier(text)[0]
    emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
    print(f"  {emoji} \"{text}\"")
    print(f"     → {result['label']} ({result['score']:.2%})\n")


# Text Generation — uses GPT-2
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
output = generator(prompt, max_length=50, num_return_sequences=1)

print("🤖 Text Generation:\n")
print(f"  Prompt: \"{prompt}\"")
print(f"  Output: \"{output[0]['generated_text']}\"")
```

---

## 9. GPU Setup (Optional)

GPU acceleration can make training **10-50x faster**. This requires an NVIDIA GPU.

### Step 1: Check your GPU
```powershell
nvidia-smi
```

### Step 2: Install CUDA Toolkit
- Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- Install **CUDA 12.1** (recommended)

### Step 3: Install cuDNN
- Download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
- Extract and copy files to the CUDA directory

### Step 4: Install GPU versions of frameworks

```powershell
# TensorFlow (GPU support is included automatically since TF 2.x)
pip install tensorflow

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Verify GPU access
```python
# TensorFlow GPU check
import tensorflow as tf
print("TF GPUs:", tf.config.list_physical_devices('GPU'))

# PyTorch GPU check
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
```

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip` not recognized | Re-install Python with **"Add to PATH"** checked |
| `Activate.ps1` execution error | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| TensorFlow import fails | Ensure Python 3.9-3.11 (TF doesn't support 3.12+ yet) |
| PyTorch CUDA not detected | Verify `nvidia-smi` works, reinstall with correct CUDA version |
| `ModuleNotFoundError` | Make sure your venv is activated: `.\venv\Scripts\Activate.ps1` |
| Out of memory during training | Reduce `batch_size`, use `model.half()` in PyTorch, or use CPU |
| Slow training without GPU | Use smaller datasets/models, or set up GPU (see Section 9) |
| OpenCV `import cv2` fails | Try `pip install opencv-python-headless` instead |

---

## 🚀 Next Steps

Once your environment is ready, you can:

1. **Run the verification script** → `python verify_setup.py`
2. **Try the quick start examples** → Pick the task that interests you
3. **Launch Jupyter** → `jupyter notebook` and open the notebooks
4. **Build your custom model** → Modify the examples in `src/models/`
5. **Train with your own data** → Place datasets in `data/raw/`

---

*Happy building! 🧠⚡*
