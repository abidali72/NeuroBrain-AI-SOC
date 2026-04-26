# 🧠 Neuro Brain — App Integration Guide

> How to take your trained neural network and use it in a real desktop or mobile application.

---

## 📋 Deployment Overview

```
TRAINED MODEL
     │
     ├──→ Desktop App (Python/Tkinter, Electron, C#)
     │       └─ Use: .keras, .pth, or .onnx
     │
     ├──→ Web App (Flask, FastAPI, Node.js)
     │       └─ Use: .onnx or REST API
     │
     ├──→ Android App (Kotlin/Java)
     │       └─ Use: .tflite or .onnx
     │
     └──→ iOS App (Swift)
             └─ Use: .tflite or CoreML (.mlmodel)
```

---

## 1. Save & Export Formats

```python
from src.model_manager import ModelManager

manager = ModelManager()

# Save in framework-native format
manager.save_number_predictor(tf_model)          # → .keras
manager.save_object_detector(pytorch_model)       # → .pth

# Export for cross-platform apps
manager.export_to_onnx(pytorch_model, "detector") # → .onnx (any platform)
manager.export_to_tflite(tf_model, "predictor")   # → .tflite (mobile)
```

| Format | Best For | File Size | Supported Platforms |
|--------|----------|-----------|-------------------|
| `.keras` | Python desktop | Medium | Python (TensorFlow) |
| `.pth` | Python desktop | Small | Python (PyTorch) |
| `.onnx` | Cross-platform | Medium | Python, C++, C#, Java, JS, Rust |
| `.tflite` | Mobile | Small | Android, iOS, Flutter, embedded |
| CoreML | iOS | Small | Swift/Obj-C only |

---

## 2. Desktop App — Python (Tkinter)

The fastest path: build a GUI directly in Python using your model.

```python
"""
🧠 Neuro Brain — Desktop App (Tkinter)
Standalone desktop application with GUI.
Run: python desktop_app.py
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

# Load your models
from src.model_manager import QuickPredictor
predictor = QuickPredictor()


class NeuroBrainApp:
    def __init__(self, root):
        root.title("🧠 Neuro Brain")
        root.geometry("600x500")
        root.configure(bg='#1a1a2e')

        # Title
        title = tk.Label(root, text="🧠 Neuro Brain", font=('Arial', 24, 'bold'),
                        bg='#1a1a2e', fg='#e94560')
        title.pack(pady=20)

        # Notebook (tabs)
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # Tab 1: Number Prediction
        num_frame = tk.Frame(notebook, bg='#16213e')
        notebook.add(num_frame, text=' 🔢 Numbers ')
        self._build_number_tab(num_frame)

        # Tab 2: Text Analysis
        text_frame = tk.Frame(notebook, bg='#16213e')
        notebook.add(text_frame, text=' 📝 Text ')
        self._build_text_tab(text_frame)

    def _build_number_tab(self, parent):
        tk.Label(parent, text="Enter features (comma-separated):",
                font=('Arial', 12), bg='#16213e', fg='white').pack(pady=10)

        self.num_input = tk.Entry(parent, font=('Arial', 14), width=40)
        self.num_input.insert(0, "1500, 3, 10, 5.0")
        self.num_input.pack(pady=5)

        tk.Button(parent, text="Predict", font=('Arial', 12, 'bold'),
                 bg='#e94560', fg='white', command=self._predict_number
                 ).pack(pady=10)

        self.num_result = tk.Label(parent, text="", font=('Arial', 16),
                                  bg='#16213e', fg='#00ff88')
        self.num_result.pack(pady=10)

    def _build_text_tab(self, parent):
        tk.Label(parent, text="Enter text to analyze:",
                font=('Arial', 12), bg='#16213e', fg='white').pack(pady=10)

        self.text_input = tk.Text(parent, font=('Arial', 12), height=3, width=45)
        self.text_input.insert('1.0', "This product is amazing!")
        self.text_input.pack(pady=5)

        tk.Button(parent, text="Analyze Sentiment", font=('Arial', 12, 'bold'),
                 bg='#e94560', fg='white', command=self._analyze_text
                 ).pack(pady=10)

        self.text_result = tk.Label(parent, text="", font=('Arial', 16),
                                   bg='#16213e', fg='#00ff88')
        self.text_result.pack(pady=10)

    def _predict_number(self):
        try:
            features = [float(x.strip()) for x in self.num_input.get().split(',')]
            result = predictor.predict_number(features)
            self.num_result.config(text=f"Prediction: {result:,.2f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _analyze_text(self):
        try:
            text = self.text_input.get('1.0', 'end').strip()
            result = predictor.predict_text(text)
            emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
            self.text_result.config(
                text=f"{emoji} {result['label']} ({result['confidence']:.1%})"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroBrainApp(root)
    root.mainloop()
```

---

## 3. Desktop App — Electron (JavaScript)

For a modern cross-platform desktop app (Windows, Mac, Linux):

### Step 1: Export model to ONNX
```python
manager.export_to_onnx(model, "detector", input_shape=(1, 3, 32, 32))
```

### Step 2: Use ONNX Runtime in Node.js
```javascript
// In your Electron app
const ort = require('onnxruntime-node');

async function predict(imageData) {
    const session = await ort.InferenceSession.create('./model.onnx');

    const inputTensor = new ort.Tensor('float32', imageData, [1, 3, 32, 32]);
    const results = await session.run({ input: inputTensor });
    const output = results.output.data;

    // Find class with highest score
    const maxIdx = output.indexOf(Math.max(...output));
    const classes = ['airplane','automobile','bird','cat','deer',
                     'dog','frog','horse','ship','truck'];
    return classes[maxIdx];
}
```

---

## 4. Web App — FastAPI (REST API)

Serve your model as an API that any app can call:

```python
"""
🧠 Neuro Brain — REST API Server
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np

from src.model_manager import QuickPredictor

app = FastAPI(title="🧠 Neuro Brain API")
predictor = QuickPredictor()


class NumberInput(BaseModel):
    features: list[float]

class TextInput(BaseModel):
    text: str


@app.post("/predict/number")
async def predict_number(data: NumberInput):
    """Predict a number from features."""
    result = predictor.predict_number(data.features)
    return {"prediction": result}


@app.post("/predict/text")
async def predict_text(data: TextInput):
    """Analyze text sentiment."""
    result = predictor.predict_text(data.text)
    return result


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Classify an uploaded image."""
    import cv2, tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(await file.read())
        results = predictor.predict_image(f.name)
    return {"predictions": results}
```

### Call from ANY app:
```bash
# Number prediction
curl -X POST http://localhost:8000/predict/number \
  -H "Content-Type: application/json" \
  -d '{"features": [1500, 3, 10, 5.0]}'

# Text sentiment
curl -X POST http://localhost:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

---

## 5. Android App (Kotlin + TFLite)

### Step 1: Export to TFLite
```python
manager.export_to_tflite(tf_model, "number_predictor")
# Output: data/models/number_predictor.tflite
```

### Step 2: Add to Android project
```
app/src/main/assets/
└── number_predictor.tflite     ← copy your .tflite file here
```

### Step 3: Add dependency (build.gradle)
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### Step 4: Run inference in Kotlin
```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class NeuroBrainPredictor(private val context: Context) {

    private val interpreter: Interpreter by lazy {
        val model = loadModelFile("number_predictor.tflite")
        Interpreter(model)
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.allocateDirect(bytes.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return buffer
    }

    fun predict(features: FloatArray): Float {
        // Input: [1, 4] array of features
        val input = arrayOf(features)
        // Output: [1, 1] predicted value
        val output = Array(1) { FloatArray(1) }

        interpreter.run(input, output)
        return output[0][0]
    }
}

// Usage in Activity:
val predictor = NeuroBrainPredictor(this)
val price = predictor.predict(floatArrayOf(1500f, 3f, 10f, 5f))
textView.text = "Predicted price: $${"%.0f".format(price)}"
```

---

## 6. iOS App (Swift + TFLite)

### Step 1: Same TFLite export as Android

### Step 2: Add via CocoaPods
```ruby
pod 'TensorFlowLiteSwift'
```

### Step 3: Run inference in Swift
```swift
import TensorFlowLite

class NeuroBrainPredictor {
    private var interpreter: Interpreter

    init() throws {
        let modelPath = Bundle.main.path(forResource: "number_predictor",
                                          ofType: "tflite")!
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
    }

    func predict(features: [Float]) throws -> Float {
        var inputData = Data(buffer: UnsafeBufferPointer(
            start: features, count: features.count))

        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.invoke()

        let outputTensor = try interpreter.output(at: 0)
        let outputData = outputTensor.data
        let result = outputData.withUnsafeBytes { $0.load(as: Float.self) }
        return result
    }
}

// Usage:
let predictor = try NeuroBrainPredictor()
let price = try predictor.predict(features: [1500, 3, 10, 5])
print("Predicted: \(price)")
```

---

## 7. Architecture Decision Guide

```
What kind of app are you building?
│
├── Python Desktop (fastest) ──→ Load .keras/.pth directly
│   └── Framework: Tkinter, PyQt, or Kivy
│
├── Cross-Platform Desktop ──→ Export to .onnx
│   └── Framework: Electron + ONNX Runtime
│
├── Web App ──→ FastAPI/Flask REST API
│   └── Frontend calls API via HTTP
│
├── Android ──→ Export to .tflite
│   └── TensorFlow Lite + Kotlin/Java
│
├── iOS ──→ Export to .tflite or CoreML
│   └── TF Lite Swift or CoreML
│
└── Flutter (both) ──→ Export to .tflite
    └── tflite_flutter package
```

### Performance Comparison

| Platform | Format | Inference Speed | Ease of Setup |
|----------|--------|----------------|--------------|
| Python desktop | `.keras`/`.pth` | ~10-50ms | ⭐⭐⭐ Easiest |
| Electron | `.onnx` | ~5-20ms | ⭐⭐ Moderate |
| REST API | any | ~50-200ms (+ network) | ⭐⭐ Moderate |
| Android | `.tflite` | ~5-30ms | ⭐ Complex |
| iOS | `.tflite`/CoreML | ~5-20ms | ⭐ Complex |

---

*Your models are ready to deploy! Start with the Python desktop app for the fastest results. 🚀*
