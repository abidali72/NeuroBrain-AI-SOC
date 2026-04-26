"""
🧠 Neuro Brain — Real-Time Input Processor
Process live data from camera, microphone, or sensors in real-time.

Architecture:
    Live Input → Thread-safe Queue → Preprocessor → Neural Network → Output

Key Design Decisions:
    • Threading: Input capture runs on a separate thread so the neural
      network never blocks the data stream.
    • Queue: Thread-safe buffer between capture and processing.
    • Batching: Small batches (1-4) for low latency, not throughput.
    • FPS Counter: Monitor if the pipeline is fast enough.

Usage:
    python -m src.models.realtime_processor
"""

import time
import threading
import queue
import numpy as np
from collections import deque


class FPSCounter:
    """Tracks frames/predictions per second."""

    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)

    def tick(self):
        self.timestamps.append(time.time())

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / max(elapsed, 1e-6)


class RealTimeProcessor:
    """
    Base class for real-time neural network processing.

    Pipeline:
    ─────────────────────────────────────────────────────────
    [Input Thread]              [Processing Thread]
     Camera/Mic/Sensor           Neural Network
         │                            │
         ▼                            ▼
    ┌──────────┐              ┌──────────────┐
    │ Capture  │──→ Queue ──→ │ Preprocess   │
    │ frame    │              │ → Predict    │
    │          │              │ → Callback   │
    └──────────┘              └──────────────┘
         30 FPS                   < 33ms/frame
    ─────────────────────────────────────────────────────────
    """

    def __init__(self, max_queue_size=30):
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.fps = FPSCounter()
        self.on_result = None  # Callback function for results

    def start(self):
        """Start the processing pipeline."""
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()
        print("🟢 Real-time pipeline started")

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        print(f"🔴 Pipeline stopped | Avg FPS: {self.fps.fps:.1f}")

    def _capture_loop(self):
        """Override in subclass to capture input data."""
        raise NotImplementedError

    def _process_loop(self):
        """Process items from the queue."""
        while self.running:
            try:
                data = self.input_queue.get(timeout=1.0)
                result = self.process(data)
                self.fps.tick()
                if self.on_result:
                    self.on_result(result)
            except queue.Empty:
                continue

    def process(self, data):
        """Override in subclass to run neural network."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
#  REAL-TIME CAMERA (Object Detection)
# ═══════════════════════════════════════════════════════════════

class RealtimeCamera(RealTimeProcessor):
    """
    Live webcam object classification using the CNN.

    Pipeline:
    Camera → capture frame → resize 32×32 → normalize →
    CNN → class probabilities → display/callback

    NOTE: Requires OpenCV and a connected webcam.
    """

    def __init__(self, camera_id=0, display=True):
        super().__init__()
        self.camera_id = camera_id
        self.display = display
        self.model = None
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    def _load_model(self):
        """Load the trained CNN model."""
        import torch
        from src.models.object_detector import NeuroBrainCNN

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuroBrainCNN(num_classes=10).to(self.device)

        import pathlib
        model_path = pathlib.Path("data/models/object_detector_best.pth")
        alt_path = pathlib.Path("data/models/object_detector.pth")

        if model_path.exists():
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        elif alt_path.exists():
            self.model.load_state_dict(torch.load(str(alt_path), map_location=self.device))
        else:
            print("⚠️  No trained model found. Using untrained model (predictions will be random).")

        self.model.eval()
        print(f"📦 CNN model loaded on {self.device}")

    def _capture_loop(self):
        """Capture frames from webcam."""
        import cv2

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            self.running = False
            return

        print(f"📷 Camera opened (ID: {self.camera_id})")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Drop old frames if queue is full (keep latest)
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
            self.input_queue.put(frame)

        cap.release()

    def process(self, frame):
        """Classify a camera frame."""
        import cv2
        import torch
        from torchvision import transforms

        if self.model is None:
            self._load_model()

        # Preprocess: resize to 32×32, normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
        ])
        tensor = transform(img).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, class_idx = probs.max(0)

        result = {
            'class': self.classes[class_idx.item()],
            'confidence': confidence.item(),
            'all_probs': {c: p.item() for c, p in zip(self.classes, probs)},
            'fps': self.fps.fps,
        }

        # Display if enabled
        if self.display:
            import cv2
            label = f"{result['class']} ({result['confidence']:.0%}) | {result['fps']:.0f} FPS"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
            cv2.imshow('Neuro Brain - Live Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        return result


# ═══════════════════════════════════════════════════════════════
#  REAL-TIME TEXT STREAM (Sentiment Monitoring)
# ═══════════════════════════════════════════════════════════════

class RealtimeTextMonitor(RealTimeProcessor):
    """
    Monitor a stream of text messages for sentiment in real-time.

    Use cases: social media feeds, chat monitoring, review streams.
    """

    def __init__(self):
        super().__init__()
        self.classifier = None
        self.stats = {'positive': 0, 'negative': 0, 'total': 0}

    def _load_model(self):
        from transformers import pipeline
        print("📥 Loading sentiment model...")
        self.classifier = pipeline("sentiment-analysis")

    def feed(self, text):
        """Send a text message into the pipeline."""
        self.input_queue.put(text)

    def _capture_loop(self):
        """No automatic capture — use feed() to send text."""
        while self.running:
            time.sleep(0.1)

    def process(self, text):
        """Analyze sentiment of a text message."""
        if self.classifier is None:
            self._load_model()

        result = self.classifier(text)[0]
        sentiment = result['label']
        confidence = result['score']

        self.stats['total'] += 1
        self.stats[sentiment.lower()] = self.stats.get(sentiment.lower(), 0) + 1

        output = {
            'text': text[:80],
            'sentiment': sentiment,
            'confidence': confidence,
            'stats': dict(self.stats),
        }

        emoji = "😊" if sentiment == "POSITIVE" else "😞"
        print(f"  {emoji} [{confidence:.0%}] {text[:60]}...")
        return output


# ═══════════════════════════════════════════════════════════════
#  REAL-TIME SENSOR STREAM (Number Prediction)
# ═══════════════════════════════════════════════════════════════

class RealtimeSensorPredictor(RealTimeProcessor):
    """
    Process streaming sensor data and make predictions in real-time.

    Use cases: IoT temperature prediction, stock price forecasting,
               health monitoring, industrial sensors.
    """

    def __init__(self, model=None, window_size=10):
        super().__init__()
        self.model = model
        self.window_size = window_size
        self.history = deque(maxlen=window_size * 2)

    def feed(self, sensor_values):
        """Send sensor reading into the pipeline."""
        self.input_queue.put(sensor_values)
        self.history.append(sensor_values)

    def _capture_loop(self):
        """No automatic capture — use feed() to send data."""
        while self.running:
            time.sleep(0.01)

    def process(self, sensor_values):
        """Make prediction from sensor reading."""
        if self.model is None:
            # Use simple moving average as fallback
            if len(self.history) >= 3:
                recent = list(self.history)[-3:]
                prediction = np.mean(recent, axis=0)
            else:
                prediction = sensor_values
        else:
            features = np.array(sensor_values).reshape(1, -1)
            prediction = self.model.predict(features, verbose=0)[0]

        return {
            'input': sensor_values,
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'fps': self.fps.fps,
        }


# ═══════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  🧠 Real-Time Processing — Demo")
    print("=" * 55)

    # Demo 1: Real-time text sentiment monitoring
    print("\n📝 Demo: Real-Time Sentiment Monitor")
    print("-" * 45)

    monitor = RealtimeTextMonitor()
    monitor.start()

    sample_texts = [
        "I absolutely love this new product!",
        "Terrible customer service, very disappointed.",
        "The update fixed all the bugs, great work!",
        "This is the worst experience I've ever had.",
        "Amazing quality and fast shipping!",
        "Broken on arrival. Want a refund immediately.",
    ]

    for text in sample_texts:
        monitor.feed(text)
        time.sleep(1.5)  # Simulate stream timing

    monitor.stop()

    print(f"\n📊 Final Stats:")
    print(f"   Total:    {monitor.stats['total']}")
    print(f"   Positive: {monitor.stats.get('positive', 0)}")
    print(f"   Negative: {monitor.stats.get('negative', 0)}")

    # Camera demo instructions
    print(f"\n\n📷 To run live camera detection:")
    print(f"   cam = RealtimeCamera(camera_id=0)")
    print(f"   cam.start()")
    print(f"   # Press 'q' to quit")
