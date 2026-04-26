# 🧠 NeuroBrain AI SOC

A real-time, neural-network-powered Security Operations Center (SOC) for advanced threat detection, featuring NIDS, WAF, and automated DFIR.

## Quick Start

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python verify_setup.py

# 4. Run a model demo
python -m src.models.number_predictor    # Number prediction
python -m src.models.object_detector     # Object recognition
python -m src.models.text_processor      # Text processing
```

## Project Structure

```
brain/
├── src/models/
│   ├── number_predictor.py   # FNN (TensorFlow)
│   ├── object_detector.py    # CNN (PyTorch)
│   └── text_processor.py     # Transformer (Hugging Face)
├── data/                     # Datasets & saved models
├── notebooks/                # Jupyter experiments
├── requirements.txt          # Dependencies
├── verify_setup.py           # Environment checker
└── SETUP_GUIDE.md            # Full setup guide
```

## Models

| Model | Architecture | Framework | Task |
|-------|-------------|-----------|------|
| NumberPredictor | FNN | TensorFlow | Regression & forecasting |
| NeuroBrainCNN | CNN | PyTorch | Image classification |
| TextProcessor | Transformer | Hugging Face | NLP tasks |

## Full Documentation

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation and usage instructions.
