"""
🧠 Neuro Brain — Environment Verification Script
Run this after installing all dependencies to confirm everything works.
Usage: python verify_setup.py
"""

import sys


def check(name, import_name=None):
    """Check if a library is installed and print its version."""
    try:
        mod = __import__(import_name or name)
        ver = getattr(mod, '__version__', '✓')
        print(f"  ✅ {name:.<25} {ver}")
        return True
    except ImportError:
        print(f"  ❌ {name:.<25} NOT INSTALLED")
        return False


def main():
    print(f"\n{'='*50}")
    print(f"  🧠 Neuro Brain — Environment Verification")
    print(f"{'='*50}")
    print(f"  Python ................. {sys.version.split()[0]}")
    print(f"  Platform ............... {sys.platform}\n")

    all_ok = True

    # Foundation
    print("📦 Foundation Libraries:")
    all_ok &= check("numpy")
    all_ok &= check("pandas")
    all_ok &= check("matplotlib")
    all_ok &= check("seaborn")
    all_ok &= check("scikit-learn", "sklearn")

    # Deep Learning
    print("\n🧠 Deep Learning Frameworks:")
    all_ok &= check("tensorflow")
    tf_gpu = False
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"     └─ TF GPU devices: {len(gpus)}")
            tf_gpu = True
        else:
            print(f"     └─ TF GPU: Not available (CPU mode)")
    except Exception:
        pass

    all_ok &= check("torch")
    torch_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            print(f"     └─ CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
            torch_gpu = True
        else:
            print(f"     └─ CUDA: Not available (CPU mode)")
    except Exception:
        pass

    all_ok &= check("torchvision")
    all_ok &= check("torchaudio")

    # Computer Vision
    print("\n👁️  Computer Vision:")
    all_ok &= check("opencv-python", "cv2")
    all_ok &= check("Pillow", "PIL")

    # NLP
    print("\n📝 NLP Libraries:")
    all_ok &= check("transformers")
    all_ok &= check("tokenizers")
    all_ok &= check("nltk")

    # Utilities
    print("\n🛠️  Utilities:")
    all_ok &= check("jupyter")
    all_ok &= check("tqdm")
    all_ok &= check("tensorboard")
    all_ok &= check("python-dotenv", "dotenv")

    # Summary
    print(f"\n{'='*50}")
    if all_ok:
        print("  🎉 All libraries installed successfully!")
    else:
        print("  ⚠️  Some libraries are missing.")
        print("  Run: pip install -r requirements.txt")

    if tf_gpu or torch_gpu:
        print("  🚀 GPU acceleration is ENABLED!")
    else:
        print("  💻 Running in CPU mode (GPU not detected)")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
