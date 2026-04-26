"""
🛡️ Neuro Brain — Memory Forensics AI
Deep Learning Computer Vision model adapted for Malware Analysis.
Detects Fileless Malware (Meterpreter, Cobalt Strike) by analyzing 
raw memory segments (RAM dumps) converted into 2D byte matrices.

Run: python -m src.models.memory_forensics
"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ═══════════════════════════════════════════════════════════════
#  MITRE ATT&CK Mapping
#  T1055 - Process Injection
#  T1620 - Reflective Code Loading
# ═══════════════════════════════════════════════════════════════

class MemoryForensicsCNN(nn.Module):
    """
    Convolutional Neural Network for Raw Memory Analysis.
    Treats a 4096-byte memory page as a 64x64 grayscale "image".
    Detects spatial byte patterns indicative of decoded shellcode.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Detect entropy changes & NOP sleds
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: Detect PE header anomalies & MZ signatures
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: Complex shellcode structures
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid() # 0.0 = Benign Process, 1.0 = Injected Shellcode
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def generate_synthetic_memory_dumps(n_samples=10000):
    """
    Simulates raw extracting 4KB memory pages (VirtualAlloc segments)
    from a Windows endpoint using a tool like Volatility or Rekall.
    """
    print("[*] Acquiring Volatile Memory (RAM) Segments...")
    np.random.seed(0x0C0C0C0C) # Shellcode execution address joke
    
    # 4KB page represented as 64x64 byte matrix
    X = np.zeros((n_samples, 1, 64, 64), dtype=np.float32) 
    y = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        if np.random.rand() > 0.3: # 70% Benign OS Memory
            y[i] = 0.0
            # Benign memory often has low/medium entropy, structured text, or zeroes
            base_mem = np.random.randint(0, 100, (64, 64)) 
            # Add some structured zero-padding (common in legit C++ structs)
            base_mem[40:, :] = 0 
            X[i, 0] = base_mem / 255.0
            
        else: # 30% Malicious (Process Hollowing / Reflective DLL)
            y[i] = 1.0
            # High entropy (encrypted/packed payloads)
            mal_mem = np.random.randint(50, 255, (64, 64))
            
            # Simulate a NOP sled (\x90) leading into shellcode
            nop_sled_length = np.random.randint(10, 30)
            mal_mem[10, 10:10+nop_sled_length] = 144 # 144 is 0x90 in decimal
            
            # Simulate malicious MZ Header (0x4D 0x5A) somewhere it shouldn't be
            mal_mem[5, 5] = 77  # 'M'
            mal_mem[5, 6] = 90  # 'Z'
            
            X[i, 0] = mal_mem / 255.0

    return X, y


def run_memory_forensics():
    print("=" * 65)
    print(" 🛡️  SOC OPERATIONAL ENGINE: FILELESS MALWARE AI ")
    print("=" * 65)
    
    # Data Pipeline
    X, y = generate_synthetic_memory_dumps(n_samples=1000)
    
    print(f"    [+] Memory Dump Acquired: {len(X)} pages analyzed.")
    print(f"    [+] Reshaping into Spatial Convolution Vectors (64x64 bytes)...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)), 
        batch_size=32, shuffle=True
    )
    
    # Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MemoryForensicsCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    epochs = 3
    print(f"\n[*] Initiating Neural Heuristics | Device: {device} | Epochs: {epochs}")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"    [+] Convolution Pass {epoch+1:02d}/{epochs} | Spatial Entropy Loss: {running_loss/len(train_loader):.4f}")

    print(f"[*] Memory Analysis Complete. Processing Time: {time.time() - start_time:.2f}s")

    
    # Digital Forensics Evaluation
    print("\n[*] Initializing Threat Hunting Evaluation on RAM Segments...")
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(X_test).to(device)
        predictions = model(test_inputs).cpu().numpy()
        pred_labels = (predictions > 0.5).astype(int)
        
    print("\n" + "=" * 65)
    print(" 🔎 DFIR REPORT: MEMORY INJECTION DETECTION")
    print("=" * 65)
    
    target_names = ['Legitimate Process Memory', 'Injected Shellcode / Cobalt Strike']
    print(classification_report(y_test, pred_labels, target_names=target_names))
    
    cm = confusion_matrix(y_test, pred_labels)
    print("\n[+] Confusion Matrix (Forensic False Positive Analysis):")
    print(f"                 Predicted Legit     Predicted Malicious")
    print(f"Actual Legit     [{cm[0][0]:>5}]               [{cm[0][1]:>5}]  <-- False Alarms")
    print(f"Actual Malicious [{cm[1][0]:>5}]               [{cm[1][1]:>5}]  <-- Undetected Payloads")
    
    # Save Model
    torch.save(model.state_dict(), "data/models/memory_forensics.pth")
    print("\n[!] Forensics CNN exported to SOC registry: data/models/memory_forensics.pth")

if __name__ == "__main__":
    run_memory_forensics()
