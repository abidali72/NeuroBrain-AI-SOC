"""
🛡️ Neuro Brain — Network Intrusion Detection System (IDS)
Advanced Deep Learning Model for Malicious Traffic Classification

Threat Context: Detects sophisticated anomalies including DDoS, 
Brute Force, and lateral movement sequences.

Run: python -m src.models.intrusion_detector
"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ═══════════════════════════════════════════════════════════════
#  MITRE ATT&CK Mapping
#  T1046 - Network Service Scanning
#  T1110 - Brute Force
#  T1498 - Network Denial of Service
# ═══════════════════════════════════════════════════════════════

class IntrusionDetectionNet(nn.Module):
    """
    Deep Neural Network for Binary Threat Classification.
    Architecture:
    Input [12 Features] → Dense(64) + GELU → Dropout(0.3) 
    → Dense(32) + GELU → Dense(1) + Sigmoid
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1),
            nn.Sigmoid() # Output: 0.0 (Benign) to 1.0 (Malicious)
        )

    def forward(self, x):
        return self.network(x)

def generate_synthetic_soc_data(n_samples=10000):
    """
    Generates synthetic network telemetry data mimicking a SIEM log.
    Features: 
    0: duration, 1: src_bytes, 2: dst_bytes, 3: wrong_fragment, 
    4: urgent, 5: hot, 6: num_failed_logins, 7: num_compromised, 
    8: root_shell, 9: su_attempted, 10: num_file_creations, 11: count
    """
    print("[*] Simulating Network Telemetry (SIEM logs)...")
    np.random.seed(0xDEADBEEF) # Cybersecurity easter egg
    
    X = np.zeros((n_samples, 12), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        # 70% Benign Traffic
        if np.random.rand() > 0.3:
            y[i] = 0.0
            X[i] = [
                np.random.exponential(5),      # short duration
                np.random.lognormal(6, 1),     # normal src bytes
                np.random.lognormal(8, 1.5),   # normal dst bytes
                0, 0, 0,                       # no fragments/urgent
                np.random.poisson(0.1),        # rare failed logins
                0, 0, 0, 0,                    # no root/compromise
                np.random.poisson(5)           # normal conn count
            ]
        # 30% Malicious Traffic (DDoS, Brute Force, Exploits)
        else:
            y[i] = 1.0
            attack_type = np.random.choice(['ddos', 'bruteforce', 'exploit'])
            
            if attack_type == 'ddos':
                X[i] = [0, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.random.poisson(500)]
            elif attack_type == 'bruteforce':
                X[i] = [5, 120, 180, 0, 0, 0, np.random.randint(20, 100), 0, 0, 0, 0, 5]
            else: # Exploit / Privilege Escalation
                X[i] = [120, 25000, 5000, 1, 0, 2, 0, 1, 1, 1, np.random.poisson(5), 2]

    return X, y

def train_ids_model():
    print("=" * 65)
    print(" 🛡️  SOC OPERATIONAL ENGINE: NEURAL IDS TRAINING ")
    print("=" * 65)
    
    # Data Pipeline
    X, y = generate_synthetic_soc_data(n_samples=20000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)), batch_size=64, shuffle=True)
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IntrusionDetectionNet(input_dim=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4) # AdamW prevents overfitting securely
    criterion = nn.BCELoss() # Binary Cross Entropy for Malicious vs Benign
    
    # Training Loop
    epochs = 15
    print(f"\n[*] Initiating Deep Learning Phase | Device: {device} | Epochs: {epochs}")
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
            
        if (epoch + 1) % 3 == 0:
            print(f"    [+] Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

    print(f"[*] Training Complete. Processing Time: {time.time() - start_time:.2f}s")
    
    # Evaluation Phase
    print("\n[*] Initializing Threat Hunting Evaluation...")
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(X_test).to(device)
        predictions = model(test_inputs).cpu().numpy()
        pred_labels = (predictions > 0.5).astype(int)
        
    print("\n" + "=" * 65)
    print(" 🔎 THREAT INTELLIGENCE REPORT: MODEL DEPLOYMENT METRICS")
    print("=" * 65)
    print(classification_report(y_test, pred_labels, target_names=['Benign Traffic', 'Malicious Threat']))
    
    cm = confusion_matrix(y_test, pred_labels)
    print("\n[+] Confusion Matrix (True Positives vs False Positives):")
    print(f"                 Predicted Benign   Predicted Malicious")
    print(f"Actual Benign    [{cm[0][0]:>5}]              [{cm[0][1]:>5}]  <-- False Alarms (Type I Error)")
    print(f"Actual Malicious [{cm[1][0]:>5}]              [{cm[1][1]:>5}]  <-- Undetected Threats (Type II Error)")
    
    # Save Model
    torch.save(model.state_dict(), "data/models/intrusion_detector.pth")
    print("\n[!] Model successfully exported to SOC registry: data/models/intrusion_detector.pth")

if __name__ == "__main__":
    train_ids_model()
