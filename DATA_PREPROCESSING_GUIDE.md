# 🧠 Neuro Brain — Data Preprocessing Guide

> Step-by-step guide to collecting, cleaning, preprocessing, splitting, and normalizing data for your neural networks.

---

## 📋 Table of Contents

1. [The Data Pipeline](#1-the-data-pipeline)
2. [Step 1: Data Collection](#2-step-1-data-collection)
3. [Step 2: Data Cleaning](#3-step-2-data-cleaning)
4. [Step 3: Data Preprocessing](#4-step-3-data-preprocessing)
5. [Step 4: Train/Test Split](#5-step-4-traintest-split)
6. [Step 5: Normalization & Scaling](#6-step-5-normalization--scaling)
7. [Complete Working Examples](#7-complete-working-examples)
8. [Common Mistakes to Avoid](#8-common-mistakes-to-avoid)

---

## 1. The Data Pipeline

Every neural network needs data to flow through this pipeline before training:

```
Raw Data → Clean → Transform → Split → Normalize → Ready for Training
```

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  COLLECT     │───▶│   CLEAN     │───▶│  PREPROCESS  │───▶│   SPLIT     │───▶│  NORMALIZE  │
│             │    │             │    │              │    │             │    │             │
│ • CSV files │    │ • Fix nulls │    │ • Encode     │    │ • Train set │    │ • Scale 0-1 │
│ • APIs      │    │ • Remove    │    │ • Feature    │    │ • Test set  │    │ • Standardize│
│ • Databases │    │   duplicates│    │   engineer   │    │ • Val set   │    │ • Per-channel│
│ • Images    │    │ • Fix types │    │ • Reshape    │    │   (80/20)   │    │   normalize │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
```

> **Why is this important?** Garbage in = garbage out. A perfectly designed neural network will fail if fed messy, unscaled, or biased data.

---

## 2. Step 1: Data Collection

### 📊 For Number Prediction (Tabular Data)

```python
import pandas as pd
import numpy as np

# ── Option A: Load from CSV file ──
df = pd.read_csv("data/raw/my_dataset.csv")

# ── Option B: Load from built-in datasets ──
from sklearn.datasets import load_boston, fetch_california_housing

dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

# ── Option C: Generate synthetic data ──
np.random.seed(42)
X = np.random.rand(1000, 3)  # 1000 samples, 3 features
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1
df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
df['target'] = y

# ── First look at your data ──
print(df.head())          # First 5 rows
print(df.shape)           # (rows, columns)
print(df.describe())      # Statistical summary
print(df.info())          # Data types and null counts
```

### 👁️ For Object Recognition (Image Data)

```python
import cv2
import os
from pathlib import Path

# ── Option A: Load local images ──
image_dir = Path("data/raw/images")
images = []
labels = []

for class_name in os.listdir(image_dir):
    class_path = image_dir / class_name
    if class_path.is_dir():
        for img_file in class_path.glob("*.jpg"):
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
            img = cv2.resize(img, (224, 224))            # Standard size
            images.append(img)
            labels.append(class_name)

print(f"Loaded {len(images)} images across {len(set(labels))} classes")

# ── Option B: Use built-in datasets (recommended for learning) ──
import torchvision
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(
    root='data/raw', train=True, download=True, transform=transform
)
print(f"Dataset size: {len(dataset)} images")
```

### 📝 For Text Processing (Text Data)

```python
import pandas as pd

# ── Option A: Load from CSV ──
df = pd.read_csv("data/raw/reviews.csv")

# ── Option B: Use built-in datasets ──
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all')
df = pd.DataFrame({
    'text': newsgroups.data,
    'label': newsgroups.target
})
print(f"Loaded {len(df)} documents across {len(set(df['label']))} categories")

# ── Option C: From Hugging Face datasets ──
from datasets import load_dataset  # pip install datasets

dataset = load_dataset("imdb")
print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
```

---

## 3. Step 2: Data Cleaning

Cleaning removes errors, inconsistencies, and noise from your raw data.

### 🔍 Inspect the Data First

```python
import pandas as pd

df = pd.read_csv("data/raw/my_dataset.csv")

# Check for problems
print("=== Data Quality Report ===\n")
print(f"Total rows:       {len(df)}")
print(f"Total columns:    {len(df.columns)}")
print(f"Duplicate rows:   {df.duplicated().sum()}")
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nData types:")
print(df.dtypes)
```

### 🧹 Handle Missing Values

```python
# ── Strategy 1: Remove rows with missing values ──
# Use when: you have lots of data and few missing values (< 5%)
df_clean = df.dropna()
print(f"Removed {len(df) - len(df_clean)} rows with missing values")

# ── Strategy 2: Fill with mean/median (numeric columns) ──
# Use when: missing values are random and column is numeric
df['age'].fillna(df['age'].median(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)

# ── Strategy 3: Fill with mode (categorical columns) ──
# Use when: column is categorical (text labels)
df['city'].fillna(df['city'].mode()[0], inplace=True)

# ── Strategy 4: Forward/backward fill (time-series) ──
# Use when: data has a time order
df['temperature'].fillna(method='ffill', inplace=True)  # Use previous value

# ── Strategy 5: Advanced — use sklearn imputer ──
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
df[['col1', 'col2']] = imputer.fit_transform(df[['col1', 'col2']])
```

### 🗑️ Remove Duplicates

```python
print(f"Before: {len(df)} rows")
df = df.drop_duplicates()
print(f"After:  {len(df)} rows")

# Remove duplicates based on specific columns
df = df.drop_duplicates(subset=['name', 'date'], keep='first')
```

### 🔧 Fix Data Types

```python
# Convert string numbers to actual numbers
df['price'] = pd.to_numeric(df['price'], errors='coerce')  # invalid → NaN

# Convert to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Convert categories to category type (saves memory)
df['status'] = df['status'].astype('category')
```

### 🚫 Remove Outliers

```python
import numpy as np

# ── Method 1: IQR (Interquartile Range) ──
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df['value'] >= lower) & (df['value'] <= upper)]
print(f"Removed {len(df) - len(df_clean)} outliers")

# ── Method 2: Z-score (remove values > 3 std deviations) ──
from scipy import stats

z_scores = np.abs(stats.zscore(df[['col1', 'col2']]))
df_clean = df[(z_scores < 3).all(axis=1)]
```

---

## 4. Step 3: Data Preprocessing

Transform raw data into a format neural networks can understand.

### 🏷️ Encode Categorical Variables

Neural networks only understand numbers — text labels must be encoded.

```python
# ── Method 1: Label Encoding (for ordered categories) ──
# small < medium < large → 0, 1, 2
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])  # small=2, medium=1, large=0
print(f"Classes: {le.classes_}")

# ── Method 2: One-Hot Encoding (for unordered categories) ──
# red, blue, green → [1,0,0], [0,1,0], [0,0,1]
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=False)

# Or with sklearn:
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df[['color']])
```

### 🔧 Feature Engineering

Create new features from existing ones to give the network more useful information.

```python
# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Polynomial features (capture non-linear relationships)
df['area'] = df['width'] * df['height']
df['price_per_sqft'] = df['price'] / df['area']

# Log transform for skewed data
df['log_income'] = np.log1p(df['income'])  # log(1 + x) handles zeros
```

### 🖼️ Image Preprocessing

```python
import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    """Standard image preprocessing pipeline."""
    # 1. Load
    img = cv2.imread(image_path)

    # 2. Color correction (OpenCV loads as BGR, most models expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Resize to standard input size
    img = cv2.resize(img, target_size)

    # 4. Convert to float and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img

# With PyTorch transforms (recommended)
from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),    # Data augmentation
    transforms.RandomRotation(degrees=15),      # Data augmentation
    transforms.ColorJitter(brightness=0.2),     # Data augmentation
    transforms.ToTensor(),                      # Converts to [0, 1] float
    transforms.Normalize(                       # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 📝 Text Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """Standard text cleaning pipeline."""
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # 3. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 4. Tokenize (split into words)
    words = text.split()

    # 5. Remove stopwords (the, is, at, etc.)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    # 6. Lemmatize (running → run, better → good)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)

# Apply to a DataFrame
df['clean_text'] = df['text'].apply(preprocess_text)

# For Transformers (tokenization is built-in — much simpler!)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokens = tokenizer(
    "I love neural networks!",
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'    # Returns PyTorch tensors
)
print(tokens['input_ids'].shape)  # torch.Size([1, 128])
```

---

## 5. Step 4: Train/Test Split

**Why split?** You need to test your model on data it has *never seen* during training. This tells you if the model truly learned patterns or just memorized the training data (overfitting).

### 📐 Standard Split Ratios

```
┌──────────────────────────────────────────────────────────────┐
│                        FULL DATASET                          │
├────────────────────────────────────┬──────────┬──────────────┤
│          TRAIN (70%)               │ VAL (15%)│  TEST (15%)  │
│  Model learns from this data       │ Tune     │  Final score │
│                                    │ settings │  (untouched) │
└────────────────────────────────────┴──────────┴──────────────┘
```

| Split | Purpose | Typical Size |
|-------|---------|-------------|
| **Training** | Model learns patterns from this data | 70-80% |
| **Validation** | Tune hyperparameters, prevent overfitting | 10-15% |
| **Test** | Final evaluation (never used during training) | 10-20% |

### 🔀 Splitting Tabular Data

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Separate features (X) and target (y)
X = df.drop('target', axis=1).values
y = df['target'].values

# ── Simple 80/20 split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42       # Reproducible results
)
print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# ── Three-way split: 70% train, 15% val, 15% test ──
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
```

### ⚖️ Stratified Split (for Classification)

When your classes are **imbalanced** (e.g., 90% cats, 10% dogs), stratified splitting ensures both train and test sets have the same class distribution.

```python
# Stratified split — keeps class ratios the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y            # ← This ensures proportional class distribution
)

# Verify the distribution
from collections import Counter
print(f"Train distribution: {Counter(y_train)}")
print(f"Test distribution:  {Counter(y_test)}")
```

### 🔄 K-Fold Cross-Validation (Most Robust)

Instead of a single split, K-Fold trains K different models on K different splits, giving a more reliable performance estimate.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold = X[train_idx]
    X_val_fold = X[val_idx]
    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]

    print(f"Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")
    # Train and evaluate your model here
```

### 🖼️ Splitting Image Data (PyTorch)

```python
from torch.utils.data import random_split, DataLoader
import torchvision

dataset = torchvision.datasets.CIFAR10(root='data/raw', train=True, download=True)

# 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
```

---

## 6. Step 5: Normalization & Scaling

**Why normalize?** Features with different scales (e.g., age: 0-100, salary: 20000-200000) confuse neural networks. The model would give more importance to salary just because the numbers are bigger. Normalization puts all features on the same scale.

### 📊 Visual Comparison

```
BEFORE normalization:          AFTER normalization:
Age:    [25, 30, 65, 18]       Age:    [0.15, 0.26, 1.00, 0.00]
Salary: [50000, 120000, ...]   Salary: [0.19, 0.63, ...]
Height: [5.4, 6.1, 5.8]       Height: [0.00, 1.00, 0.57]
```

### Method 1: Min-Max Normalization (Scale to 0-1)

Best for: Most neural networks, image data, bounded features.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  # Scales to [0, 1]

# ⚠️ CRITICAL: Fit ONLY on training data, then transform both
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled = scaler.transform(X_test)          # transform only!

print(f"Before: min={X_train[:, 0].min():.1f}, max={X_train[:, 0].max():.1f}")
print(f"After:  min={X_train_scaled[:, 0].min():.4f}, max={X_train_scaled[:, 0].max():.4f}")
```

> [!CAUTION]
> **Never fit the scaler on test data!** This causes data leakage — the model indirectly learns information about the test set, giving artificially good scores.

### Method 2: Standardization (Z-Score, Mean=0, Std=1)

Best for: Data with outliers, features that follow a normal distribution.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # Mean=0, Std=1

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Mean: {X_train_scaled.mean(axis=0)}")  # ≈ [0, 0, 0, ...]
print(f"Std:  {X_train_scaled.std(axis=0)}")   # ≈ [1, 1, 1, ...]
```

### Method 3: Image Normalization

```python
# ── Simple: Scale to [0, 1] ──
images = images.astype(np.float32) / 255.0

# ── Per-channel normalization (ImageNet standard) ──
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],   # ImageNet RGB means
    std=[0.229, 0.224, 0.225]     # ImageNet RGB stds
)

# Used in a transform pipeline:
transform = transforms.Compose([
    transforms.ToTensor(),        # [0, 255] → [0.0, 1.0]
    normalize                     # Per-channel standardization
])
```

### Method 4: Text Data (No Numeric Scaling Needed)

Text uses **tokenization** instead of numeric scaling. Transformers handle this automatically.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenizer handles all preprocessing:
# text → token IDs → padded/truncated to fixed length
encoded = tokenizer(
    ["I love AI", "Neural networks are great"],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
print(encoded['input_ids'])
# tensor([[ 101, 1045, 2293, 9932,  102,    0,    0],
#         [ 101, 7842, 5765, 2024, 2307,  102,    0]])
```

### 🗂️ Which Method to Choose?

| Method | When to Use | Formula |
|--------|-------------|---------|
| **Min-Max** | Bounded features, no outliers, CNNs | `(x - min) / (max - min)` |
| **Standard** | Normally distributed, has outliers | `(x - mean) / std` |
| **Image /255** | Raw pixel values | `x / 255.0` |
| **ImageNet Norm** | Pre-trained CNN models | `(x - channel_mean) / channel_std` |
| **Tokenization** | Text data for Transformers | Built-in with tokenizer |

---

## 7. Complete Working Examples

### 🔢 Full Pipeline: Number Prediction

```python
"""Complete data pipeline for the Number Predictor."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── 1. COLLECT ──
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'square_feet': np.random.randint(500, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'age_years': np.random.randint(0, 50, n_samples),
    'distance_city': np.random.uniform(0.5, 30, n_samples),
})
# Target: house price (synthetic formula)
df['price'] = (
    df['square_feet'] * 150 +
    df['bedrooms'] * 20000 -
    df['age_years'] * 1000 -
    df['distance_city'] * 5000 +
    np.random.randn(n_samples) * 10000
)
df.to_csv("data/raw/house_prices.csv", index=False)

# ── 2. CLEAN ──
print(f"Missing values:\n{df.isnull().sum()}\n")
print(f"Duplicates: {df.duplicated().sum()}")
df = df.drop_duplicates()

# ── 3. PREPROCESS ──
X = df[['square_feet', 'bedrooms', 'age_years', 'distance_city']].values
y = df['price'].values

# ── 4. SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. NORMALIZE ──
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ── READY! ──
print(f"\n✅ Data pipeline complete!")
print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}  |  y_test: {y_test.shape}")
print(f"X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"X_train mean:  {X_train.mean():.4f} (should be ≈ 0)")
print(f"X_train std:   {X_train.std():.4f} (should be ≈ 1)")

# Save processed data
np.savez("data/processed/house_prices.npz",
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)
print("\n💾 Saved to data/processed/house_prices.npz")
```

### 👁️ Full Pipeline: Object Recognition

```python
"""Complete data pipeline for the Object Detector."""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# ── 1. COLLECT + 3. PREPROCESS (transforms handle both) ──
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),              # Data augmentation
    transforms.RandomCrop(32, padding=4),            # Data augmentation
    transforms.ToTensor(),                           # [0,255] → [0,1]
    transforms.Normalize(                            # Per-channel norm
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

# Download and load
full_trainset = torchvision.datasets.CIFAR10(
    root='data/raw', train=True, download=True, transform=train_transform
)
testset = torchvision.datasets.CIFAR10(
    root='data/raw', train=False, download=True, transform=test_transform
)

# ── 4. SPLIT (train → train + validation) ──
train_size = int(0.85 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

# ── 5. CREATE DATA LOADERS ──
train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# ── READY! ──
print(f"✅ Image pipeline complete!")
print(f"Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")

# Verify normalization
batch = next(iter(train_loader))
images, labels = batch
print(f"Batch shape: {images.shape}")              # [64, 3, 32, 32]
print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")
print(f"Channel means: {images.mean(dim=[0,2,3])}")  # Should be ≈ 0
```

### 📝 Full Pipeline: Text Processing

```python
"""Complete data pipeline for the Text Processor."""
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

# ── 1. COLLECT ──
data = {
    'text': [
        "This movie was absolutely fantastic! Best film of the year.",
        "Terrible waste of time. Poor acting and boring plot.",
        "Decent film, nothing special but enjoyable enough.",
        "I loved every minute! The cinematography was breathtaking.",
        "Worst movie I've ever seen. Complete disaster.",
        # ... add more data
    ],
    'label': [1, 0, 1, 1, 0]  # 1=positive, 0=negative
}
df = pd.DataFrame(data)

# ── 2. CLEAN ──
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)     # Remove URLs
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)   # Keep letters + basic punct
    text = re.sub(r'\s+', ' ', text).strip()      # Remove extra spaces
    return text

df['clean_text'] = df['text'].apply(clean_text)

# ── 3. PREPROCESS (Tokenize) ──
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

encodings = tokenizer(
    df['clean_text'].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# ── 4. SPLIT ──
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
labels = torch.tensor(df['label'].values)

# Create train/test indices
train_idx, test_idx = train_test_split(
    range(len(df)), test_size=0.2, random_state=42, stratify=df['label']
)

train_ids = input_ids[train_idx]
train_mask = attention_mask[train_idx]
train_labels = labels[train_idx]

test_ids = input_ids[test_idx]
test_mask = attention_mask[test_idx]
test_labels = labels[test_idx]

# ── READY! ──
print(f"✅ Text pipeline complete!")
print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
print(f"Token shape: {train_ids.shape}")      # [n_samples, max_length]
print(f"Vocab size: {tokenizer.vocab_size}")
```

---

## 8. Common Mistakes to Avoid

| ❌ Mistake | ✅ Correct Approach |
|-----------|-------------------|
| Fit scaler on entire dataset, then split | **Split first**, then fit scaler on training data only |
| Not shuffling before splitting | Use `shuffle=True` or `random_state` for reproducibility |
| Using the same transforms for train and test | Use **data augmentation only on training** set |
| Ignoring class imbalance | Use **stratified split** or class weights |
| Normalizing target variable but forgetting to inverse-transform predictions | Always **inverse-transform** before interpreting results |
| Removing too many outliers | Only remove **clearly erroneous** values, not natural variation |
| Using test data during any preprocessing step | Test data should be **completely isolated** until final evaluation |

### The Golden Rule

```
┌──────────────────────────────────────────────────────────────┐
│  SPLIT FIRST → THEN FIT ON TRAIN → THEN TRANSFORM BOTH     │
│                                                              │
│  ✅  scaler.fit_transform(X_train)                          │
│  ✅  scaler.transform(X_test)                               │
│                                                              │
│  ❌  scaler.fit_transform(X)  ← Data leakage!              │
│  ❌  scaler.fit_transform(X_test)  ← Data leakage!         │
└──────────────────────────────────────────────────────────────┘
```

---

*Next step: Feed this preprocessed data into your neural network models! 🧠*
