"""
🧠 Neuro Brain — Data Loader & Preprocessor
Reusable utilities for loading, cleaning, splitting, and normalizing data.
Supports tabular (CSV), image, and text data pipelines.

Usage:
    from src.data_loader import TabularLoader, ImageLoader, TextLoader
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class TabularLoader:
    """
    Data pipeline for tabular (CSV/numeric) data.
    Handles loading, cleaning, splitting, and normalization.
    """

    def __init__(self):
        self.scaler_X = None
        self.scaler_y = None
        self.label_encoders = {}
        self.df = None
        self.feature_names = None

    def load(self, path, **kwargs):
        """Load data from CSV file."""
        self.df = pd.read_csv(path, **kwargs)
        print(f"📂 Loaded {len(self.df)} rows × {len(self.df.columns)} columns from {path}")
        return self

    def from_dataframe(self, df):
        """Load from an existing DataFrame."""
        self.df = df.copy()
        print(f"📂 Loaded DataFrame: {len(self.df)} rows × {len(self.df.columns)} columns")
        return self

    def report(self):
        """Print a data quality report."""
        print(f"\n{'='*50}")
        print(f"  📊 Data Quality Report")
        print(f"{'='*50}")
        print(f"  Rows:       {len(self.df)}")
        print(f"  Columns:    {len(self.df.columns)}")
        print(f"  Duplicates: {self.df.duplicated().sum()}")
        print(f"\n  Missing values:")
        missing = self.df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                pct = count / len(self.df) * 100
                print(f"    {col}: {count} ({pct:.1f}%)")
        if missing.sum() == 0:
            print(f"    None ✅")
        print(f"\n  Data types:")
        for col in self.df.columns:
            print(f"    {col}: {self.df[col].dtype}")
        print(f"{'='*50}\n")
        return self

    def clean(self, drop_duplicates=True, fill_strategy='median'):
        """
        Clean the data: remove duplicates, fill missing values.

        Args:
            drop_duplicates: Remove duplicate rows
            fill_strategy: 'median', 'mean', 'mode', or 'drop'
        """
        original_len = len(self.df)

        if drop_duplicates:
            self.df = self.df.drop_duplicates()

        if fill_strategy == 'drop':
            self.df = self.df.dropna()
        else:
            # Numeric columns: fill with median or mean
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                strategy = fill_strategy if fill_strategy in ['median', 'mean'] else 'median'
                imputer = SimpleImputer(strategy=strategy)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

            # Categorical columns: fill with most frequent
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                self.df[cat_cols] = imputer.fit_transform(self.df[cat_cols])

        cleaned = original_len - len(self.df)
        if cleaned > 0:
            print(f"🧹 Cleaned: removed {cleaned} rows ({original_len} → {len(self.df)})")
        else:
            print(f"🧹 Cleaned: {len(self.df)} rows (no rows removed)")
        return self

    def encode_categories(self, columns=None):
        """Label-encode categorical columns."""
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"🏷️ Encoded '{col}': {list(le.classes_)}")
        return self

    def split(self, target_column, test_size=0.2, val_size=None, random_state=42, stratify=False):
        """
        Split into train/test (and optional validation) sets.

        Args:
            target_column: Name of the target/label column
            test_size: Fraction for test set (default 0.2)
            val_size: Fraction for validation set (default None)
            random_state: Random seed for reproducibility
            stratify: Use stratified splitting (for classification)

        Returns:
            dict with X_train, X_test, y_train, y_test (and X_val, y_val if val_size)
        """
        X = self.df.drop(target_column, axis=1).values
        y = self.df[target_column].values
        self.feature_names = [c for c in self.df.columns if c != target_column]

        strat = y if stratify else None

        if val_size:
            # Three-way split
            total_test = test_size + val_size
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=total_test, random_state=random_state, stratify=strat
            )
            relative_val = val_size / total_test
            strat_temp = y_temp if stratify else None
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - relative_val),
                random_state=random_state, stratify=strat_temp
            )
            print(f"✂️ Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
            return {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=strat
            )
            print(f"✂️ Split: Train={len(X_train)} | Test={len(X_test)}")
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
            }

    def normalize(self, data, method='standard', fit_on='X_train'):
        """
        Normalize features. Fits scaler ONLY on training data.

        Args:
            data: dict from split() method
            method: 'standard' (z-score) or 'minmax' (0-1)
            fit_on: key to fit the scaler on (default 'X_train')

        Returns:
            Same dict with normalized values
        """
        if method == 'standard':
            self.scaler_X = StandardScaler()
        elif method == 'minmax':
            self.scaler_X = MinMaxScaler()

        # Fit on training data ONLY
        data[fit_on] = self.scaler_X.fit_transform(data[fit_on])

        # Transform other sets (without fitting!)
        for key in data:
            if key.startswith('X_') and key != fit_on:
                data[key] = self.scaler_X.transform(data[key])

        print(f"📏 Normalized with {method} scaler (fit on {fit_on})")
        return data

    def save(self, data, path="data/processed/dataset.npz"):
        """Save processed data to .npz file."""
        np.savez(path, **data)
        print(f"💾 Saved to {path}")

    def load_processed(self, path="data/processed/dataset.npz"):
        """Load previously processed data."""
        loaded = np.load(path)
        data = {key: loaded[key] for key in loaded.files}
        print(f"📂 Loaded processed data from {path}")
        return data


class ImageLoader:
    """Data pipeline for image classification data using PyTorch."""

    def __init__(self, image_size=(32, 32)):
        self.image_size = image_size

    def get_transforms(self, augment=True):
        """Get image transforms for training and testing."""
        import torchvision.transforms as transforms

        if augment:
            train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        test_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        return train_transform, test_transform

    def load_cifar10(self, data_dir='data/raw', batch_size=64, val_split=0.15):
        """Load CIFAR-10 with train/val/test split."""
        import torchvision
        from torch.utils.data import DataLoader, random_split

        train_tf, test_tf = self.get_transforms(augment=True)

        full_train = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_tf
        )

        # Split training into train + validation
        val_size = int(val_split * len(full_train))
        train_size = len(full_train) - val_size
        trainset, valset = random_split(full_train, [train_size, val_size])

        loaders = {
            'train': DataLoader(trainset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(valset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(testset, batch_size=batch_size, shuffle=False),
        }

        print(f"🖼️ CIFAR-10 loaded: Train={train_size} | Val={val_size} | Test={len(testset)}")
        return loaders

    def load_from_folder(self, data_dir, batch_size=64, val_split=0.15):
        """
        Load images from folder structure:
        data_dir/
            class_a/ (image1.jpg, image2.jpg, ...)
            class_b/ (image1.jpg, image2.jpg, ...)
        """
        import torchvision
        from torch.utils.data import DataLoader, random_split

        train_tf, test_tf = self.get_transforms(augment=True)

        full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_tf)
        print(f"🖼️ Found {len(full_dataset)} images in {len(full_dataset.classes)} classes")
        print(f"   Classes: {full_dataset.classes}")

        # Split
        test_size = int(0.15 * len(full_dataset))
        val_size = int(val_split * len(full_dataset))
        train_size = len(full_dataset) - test_size - val_size
        trainset, valset, testset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        loaders = {
            'train': DataLoader(trainset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(valset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(testset, batch_size=batch_size, shuffle=False),
        }

        print(f"   Split: Train={train_size} | Val={val_size} | Test={test_size}")
        return loaders


class TextLoader:
    """Data pipeline for text data using Hugging Face Transformers."""

    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None

    def _get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def clean_text(self, text):
        """Basic text cleaning."""
        import re
        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)        # Remove URLs
        text = re.sub(r'<[^>]+>', '', text)              # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)     # Keep letters + basic punct
        text = re.sub(r'\s+', ' ', text).strip()         # Remove extra spaces
        return text

    def process(self, texts, labels=None, test_size=0.2, random_state=42):
        """
        Full text pipeline: clean → tokenize → split.

        Args:
            texts: list of text strings
            labels: list of labels (ints)
            test_size: fraction for test set
            random_state: random seed

        Returns:
            dict with train/test tokenized tensors and labels
        """
        import torch

        # Clean
        cleaned = [self.clean_text(t) for t in texts]
        print(f"🧹 Cleaned {len(cleaned)} texts")

        # Tokenize
        tokenizer = self._get_tokenizer()
        encodings = tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        print(f"🏷️ Tokenized: shape={encodings['input_ids'].shape}")

        if labels is not None:
            labels = torch.tensor(labels)

            # Split
            indices = list(range(len(texts)))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state,
                stratify=labels.numpy()
            )

            result = {
                'train_ids': encodings['input_ids'][train_idx],
                'train_mask': encodings['attention_mask'][train_idx],
                'train_labels': labels[train_idx],
                'test_ids': encodings['input_ids'][test_idx],
                'test_mask': encodings['attention_mask'][test_idx],
                'test_labels': labels[test_idx],
            }
            print(f"✂️ Split: Train={len(train_idx)} | Test={len(test_idx)}")
            return result
        else:
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
            }


# ── Quick Demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🧠 Neuro Brain — Data Loader Demo")
    print("="*50)

    # Demo: Tabular data pipeline
    print("\n📊 Tabular Data Pipeline:")
    print("-" * 40)

    np.random.seed(42)
    demo_df = pd.DataFrame({
        'feature_1': np.random.rand(200) * 100,
        'feature_2': np.random.rand(200) * 50,
        'feature_3': np.random.choice(['A', 'B', 'C'], 200),
        'target': np.random.rand(200) * 10,
    })

    loader = TabularLoader()
    loader.from_dataframe(demo_df)
    loader.report()
    loader.clean()
    loader.encode_categories()
    data = loader.split('target', test_size=0.2)
    data = loader.normalize(data, method='standard')

    print(f"\n  X_train shape: {data['X_train'].shape}")
    print(f"  X_test shape:  {data['X_test'].shape}")
    print(f"  X_train mean:  {data['X_train'].mean():.4f} (should be ≈ 0)")
    print(f"  X_train std:   {data['X_train'].std():.4f} (should be ≈ 1)")

    print("\n✅ All data loaders ready!")
