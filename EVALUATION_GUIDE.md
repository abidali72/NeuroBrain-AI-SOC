# 🧠 Neuro Brain — Evaluation & Metrics Guide

> How to evaluate your neural network, interpret results, and know if your model is actually good.

---

## 📋 Table of Contents

1. [Metrics by Task Type](#1-metrics-by-task-type)
2. [Regression Metrics (Number Prediction)](#2-regression-metrics)
3. [Classification Metrics (Objects & Text)](#3-classification-metrics)
4. [Interpreting Results](#4-interpreting-results)
5. [What to Do When Results Are Bad](#5-what-to-do-when-results-are-bad)

---

## 1. Metrics by Task Type

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| **Number Prediction** | R² Score | MSE, MAE, RMSE |
| **Object Recognition** | Accuracy | Precision, Recall, F1, Confusion Matrix |
| **Text Classification** | F1 Score | Accuracy, Precision, Recall, ROC-AUC |

---

## 2. Regression Metrics

### MSE (Mean Squared Error)
```
MSE = average of (predicted - actual)²
```
- **Lower is better.** 0 = perfect predictions.
- Penalizes big errors more than small ones (due to squaring).
- Hard to interpret directly (units are squared).

### MAE (Mean Absolute Error)
```
MAE = average of |predicted - actual|
```
- **Lower is better.** 0 = perfect.
- Easy to understand: "On average, predictions are off by $X."
- Treats all errors equally (unlike MSE).

### RMSE (Root Mean Squared Error)
```
RMSE = √MSE
```
- Same units as the target variable (e.g., dollars, degrees).
- More sensitive to outlier errors than MAE.

### R² Score (Coefficient of Determination) ⭐
```
R² = 1 - (sum of squared errors / total variance)
```
- **Range:** -∞ to 1.0. Higher is better.
- **1.0** = perfect predictions
- **0.0** = model is as good as just predicting the average every time
- **< 0** = model is WORSE than predicting the average (very bad!)

| R² Score | Interpretation |
|----------|---------------|
| 0.90 - 1.00 | Excellent |
| 0.70 - 0.89 | Good |
| 0.50 - 0.69 | Moderate |
| 0.30 - 0.49 | Weak |
| < 0.30 | Poor |

---

## 3. Classification Metrics

### Accuracy
```
Accuracy = correct predictions / total predictions
```
- Simple and intuitive: "85% of images were classified correctly."
- **Warning:** Misleading with imbalanced data! If 95% of emails are not spam, a model that always says "not spam" gets 95% accuracy but catches zero spam.

### Precision
```
Precision = true positives / (true positives + false positives)
```
- "Of everything the model SAID was a cat, how many actually WERE cats?"
- **High precision:** model rarely makes false alarms.

### Recall (Sensitivity)
```
Recall = true positives / (true positives + false negatives)
```
- "Of all ACTUAL cats, how many did the model correctly find?"
- **High recall:** model rarely misses real positives.

### F1 Score ⭐
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall — balances both.
- **Best single metric for classification**, especially with imbalanced data.

### Confusion Matrix
```
                PREDICTED
              Cat    Dog
ACTUAL Cat  [ 45     5  ]  ← 45 correct cats, 5 missed (called dog)
       Dog  [  3    47  ]  ← 47 correct dogs, 3 false alarms (called cat)
```
- Shows exactly WHERE the model makes mistakes.
- Diagonal = correct predictions. Off-diagonal = errors.

---

## 4. Interpreting Results

### For Regression:
```
Good signs:           Bad signs:
─────────────         ─────────────
R² > 0.70             R² < 0.30
MAE is small          MAE is huge
Train ≈ Test loss     Train << Test loss (overfitting)
                      Train >> Test loss (data leak)
```

### For Classification:
```
Good signs:           Bad signs:
─────────────         ─────────────
F1 > 0.80             F1 < 0.50
Precision ≈ Recall    Precision >> Recall (misses too many)
Confusion matrix      Precision << Recall (too many false alarms)
  is diagonal-heavy   Confusion matrix has big off-diagonal values
```

---

## 5. What to Do When Results Are Bad

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Underfitting** | Both train & test loss are high | More epochs, bigger model, more features |
| **Overfitting** | Train loss low, test loss high | More data, dropout, early stopping, augmentation |
| **Class imbalance** | High accuracy but low F1 | Use class weights, over/undersample, use F1 as metric |
| **Bad features** | R² stays near 0 | Better feature engineering, more relevant data |
| **Learning rate** | Loss jumps around | Reduce learning rate (try 0.0001) |

---

*Run `python -m src.evaluate --task all` to evaluate your trained models! 🧠*
