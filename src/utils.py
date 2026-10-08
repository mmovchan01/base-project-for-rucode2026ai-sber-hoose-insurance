import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

def seed_everything(seed: int = 42) -> None:
    """Fix random seeds across all libraries for exact reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, step: float = 0.005) -> tuple[float, float]:
    """Find probability decision threshold that maximizes the F1 score."""
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.05, 0.95, step):
        preds = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return round(best_thresh, 4), round(best_f1, 5)

def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive classification metrics."""
    preds = (y_probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, preds, zero_division=0)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds).tolist()
    
    return {
        'threshold': round(threshold, 4),
        'roc_auc': round(auc, 5),
        'f1_score': round(f1, 5),
        'accuracy': round(acc, 5),
        'precision': round(prec, 5),
        'recall': round(rec, 5),
        'confusion_matrix': cm
    }
