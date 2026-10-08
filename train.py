#!/usr/bin/env python3
"""
RuCode 2026 AI - Sber House / Device Insurance Competition
B. BASE: Подбери страховку (Сбер) - Model Training Script

Usage:
    python train.py [--data train.csv] [--model_dir models] [--seed 42] [--n_splits 5]
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import InsurancePropensityEnsemble
from src.utils import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sber Insurance Propensity Model Ensemble")
    parser.add_argument('--data', type=str, default='train.csv', help='Path to training CSV file')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model weights')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of Stratified K-Fold splits')
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    
    print("=" * 70)
    print("  SBER INSURANCE: SMARTPHONE SCREEN PROTECTION PROPENSITY MODEL")
    print("  Competition: RuCode 2026 AI")
    print("=" * 70)
    print(f"Loading training data from: {args.data}")
    
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Training dataset '{args.data}' not found!")
        
    train_df = pd.read_csv(args.data)
    print(f"Dataset shape: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Target distribution ('accepted'):")
    print(train_df['accepted'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_string())
    
    ensemble = InsurancePropensityEnsemble(n_splits=args.n_splits, seed=args.seed)
    cv_metrics = ensemble.fit(train_df)
    
    # Save model weights and metadata
    ensemble.save(args.model_dir)
    
    print("\n" + "=" * 70)
    print(" TRAINING SUMMARY & VALIDATION METRICS")
    print("=" * 70)
    print(f"Ensemble OOF ROC-AUC  : {cv_metrics['ensemble']['roc_auc']:.5f}")
    print(f"Ensemble OOF F1-Score : {cv_metrics['ensemble']['f1_score']:.5f}")
    print(f"Ensemble Accuracy     : {cv_metrics['ensemble']['accuracy']:.5f}")
    print(f"Ensemble Precision    : {cv_metrics['ensemble']['precision']:.5f}")
    print(f"Ensemble Recall       : {cv_metrics['ensemble']['recall']:.5f}")
    print(f"Calibrated Threshold  : {cv_metrics['optimal_threshold']:.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP):")
    print(f"  {cv_metrics['ensemble']['confusion_matrix']}")
    print(f"Artifacts successfully saved to '{args.model_dir}/'")
    print("=" * 70)

if __name__ == '__main__':
    main()
