#!/usr/bin/env python3
"""
RuCode 2026 AI - Sber House / Device Insurance Competition
B. BASE: Подбери страховку (Сбер) - Inference & Submission Generation Script

Usage:
    python predict.py [--test_data public_test.csv] [--model_dir models] [--output submission_seed_42.csv]
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

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Predictions on Test Data")
    parser.add_argument('--test_data', type=str, default='public_test.csv', help='Path to test CSV file')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory with saved model weights')
    parser.add_argument('--output', type=str, default=None, help='Output path for submission CSV')
    parser.add_argument('--threshold', type=float, default=None, help='Override decision threshold (optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("  SBER INSURANCE: SMARTPHONE SCREEN PROTECTION - INFERENCE")
    print("=" * 70)
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory '{args.model_dir}' does not exist! Run train.py first.")
        
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test dataset '{args.test_data}' not found!")
        
    print(f"Loading trained model ensemble from '{args.model_dir}'...")
    ensemble = InsurancePropensityEnsemble.load(args.model_dir)
    seed = ensemble.seed
    
    # Determine output path
    output_path = args.output if args.output else f"submission_seed_{seed}.csv"
    threshold = args.threshold if args.threshold is not None else ensemble.optimal_threshold
    
    print(f"Model trained with Seed: {seed}")
    print(f"Optimal decision threshold: {threshold:.4f}")
    print(f"Loading test data from '{args.test_data}'...")
    test_df = pd.read_csv(args.test_data)
    print(f"Test data shape: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Generate probabilities and binary predictions
    print("Generating predictions...")
    probs = ensemble.predict_proba(test_df)
    preds = (probs >= threshold).astype(int)
    
    # Build submission DataFrame
    submission = pd.DataFrame({
        'customer_id': test_df['customer_id'],
        'accepted': preds
    })
    
    # Submission Validation Checks
    assert len(submission) == len(test_df), f"Row count mismatch: {len(submission)} vs {len(test_df)}"
    assert list(submission.columns) == ['customer_id', 'accepted'], f"Columns mismatch: {submission.columns}"
    assert submission['accepted'].isin([0, 1]).all(), "Target contains values other than 0 and 1"
    assert submission.isnull().sum().sum() == 0, "Submission contains null/NaN values"
    
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission successfully saved to: {output_path}")
    print(f"Submission distribution of predicted classes:")
    print(submission['accepted'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_string())
    print(f"Total positive conversions predicted: {preds.sum()} / {len(preds)}")
    print("=" * 70)

if __name__ == '__main__':
    main()
