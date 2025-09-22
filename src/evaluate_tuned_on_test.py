# src/evaluate_tuned_on_test.py
"""
Load tuned pipeline and evaluate on the test CSV.
Usage:
  python src/evaluate_tuned_on_test.py --tuned experiments/models/RandomForest_tuned.joblib --test data/processed/test.csv --out experiments --retrain-full

Saves:
  experiments/test_classification_report.csv
  (optionally) experiments/models/RandomForest_tuned_final_full.joblib
"""
import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_csv(path):
    df = pd.read_csv(path)
    if 'text_clean' not in df.columns and 'text' in df.columns:
        df['text_clean'] = df['text'].astype(str)
    if 'label' not in df.columns and 'label_enc' in df.columns:
        df['label'] = df['label_enc']
    return df

def main(args):
    tuned_path = args.tuned
    test_csv = args.test
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # load tuned pipeline
    obj = joblib.load(tuned_path)
    pipeline = obj['pipeline']
    label_encoder = obj.get('label_encoder', None)

    # load test set
    df_test = load_csv(test_csv)
    X_test = df_test['text_clean'].astype(str).values
    y_test = df_test['label'].values

    # If label_encoder present, transform y_test for comparison if needed
    if label_encoder is not None:
        try:
            # inverse transform is for predictions; ensure comparison uses same labels
            pass
        except Exception:
            pass

    preds = pipeline.predict(X_test)

    # If y_test are strings and preds are numeric, try to align via label_encoder
    # But usually pipeline and label_encoder were saved during tuning; assume preds and y_test comparable
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(out_dir, 'test_classification_report.csv')
    report_df.to_csv(report_path)
    print("Saved test classification report to:", report_path)
    print("\nTest classification report:\n")
    print(pd.DataFrame(report).transpose())

    # Optionally retrain on full dataset (train+test)
    if args.retrain_full:
        print("\nRetraining on full dataset and saving final pipeline...")
        full_df = pd.read_csv('data/processed/dataset.csv')
        if 'text_clean' not in full_df.columns and 'text' in full_df.columns:
            full_df['text_clean'] = full_df['text'].astype(str)
        X_full = full_df['text_clean'].astype(str).values
        y_full = full_df['label'].values
        # best_pipeline is already a fitted pipeline on train during tuning; rebuild and fit on full
        pipeline.fit(X_full, y_full)
        final_save = os.path.join(out_dir, 'models', 'RandomForest_tuned_final_full.joblib')
        os.makedirs(os.path.dirname(final_save), exist_ok=True)
        joblib.dump({'pipeline': pipeline, 'label_encoder': label_encoder}, final_save)
        print("Saved final pipeline trained on full dataset to:", final_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuned', type=str, default='experiments/models/RandomForest_tuned.joblib')
    parser.add_argument('--test', type=str, default='data/processed/test.csv')
    parser.add_argument('--out', type=str, default='experiments')
    parser.add_argument('--retrain-full', action='store_true', help='Retrain tuned pipeline on full dataset and save final model')
    args = parser.parse_args()
    main(args)
