# src/train.py
"""
Train baseline classifiers for Fake News Detector.
Usage (Windows PowerShell, from project root with venv activated):
  python src\train.py --data data/processed/dataset.csv --out experiments

This script:
 - loads processed CSV with columns: text_clean (or text) and label (0/1 or strings)
 - builds TF-IDF pipeline + classifiers (LogisticRegression, MultinomialNB, LinearSVC, RandomForest)
 - runs 5-fold stratified CV and reports mean/std for accuracy, precision, recall, f1_macro
 - saves experiments/results.csv and the best model pipeline to experiments/models/
"""
import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer

def load_data(path):
    df = pd.read_csv(path)
    # prefer text_clean, fallback to text
    if 'text_clean' in df.columns:
        X = df['text_clean'].astype(str).values
    elif 'text' in df.columns:
        X = df['text'].astype(str).values
    else:
        raise ValueError("No 'text_clean' or 'text' column found in processed dataset.")
    if 'label' in df.columns:
        y_raw = df['label'].values
    elif 'label_enc' in df.columns:
        y_raw = df['label_enc'].values
    else:
        raise ValueError("No 'label' or 'label_enc' column found in processed dataset.")
    # encode labels to integers 0..k-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le

def get_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced'),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(max_iter=2000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    }
    # try to include XGBoost if installed (optional)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
    except Exception:
        pass
    return models

def evaluate_models(X, y, out_dir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    results = []
    models = get_models()
    for name, clf in models.items():
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1,2), min_df=3)),
            ('clf', clf)
        ])
        print(f"\nRunning 5-fold CV for: {name}")
        cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
        row = {
            'model': name,
            'accuracy_mean': cv['test_accuracy'].mean(),
            'accuracy_std': cv['test_accuracy'].std(),
            'precision_mean': cv['test_precision_macro'].mean(),
            'recall_mean': cv['test_recall_macro'].mean(),
            'f1_mean': cv['test_f1_macro'].mean(),
            'f1_std': cv['test_f1_macro'].std()
        }
        results.append(row)
        print(f" {name:15s} acc={row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}  f1={row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
    results_df = pd.DataFrame(results).sort_values(by='f1_mean', ascending=False).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
    print("\nSaved CV results to:", os.path.join(out_dir, 'results.csv'))
    return results_df

def fit_and_save_best(X, y, results_df, label_encoder, out_dir):
    best_name = results_df.iloc[0]['model']
    print("\nFitting best model on full dataset:", best_name)
    models = get_models()
    best_clf = models[best_name]
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1,2), min_df=3)),
        ('clf', best_clf)
    ])
    pipeline.fit(X, y)
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{best_name}_tfidf.joblib")
    joblib.dump({'pipeline': pipeline, 'label_encoder': label_encoder}, save_path)
    print("Saved best pipeline to:", save_path)
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/dataset.csv')
    parser.add_argument('--out', type=str, default='experiments')
    args = parser.parse_args()

    X, y, le = load_data(args.data)
    results_df = evaluate_models(X, y, args.out)
    print("\nSummary (top models):")
    print(results_df.to_string(index=False))
    fit_and_save_best(X, y, results_df, le, args.out)
