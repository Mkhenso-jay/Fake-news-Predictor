# src/tune_on_train.py
"""
Tune a pipeline using RandomizedSearchCV on a provided training CSV.
Usage:
  python src/tune_on_train.py --data data/processed/train.csv --out experiments --n_iter 40 --cv 3

Saves:
  experiments/models/RandomForest_tuned.joblib
  experiments/tuning_results.csv
"""
import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score

def load_data(path):
    df = pd.read_csv(path)
    if 'text_clean' in df.columns:
        X = df['text_clean'].astype(str).values
    elif 'text' in df.columns:
        X = df['text'].astype(str).values
    else:
        raise ValueError("No text column found in training CSV.")
    if 'label' in df.columns:
        y_raw = df['label'].values
    elif 'label_enc' in df.columns:
        y_raw = df['label_enc'].values
    else:
        raise ValueError("No label column found in training CSV.")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le

def build_pipeline():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1,2), min_df=3)),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return pipe

def param_distributions():
    return {
        'clf__n_estimators': [100, 200, 300, 400, 500],
        'clf__max_depth': [None, 10, 20, 30, 50],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2', 0.3, 0.5],
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__ngram_range': [(1,1), (1,2)]
    }

def main(args):
    X, y, le = load_data(args.data)
    pipe = build_pipeline()
    param_dist = param_distributions()

    cv_inner = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    scorer = make_scorer(f1_score, average='macro')

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring=scorer,
        n_jobs=-1,
        cv=cv_inner,
        verbose=2,
        random_state=42,
        return_train_score=False
    )

    print("Starting RandomizedSearchCV on training set...")
    search.fit(X, y)
    print("Search completed.")
    print("Best score (f1_macro):", search.best_score_)
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    os.makedirs(args.out, exist_ok=True)
    # save best pipeline
    best_pipeline = search.best_estimator_
    save_path = os.path.join(args.out, "models", f"RandomForest_tuned.joblib")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'pipeline': best_pipeline, 'label_encoder': le, 'best_score': search.best_score_, 'best_params': search.best_params_}, save_path)
    print("Saved tuned pipeline to:", save_path)

    # Save search results
    cv_results_df = pd.DataFrame(search.cv_results_)
    results_csv = os.path.join(args.out, "tuning_results.csv")
    cv_results_df.to_csv(results_csv, index=False)
    print("Saved tuning CV results to:", results_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/train.csv')
    parser.add_argument('--out', type=str, default='experiments')
    parser.add_argument('--n_iter', type=int, default=30, help='number of random parameter settings')
    parser.add_argument('--cv', type=int, default=3, help='inner CV folds for search')
    args = parser.parse_args()
    main(args)
