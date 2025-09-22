# src/split_data.py
"""
Create a stratified train/test split and save CSVs.
Usage:
  python src/split_data.py --data data/processed/dataset.csv --out_dir data/processed --test-size 0.20
Produces:
  data/processed/train.csv
  data/processed/test.csv
"""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main(args):
    df = pd.read_csv(args.data)
    # prefer text_clean or text
    if 'text_clean' not in df.columns and 'text' in df.columns:
        df['text_clean'] = df['text'].astype(str)
    if 'label' not in df.columns and 'label_enc' in df.columns:
        df['label'] = df['label_enc']
    if 'label' not in df.columns:
        raise ValueError("No 'label' or 'label_enc' column found in dataset.")
    os.makedirs(args.out_dir, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=args.test_size, stratify=df['label'], random_state=args.seed)
    train_path = os.path.join(args.out_dir, 'train.csv')
    test_path = os.path.join(args.out_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train ({len(train_df)}) -> {train_path}")
    print(f"Saved test  ({len(test_df)}) -> {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/dataset.csv')
    parser.add_argument('--out_dir', type=str, default='data/processed')
    parser.add_argument('--test-size', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
