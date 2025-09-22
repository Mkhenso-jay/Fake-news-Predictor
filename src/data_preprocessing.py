# src\data_preprocessing.py
"""
Preprocessing script for Fake News Detector.
Reads Fake.csv + True.csv, cleans text, encodes labels, saves processed dataset.
Usage:
  python src\data_preprocessing.py --input_dir data/raw --out_file data/processed/dataset.csv
"""

import os
import argparse
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # remove URLs
    text = re.sub(r"<.*?>", " ", text)  # remove HTML
    text = re.sub(r"[^a-z\s]", " ", text)  # letters only
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def load_and_process(input_dir):
    fake_path = os.path.join(input_dir, "Fake.csv")
    true_path = os.path.join(input_dir, "True.csv")
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Fake.csv or True.csv not found in data/raw")
    
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    
    # Detect text columns
    def detect_text_col(df):
        for col in df.columns:
            if col.lower() in ["text", "content", "article", "body"]:
                return col
        # fallback: longest median column
        lengths = df.apply(lambda c: c.astype(str).map(len).median())
        return lengths.idxmax()
    
    fake_col = detect_text_col(fake)
    true_col = detect_text_col(true)
    
    fake = fake.rename(columns={fake_col: "text"})[["text"]].copy()
    true = true.rename(columns={true_col: "text"})[["text"]].copy()
    
    fake["label"] = 0  # FAKE
    true["label"] = 1  # REAL
    
    df = pd.concat([fake, true], ignore_index=True)
    
    print("Raw dataset shape:", df.shape)
    print("Label distribution:\n", df['label'].value_counts())
    
    print("Cleaning text...")
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    
    # drop empty cleaned text
    df = df[df['text_clean'].str.strip() != ""].reset_index(drop=True)
    print("After cleaning shape:", df.shape)
    return df

def main(args):
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    df = load_and_process(args.input_dir)
    df.to_csv(args.out_file, index=False)
    print(f"Processed dataset saved to {args.out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw", help="folder containing Fake.csv and True.csv")
    parser.add_argument("--out_file", type=str, default="data/processed/dataset.csv", help="where to save processed dataset")
    args = parser.parse_args()
    main(args)
