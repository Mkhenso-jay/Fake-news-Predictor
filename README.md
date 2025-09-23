# Fake News Detection System

A robust machine learning solution for detecting fake news using a
RandomForest classifier with TF-IDF feature extraction. This project
provides an end-to-end pipeline for data preprocessing, model training,
tuning, and evaluation, achieving a test accuracy of approximately
99.78%. A Streamlit-based demo app is included for interactive
predictions. Processed data and models are excluded from the repository
and regenerated via scripts to ensure a lightweight, reproducible
codebase.

## Overview

This repository contains the core implementation of a fake news
detection system, designed to classify news articles as "fake" (label 0)
or "real" (label 1). The system leverages a tuned RandomForest model,
optimized with RandomizedSearchCV, and includes a user-friendly
interface for real-time predictions.

### Key Features

-   Data preprocessing with text cleaning and lemmatization.
-   Stratified train-test splitting for unbiased evaluation.
-   Hyperparameter tuning and cross-validation.
-   Optional retraining on full dataset for deployment.
-   Interactive Streamlit demo app.

### Generated Artifacts (Regeneratable)

-   `data/processed/dataset.csv`: Cleaned dataset.
-   `data/processed/train.csv`, `data/processed/test.csv`: Stratified
    80/20 split.
-   `experiments/models/RandomForest_tuned.joblib`: Tuned pipeline.
-   `experiments/test_classification_report.csv`: Test evaluation.
-   `experiments/models/RandomForest_tuned_final_full.joblib`: Final
    model (with `--retrain-full`).

## Installation

1.  **Clone the Repository**:

    ``` bash
    git clone https://github.com/Mkhenso-jay/Fake-news-Predictor.git
    cd Fake-news-Predictor
    ```

2.  **Set Up Virtual Environment**:

    ``` bash
    python -m venv .venv
    source .venv/bin/activate  # On Unix/macOS
    .venv\Scripts\activate     # On Windows
    ```

3.  **Install Dependencies**:

    ``` bash
    pip install -r requirements.txt
    ```

4. **Raw Data**:

    - The repository already includes the raw datasets (`Fake.csv` and `True.csv`) under `data/raw/`.
    - You can directly proceed with preprocessing and training steps without downloading additional files.


## Usage

Run the following commands sequentially (in your activated virtual
environment) 

1.  **Preprocess Raw Data**:

    ``` bash
    python src/data_preprocessing.py --input_dir data/raw --out_file data/processed/dataset.csv
    ```

    Output: `data/processed/dataset.csv`.

2.  **Create Train/Test Split**:

    ``` bash
    python src/split_data.py --data data/processed/dataset.csv --out_dir data/processed --test-size 0.20 --seed 42
    ```

    Output: `data/processed/train.csv`, `data/processed/test.csv`.

3.  **Tune RandomForest**:

    ``` bash
    python src/tune_on_train.py --data data/processed/train.csv --out experiments --n_iter 40 --cv 3
    ```

    Output: `experiments/models/RandomForest_tuned.joblib`,
    `experiments/tuning_results.csv`.

4.  **Evaluate and Retrain**:

    ``` bash
    python src/evaluate_tuned_on_test.py --tuned experiments/models/RandomForest_tuned.joblib --test data/processed/test.csv --out experiments --retrain-full
    ```

    Output: `experiments/test_classification_report.csv`,
    `experiments/models/RandomForest_tuned_final_full.joblib` (if
    --retrain-full).

5.  **Run Streamlit Demo**:

    ``` bash
    streamlit run src/app.py
    ```

    Open the URL displayed in the terminal to predict interactively.


## Performance

-   **Cross-Validation**: RandomForest achieves \~99.74% accuracy,
    \~99.73% F1-macro.
-   **Test Set**: \~99.78% accuracy, F1-scores \~99.79% (Fake), \~99.78%
    (Real).
-   Outperforms baselines (e.g., LinearSVC: 99.57%).

## Repository Structure

-   `data/raw/`: Raw datasets (Fake.csv, True.csv).
-   `src/`: Core scripts (`data_preprocessing.py`, `split_data.py`,
    `train.py`, `tune_on_train.py`, `evaluate_tuned_on_test.py`,
    `app.py`).
-   `experiments/`: Generated models and results (excluded,
    regeneratable).
-   `.gitignore`: Excludes virtual environments, processed data, and
    models, ecxept the final model.
-   `requirements.txt`: Dependency list.
 
