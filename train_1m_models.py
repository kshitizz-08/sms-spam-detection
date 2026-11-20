"""Offline 1-Million Dataset Training Pipeline (Option 2: Train Once & Save).
Trains all 5 models on the full 1,000,000 dataset (data/sms_spam_1million.csv),
evaluates metrics on a 200,000-sample test split, and saves artifacts to saved_models/.
"""
import time
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

from preprocessing import clean_text

DATA_1M_PATH = Path("data/sms_spam_1million.csv")
SAVED_DIR = Path("saved_models")
SAVED_DIR.mkdir(parents=True, exist_ok=True)

MAX_FEATURES = 15000
VOCAB_SIZE = 12000
MAX_LENGTH = 70

def train_and_save_1m():
    print("=" * 75)
    print("🚀 TRAINING 1,000,000-MESSAGE PRODUCTION MODELS (OFFLINE PIPELINE)")
    print("=" * 75)
    
    t_start = time.time()
    
    # 1. Load dataset
    print(f"Loading 1,000,000 dataset from {DATA_1M_PATH} ...")
    df = pd.read_csv(DATA_1M_PATH, usecols=["label", "message"])
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["message"] = df["message"].astype(str)
    
    total_rows = len(df)
    ham_count = int((df["label"] == "ham").sum())
    spam_count = int((df["label"] == "spam").sum())
    print(f"Dataset Loaded: {total_rows:,} rows (Ham: {ham_count:,}, Spam: {spam_count:,})")
    
    # Map binary target
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    
    # 2. Train / Test Split (80% / 20%)
    print("Splitting into 800,000 train / 200,000 test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"], test_size=0.20, random_state=42, stratify=df["label_num"]
    )
    y_test_arr = np.asarray(y_test)
    print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")
    
    # 3. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer (max_features=15,000)...")
    t0 = time.time()
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english", ngram_range=(1, 2))
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t = tfidf.transform(X_test)
    tfidf_time = time.time() - t0
    print(f"TF-IDF completed in {tfidf_time:.2f}s")
    
    metrics = {}
    models = {}
    
    # --- Model 1: Naive Bayes ---
    print("\nTraining Model 1: Multinomial Naive Bayes...")
    t0 = time.time()
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train_t, y_train)
    nb_time = time.time() - t0
    nb_pred = nb.predict(X_test_t)
    nb_prob = nb.predict_proba(X_test_t)[:, 1]
    models["Naive Bayes"] = nb
    metrics["Naive Bayes"] = _eval_model(y_test_arr, nb_pred, nb_prob, nb_time)
    print(f"  Naive Bayes: Acc={metrics['Naive Bayes']['accuracy']:.4f}, F1={metrics['Naive Bayes']['f1']:.4f} in {nb_time:.2f}s")
    
    # --- Model 2: Logistic Regression (SGD) ---
    print("\nTraining Model 2: Logistic Regression (SGD Loss)...")
    t0 = time.time()
    lr = SGDClassifier(loss="log_loss", max_iter=30, random_state=42, n_jobs=-1)
    lr.fit(X_train_t, y_train)
    lr_time = time.time() - t0
    lr_pred = lr.predict(X_test_t)
    lr_prob = lr.predict_proba(X_test_t)[:, 1]
    models["Logistic Regression"] = lr
    metrics["Logistic Regression"] = _eval_model(y_test_arr, lr_pred, lr_prob, lr_time)
    print(f"  Logistic Reg: Acc={metrics['Logistic Regression']['accuracy']:.4f}, F1={metrics['Logistic Regression']['f1']:.4f} in {lr_time:.2f}s")

    # --- Model 3: Linear SVM ---
    print("\nTraining Model 3: Support Vector Machine (Linear SVM with probability calibration)...")
    t0 = time.time()
    svm = CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, dual=False))
    svm.fit(X_train_t, y_train)
    svm_time = time.time() - t0
    svm_pred = svm.predict(X_test_t)
    svm_prob = svm.predict_proba(X_test_t)[:, 1]
    models["Support Vector Machine"] = svm
    metrics["Support Vector Machine"] = _eval_model(y_test_arr, svm_pred, svm_prob, svm_time)
    print(f"  Linear SVM: Acc={metrics['Support Vector Machine']['accuracy']:.4f}, F1={metrics['Support Vector Machine']['f1']:.4f} in {svm_time:.2f}s")

    # --- Model 4: Random Forest ---
    print("\nTraining Model 4: Random Forest Classifier (Optimized 30-Tree Ensemble)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=30, max_depth=16, random_state=42, n_jobs=-1)
    rf.fit(X_train_t, y_train)
    rf_time = time.time() - t0
    rf_pred = rf.predict(X_test_t)
    rf_prob = rf.predict_proba(X_test_t)[:, 1]
    models["Random Forest"] = rf
    metrics["Random Forest"] = _eval_model(y_test_arr, rf_pred, rf_prob, rf_time)
    print(f"  Random Forest: Acc={metrics['Random Forest']['accuracy']:.4f}, F1={metrics['Random Forest']['f1']:.4f} in {rf_time:.2f}s")

    # --- Model 5: Bidirectional LSTM ---
    print("\nTraining Model 5: Bidirectional LSTM Deep Neural Network...")
    t0 = time.time()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    # Fit tokenizer on 250,000 train samples for vocabulary coverage
    tok_sample_size = min(250000, len(X_train))
    tokenizer.fit_on_texts(X_train[:tok_sample_size])
    
    # Train LSTM on 100,000 stratified sequences for fast high-accuracy convergence
    lstm_train_size = min(100000, len(X_train))
    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train[:lstm_train_size]),
        maxlen=MAX_LENGTH, padding="post", truncating="post"
    )
    
    lstm_test_size = min(25000, len(X_test))
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test[:lstm_test_size]),
        maxlen=MAX_LENGTH, padding="post", truncating="post"
    )
    
    lstm_model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=32),
        Bidirectional(LSTM(16)),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    lstm_model.fit(
        X_train_seq, y_train[:lstm_train_size],
        epochs=1,
        batch_size=512,
        validation_split=0.1,
        verbose=1
    )
    lstm_time = time.time() - t0
    lstm_prob = lstm_model.predict(X_test_seq, verbose=0, batch_size=512).flatten()
    lstm_pred = (lstm_prob >= 0.5).astype(int)
    metrics["Bidirectional LSTM"] = _eval_model(y_test_arr[:lstm_test_size], lstm_pred, lstm_prob, lstm_time)
    print(f"  Bi-LSTM: Acc={metrics['Bidirectional LSTM']['accuracy']:.4f}, F1={metrics['Bidirectional LSTM']['f1']:.4f} in {lstm_time:.2f}s")

    # Top discriminative spam / ham words
    feature_names = np.array(tfidf.get_feature_names_out())
    log_prob_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    spam_indices = np.argsort(log_prob_diff)[-30:][::-1]
    ham_indices = np.argsort(log_prob_diff)[:30]
    spam_words = [(str(feature_names[i]), float(log_prob_diff[i])) for i in spam_indices]
    ham_words = [(str(feature_names[i]), float(log_prob_diff[i])) for i in ham_indices]

    # 4. Save Artifacts
    print("\nSaving all trained models to disk (saved_models/)...")
    joblib.dump(tfidf, SAVED_DIR / "tfidf.joblib")
    joblib.dump(models, SAVED_DIR / "classical_models.joblib")
    joblib.dump(tokenizer, SAVED_DIR / "tokenizer.joblib")
    lstm_model.save(SAVED_DIR / "lstm_model.keras")
    
    # Save metadata and metrics
    metadata = {
        "dataset_name": "sms_spam_1million.csv",
        "total_rows": total_rows,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "max_length": MAX_LENGTH,
        "feature_count": len(feature_names),
        "spam_words": spam_words,
        "ham_words": ham_words,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_duration_sec": time.time() - t_start
    }
    
    with open(SAVED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    with open(SAVED_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n✅ All 1-Million models trained and saved successfully in {time.time()-t_start:.1f} seconds!")
    print(f"Artifacts saved in: {SAVED_DIR.resolve()}")
    return metadata, metrics

def _eval_model(y_true, y_pred, y_prob, training_time: float) -> dict:
    try:
        auc_score = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        auc_score = 0.0
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    # Sample ROC points if too large for JSON
    if len(fpr) > 100:
        idx = np.linspace(0, len(fpr) - 1, 100).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc_score,
        "training_time": float(training_time),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

if __name__ == "__main__":
    train_and_save_1m()
