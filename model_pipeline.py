"""Modules 1, 3, 4 & 5: Complete Machine Learning & Deep Learning Pipeline.
Supports both:
  1. Production Mode: Pre-trained 1,000,000-message models (loads in < 0.5s from saved_models/)
  2. Live Benchmark Mode: 5,572-message UCI dataset (trains live in ~6s for viva defense)
"""
import time
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

import preprocessing

DATA_DIR = Path("data")
SMS_TSV_PATH = DATA_DIR / "sms.tsv"
SMS_1M_PATH = DATA_DIR / "sms_spam_1million.csv"
DATA_FALLBACK_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
SAVED_DIR = Path("saved_models")

def load_1m_bundle() -> dict:
    """Loads pre-trained 1,000,000-message production model artifacts in < 0.5 seconds."""
    if not (SAVED_DIR / "classical_models.joblib").exists():
        raise FileNotFoundError("saved_models directory not found. Run train_1m_models.py first.")

    tfidf = joblib.load(SAVED_DIR / "tfidf.joblib")
    classical_models = joblib.load(SAVED_DIR / "classical_models.joblib")
    tokenizer = joblib.load(SAVED_DIR / "tokenizer.joblib")
    lstm_model = tf.keras.models.load_model(SAVED_DIR / "lstm_model.keras")

    with open(SAVED_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)

    with open(SAVED_DIR / "metadata.json", "r") as f:
        meta = json.load(f)

    all_models = {
        "Naive Bayes": classical_models["Naive Bayes"],
        "Logistic Regression": classical_models["Logistic Regression"],
        "Support Vector Machine": classical_models["Support Vector Machine"],
        "Random Forest": classical_models["Random Forest"],
        "Bidirectional LSTM": lstm_model
    }

    # Create a lightweight preview dataframe for EDA module
    if SMS_1M_PATH.exists():
        preview_df = pd.read_csv(SMS_1M_PATH, nrows=5000)
        preview_df["char_length"] = preview_df["message"].astype(str).apply(len)
        preview_df["word_count"] = preview_df["message"].astype(str).apply(lambda s: len(s.split()))
    else:
        preview_df = pd.DataFrame([{"label": "ham", "message": "Sample", "char_length": 6, "word_count": 1}])

    return {
        "mode": "1m",
        "dataset_name": "1,000,000 Augmented SMS Dataset",
        "total_messages": meta["total_rows"],
        "ham_count": meta["ham_count"],
        "spam_count": meta["spam_count"],
        "training_samples": meta["train_rows"],
        "testing_samples": meta["test_rows"],
        "max_length": meta.get("max_length", 70),
        "df": preview_df,
        "tfidf": tfidf,
        "tokenizer": tokenizer,
        "models": all_models,
        "metrics": metrics,
        "feature_names": tfidf.get_feature_names_out(),
        "spam_words": meta["spam_words"],
        "ham_words": meta["ham_words"]
    }

def train_5k_bundle() -> dict:
    """Trains all 5 models locally on the 5,572-message UCI dataset (~6 seconds)."""
    np.random.seed(42)
    tf.random.set_seed(42)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SMS_TSV_PATH.exists():
        urllib.request.urlretrieve(DATA_FALLBACK_URL, SMS_TSV_PATH)

    df = pd.read_csv(SMS_TSV_PATH, sep="\t", header=None, names=["label", "message"], on_bad_lines="skip")
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["message"] = df["message"].astype(str)
    df["clean_message"] = df["message"].apply(clean_text)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    df["char_length"] = df["message"].apply(len)
    df["word_count"] = df["message"].apply(lambda s: len(s.split()))

    X = df["clean_message"]
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    y_test_arr = np.asarray(y_test)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t = tfidf.transform(X_test)

    # Tokenizer for LSTM
    vocab_size = 10000
    max_length = 100
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length, padding="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_length, padding="post")

    models = {}
    metrics = {}

    # 1. Naive Bayes
    t0 = time.time()
    nb = MultinomialNB()
    nb.fit(X_train_t, y_train)
    t_nb = time.time() - t0
    models["Naive Bayes"] = nb
    metrics["Naive Bayes"] = _eval_model(y_test_arr, nb.predict(X_test_t), nb.predict_proba(X_test_t)[:, 1], t_nb)

    # 2. Logistic Regression
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_t, y_train)
    t_lr = time.time() - t0
    models["Logistic Regression"] = lr
    metrics["Logistic Regression"] = _eval_model(y_test_arr, lr.predict(X_test_t), lr.predict_proba(X_test_t)[:, 1], t_lr)

    # 3. Linear SVM
    t0 = time.time()
    svm = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False, max_iter=2000))
    svm.fit(X_train_t, y_train)
    t_svm = time.time() - t0
    models["Support Vector Machine"] = svm
    metrics["Support Vector Machine"] = _eval_model(y_test_arr, svm.predict(X_test_t), svm.predict_proba(X_test_t)[:, 1], t_svm)

    # 4. Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1)
    rf.fit(X_train_t, y_train)
    t_rf = time.time() - t0
    models["Random Forest"] = rf
    metrics["Random Forest"] = _eval_model(y_test_arr, rf.predict(X_test_t), rf.predict_proba(X_test_t)[:, 1], t_rf)

    # 5. Bi-LSTM
    t0 = time.time()
    lstm = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    lstm.fit(X_train_seq, y_train, epochs=3, batch_size=64, verbose=0)
    t_lstm = time.time() - t0
    l_prob = lstm.predict(X_test_seq, verbose=0).flatten()
    models["Bidirectional LSTM"] = lstm
    metrics["Bidirectional LSTM"] = _eval_model(y_test_arr, (l_prob >= 0.5).astype(int), l_prob, t_lstm)

    # Feature Importance
    feature_names = np.array(tfidf.get_feature_names_out())
    log_prob_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    spam_idx = np.argsort(log_prob_diff)[-25:][::-1]
    ham_idx = np.argsort(log_prob_diff)[:25]
    spam_words = [(str(feature_names[i]), float(log_prob_diff[i])) for i in spam_idx]
    ham_words = [(str(feature_names[i]), float(log_prob_diff[i])) for i in ham_idx]

    return {
        "mode": "5k",
        "dataset_name": "UCI SMS Spam Benchmark Collection",
        "total_messages": len(df),
        "ham_count": int((df["label"] == "ham").sum()),
        "spam_count": int((df["label"] == "spam").sum()),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "max_length": max_length,
        "df": df,
        "tfidf": tfidf,
        "tokenizer": tokenizer,
        "models": models,
        "metrics": metrics,
        "feature_names": feature_names,
        "spam_words": spam_words,
        "ham_words": ham_words
    }

def _eval_model(y_true, y_pred, y_prob, training_time: float) -> dict:
    try:
        auc_score = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        auc_score = 0.0
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

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

def predict_message(bundle: dict, message: str) -> dict:
    cleaned = preprocessing.clean_text(message)
    tfidf_x = bundle["tfidf"].transform([cleaned])

    # 1. Naive Bayes
    nb = bundle["models"]["Naive Bayes"]
    nb_pred = "SPAM" if nb.predict(tfidf_x)[0] == 1 else "HAM"
    nb_prob = float(nb.predict_proba(tfidf_x)[0, 1])

    # 2. Logistic Regression
    lr = bundle["models"]["Logistic Regression"]
    lr_pred = "SPAM" if lr.predict(tfidf_x)[0] == 1 else "HAM"
    lr_prob = float(lr.predict_proba(tfidf_x)[0, 1])

    # 3. Support Vector Machine
    svm = bundle["models"]["Support Vector Machine"]
    svm_pred = "SPAM" if svm.predict(tfidf_x)[0] == 1 else "HAM"
    svm_prob = float(svm.predict_proba(tfidf_x)[0, 1])

    # 4. Random Forest
    rf = bundle["models"]["Random Forest"]
    rf_pred = "SPAM" if rf.predict(tfidf_x)[0] == 1 else "HAM"
    rf_prob = float(rf.predict_proba(tfidf_x)[0, 1])

    # 5. Bidirectional LSTM
    seq = bundle["tokenizer"].texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=bundle["max_length"], padding="post", truncating="post")
    lstm_prob = float(bundle["models"]["Bidirectional LSTM"].predict(padded, verbose=0)[0][0])
    lstm_pred = "SPAM" if lstm_prob >= 0.5 else "HAM"

    spam_votes = sum([
        nb_pred == "SPAM",
        lr_pred == "SPAM",
        svm_pred == "SPAM",
        rf_pred == "SPAM",
        lstm_pred == "SPAM"
    ])
    consensus_pred = "SPAM" if spam_votes >= 3 else "HAM"
    avg_prob = (nb_prob + lr_prob + svm_prob + rf_prob + lstm_prob) / 5.0

    return {
        "message": str(message),
        "clean_message": cleaned,
        "Naive Bayes prediction": nb_pred,
        "Naive Bayes spam probability": nb_prob,
        "Logistic Regression prediction": lr_pred,
        "Logistic Regression spam probability": lr_prob,
        "Support Vector Machine prediction": svm_pred,
        "Support Vector Machine spam probability": svm_prob,
        "Random Forest prediction": rf_pred,
        "Random Forest spam probability": rf_prob,
        "LSTM prediction": lstm_pred,
        "LSTM spam probability": lstm_prob,
        "Consensus prediction": consensus_pred,
        "Consensus spam probability": avg_prob,
        "Spam votes": f"{spam_votes}/5",
    }
