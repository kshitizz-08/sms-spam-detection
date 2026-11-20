"""Module 5 Support: Local SQLite storage for users and multi-model prediction history.
Completely local, self-contained database with zero external connections.
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/spam_history.db")

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            nb_prediction TEXT,
            nb_spam_probability REAL,
            lr_prediction TEXT,
            lr_spam_probability REAL,
            svm_prediction TEXT,
            svm_spam_probability REAL,
            rf_prediction TEXT,
            rf_spam_probability REAL,
            lstm_prediction TEXT,
            lstm_spam_probability REAL,
            consensus_prediction TEXT,
            consensus_spam_probability REAL,
            created_at TEXT NOT NULL
        )""")
        # Ensure backwards-compatibility with older db tables
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(predictions)")
        existing_cols = {col[1] for col in cursor.fetchall()}
        new_cols = [
            ("lr_prediction", "TEXT"),
            ("lr_spam_probability", "REAL"),
            ("svm_prediction", "TEXT"),
            ("svm_spam_probability", "REAL"),
            ("rf_prediction", "TEXT"),
            ("rf_spam_probability", "REAL"),
            ("consensus_prediction", "TEXT"),
            ("consensus_spam_probability", "REAL")
        ]
        for col_name, col_type in new_cols:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")

def add_user(username, password_hash):
    with connect() as conn:
        conn.execute("INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                     (username, password_hash, datetime.now().isoformat(timespec="seconds")))

def get_user(username):
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

def add_prediction(username, result):
    with connect() as conn:
        conn.execute("""INSERT INTO predictions
            (username, message, 
             nb_prediction, nb_spam_probability,
             lr_prediction, lr_spam_probability,
             svm_prediction, svm_spam_probability,
             rf_prediction, rf_spam_probability,
             lstm_prediction, lstm_spam_probability,
             consensus_prediction, consensus_spam_probability,
             created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (username, 
             result["message"], 
             result.get("Naive Bayes prediction", "UNKNOWN"),
             result.get("Naive Bayes spam probability", 0.0),
             result.get("Logistic Regression prediction", "UNKNOWN"),
             result.get("Logistic Regression spam probability", 0.0),
             result.get("Support Vector Machine prediction", "UNKNOWN"),
             result.get("Support Vector Machine spam probability", 0.0),
             result.get("Random Forest prediction", "UNKNOWN"),
             result.get("Random Forest spam probability", 0.0),
             result.get("LSTM prediction", "UNKNOWN"),
             result.get("LSTM spam probability", 0.0),
             result.get("Consensus prediction", "UNKNOWN"),
             result.get("Consensus spam probability", 0.0),
             datetime.now().isoformat(timespec="seconds")))

def get_predictions(username):
    with connect() as conn:
        rows = conn.execute("SELECT * FROM predictions WHERE username=? ORDER BY id DESC", (username,)).fetchall()
    return [dict(row) for row in rows]

def clear_predictions(username):
    with connect() as conn:
        conn.execute("DELETE FROM predictions WHERE username=?", (username,))

