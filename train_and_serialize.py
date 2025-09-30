# train_and_serialize.py
import os, json, re
from pathlib import Path
from html import unescape
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT = Path(__file__).parent.resolve()
DATA_CSV = PROJECT / "twitter_disaster.csv"   # put your CSV here
OUT_DIR = PROJECT / "model"
OUT_DIR.mkdir(exist_ok=True)

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s)
    s = unescape(s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'@\w+', ' ', s)
    s = re.sub(r'#', ' ', s)
    s = re.sub(r'[^0-9A-Za-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()

def main():
    assert DATA_CSV.exists(), f"Place your dataset at {DATA_CSV}"
    df = pd.read_csv(DATA_CSV)
    assert 'text' in df.columns and 'target' in df.columns, "CSV must have 'text' and 'target' columns"
    df['clean_text'] = df['text'].fillna("").apply(clean_text)

    X = df['clean_text']; y = df['target'].astype(int)

    # train/val/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full)

    tfv = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english', min_df=3)
    X_train_tfidf = tfv.fit_transform(X_train)
    X_val_tfidf = tfv.transform(X_val)
    X_test_tfidf = tfv.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # choose threshold on validation set (maximize F1)
    val_probs = clf.predict_proba(X_val_tfidf)[:,1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1, best_t = f, float(t)

    print(f"Chosen threshold={best_t:.3f}, val F1={best_f1:.3f}")

    # test
    test_probs = clf.predict_proba(X_test_tfidf)[:,1]
    test_preds = (test_probs >= best_t).astype(int)
    print("Test F1:", f1_score(y_test, test_preds))

    # save artifacts
    joblib.dump(clf, OUT_DIR / "best_model.joblib")
    joblib.dump(tfv, OUT_DIR / "tfidf_vectorizer.joblib")
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump({"threshold": best_t}, f)

    print("Saved to", OUT_DIR)

if __name__ == "__main__":
    main()
