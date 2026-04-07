#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import string
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Load dataset
df = pd.read_csv("reddit_depression_suicidewatch.csv")

df = df[["text", "label"]].copy()
df["text"] = df["text"].fillna("").astype(str)
df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

label_map = {
    "depression": 0,
    "SuicideWatch": 1
}
df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

X = df["clean_text"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

threshold_balanced = 0.50
threshold_safety = 0.25

y_test_probs_balanced = model.predict_proba(X_test_tfidf)[:, 1]
y_test_pred_balanced = (y_test_probs_balanced >= threshold_balanced).astype(int)

y_test_probs_safety = model.predict_proba(X_test_tfidf)[:, 1]
y_test_pred_safety = (y_test_probs_safety >= threshold_safety).astype(int)

metrics = {
    "balanced": {
        "threshold": threshold_balanced,
        "accuracy": accuracy_score(y_test, y_test_pred_balanced),
        "f1_class1": f1_score(y_test, y_test_pred_balanced, pos_label=1),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred_balanced),
        "report": classification_report(y_test, y_test_pred_balanced, output_dict=True)
    },
    "safety": {
        "threshold": threshold_safety,
        "accuracy": accuracy_score(y_test, y_test_pred_safety),
        "f1_class1": f1_score(y_test, y_test_pred_safety, pos_label=1),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred_safety),
        "report": classification_report(y_test, y_test_pred_safety, output_dict=True)
    }
}

feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

top_positive_idx = np.argsort(coef)[-20:][::-1]
top_negative_idx = np.argsort(coef)[:20]

top_positive = [(feature_names[i], float(coef[i])) for i in top_positive_idx]
top_negative = [(feature_names[i], float(coef[i])) for i in top_negative_idx]

artifacts = {
    "vectorizer": vectorizer,
    "model": model,
    "metrics": metrics,
    "top_positive": top_positive,
    "top_negative": top_negative
}

joblib.dump(artifacts, os.path.join(ARTIFACT_DIR, "mental_health_model.joblib"))

print("Saved artifacts to artifacts/mental_health_model.joblib")

