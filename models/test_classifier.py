# models/test_classifier.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score


# =========================
# LOAD DATA
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../data/Training.csv")

df = pd.read_csv(data_path)

X = df.iloc[:, :-1].values
y = np.array([label.strip() for label in df.iloc[:, -1].values])

print("✅ Dataset loaded:", X.shape)


# =========================
# LOAD CLASSIFIER
# =========================

model_path = os.path.join(BASE_DIR, "../saved_models/classifier.pkl")

with open(model_path, "rb") as f:
    clf = pickle.load(f)

print("✅ Classifier loaded!")


# =========================
# FULL SYMPTOM TEST
# =========================

preds = clf.predict(X)

print("\n==============================")
print("✅ FULL SYMPTOM PERFORMANCE")
print("==============================")

acc = accuracy_score(y, preds)
print("Accuracy:", acc)


# =========================
# PARTIAL SYMPTOM TEST
# =========================

print("\n==============================")
print("🧠 PARTIAL SYMPTOM TEST")
print("==============================")

top1_correct = 0
top3_correct = 0
num_samples = 20

for i in range(num_samples):

    idx = np.random.randint(0, len(X))

    x_full = X[idx]
    true_label = y[idx]

    # 🔥 simulate partial symptoms (keep only 30–50%)
    mask = np.random.rand(len(x_full)) < 0.4
    x_partial = x_full * mask

    # ensure at least 1 symptom exists
    if np.sum(x_partial) == 0:
        x_partial[np.random.randint(len(x_partial))] = 1

    x_partial = x_partial.reshape(1, -1)

    top3, probs = clf.predict_top_k(x_partial, 3)

    print(f"\nSample {i+1}")
    print("TRUE:", true_label)
    print("TOP 3:", list(top3))
    print("PROBS:", np.round(probs, 3))

    # ✅ correct comparison (STRING vs STRING)
    if true_label == top3[0]:
        top1_correct += 1

    if true_label in top3:
        top3_correct += 1


# =========================
# FINAL RESULTS
# =========================

print("\n==============================")
print("📊 FINAL RESULTS")
print("==============================")

print("Top-1 Accuracy:", round(top1_correct / num_samples, 3))
print("Top-3 Accuracy:", round(top3_correct / num_samples, 3))
