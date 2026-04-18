# models/train_classifier.py

import pandas as pd
import numpy as np
import os
import joblib

from classifier import DiseaseClassifier


# =========================
# LOAD DATA
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../data/Training.csv")

df = pd.read_csv(data_path)

print("✅ Dataset loaded:", df.shape)


# =========================
# PREPARE DATA
# =========================

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# clean labels
y = np.array([label.strip() for label in y])


# =========================
# TRAIN
# =========================

clf = DiseaseClassifier()

print("🚀 Training classifier...")
clf.train(X, y)
print("✅ Training complete!")


# =========================
# SAVE MODEL (FIXED)
# =========================

save_dir = os.path.join(BASE_DIR, "../saved_models")
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "classifier.pkl")

joblib.dump(clf, model_path)

print(f"💾 Model saved at: {model_path}")


# =========================
# SANITY CHECK
# =========================

sample = X[0]

pred = clf.predict(sample)
top3, probs = clf.predict_top_k(sample, 3)

print("\n🔍 Sample Check:")
print("TRUE:", y[0])
print("PRED:", pred[0])
print("TOP 3:", top3)
print("PROBS:", np.round(probs, 3))
