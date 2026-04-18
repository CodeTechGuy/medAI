# models/classifier.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class DiseaseClassifier:

    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            class_weight="balanced",
            random_state=42
        )

        self.encoder = LabelEncoder()
        self.is_trained = False

    # =========================
    # TRAIN
    # =========================
    def train(self, X, y):

        y_encoded = self.encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

        self.is_trained = True

    # =========================
    # PREDICT (TOP 1)
    # =========================
    def predict(self, X):

        if not self.is_trained:
            raise Exception("Classifier not trained!")

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        pred = self.model.predict(X)
        return self.encoder.inverse_transform(pred)

    # =========================
    # TOP-K PREDICTION
    # =========================
    def predict_top_k(self, X, k=3):

        if not self.is_trained:
            raise Exception("Classifier not trained!")

        # 🔥 FIX: ensure 2D input
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        probs = self.model.predict_proba(X)[0]

        # 🔥 get top-k indices
        idx = np.argsort(probs)[-k:][::-1]

        # 🔥 SAFER mapping
        diseases = self.encoder.classes_[idx]

        return diseases.tolist(), probs[idx]
    