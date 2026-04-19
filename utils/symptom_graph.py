import pandas as pd
import numpy as np


class SymptomGraph:

    def __init__(self, training_path):

        df = pd.read_csv(training_path)

        self.symptoms = df.columns[:-1]
        data = df[self.symptoms].values

        # 🔥 co-occurrence matrix
        self.co_matrix = np.dot(data.T, data)

        # normalize
        self.co_matrix = self.co_matrix / (np.max(self.co_matrix) + 1e-5)

        self.symptom_index = {
            s: i for i, s in enumerate(self.symptoms)
        }

    def get_related(self, symptom, top_k=5):

        if symptom not in self.symptom_index:
            return []

        idx = self.symptom_index[symptom]

        scores = self.co_matrix[idx]

        sorted_idx = np.argsort(scores)[::-1]

        related = [
            self.symptoms[i]
            for i in sorted_idx[1:top_k+1]
        ]

        return related

    def is_relevant(self, candidate, current_symptoms):

        if not current_symptoms:
            return True

        for s in current_symptoms:
            related = self.get_related(s, top_k=10)

            if candidate in related:
                return True

        return False
    
    