# utils/preprocessing.py

import pandas as pd

class SymptomEncoder:
    def __init__(self, training_path):
        self.df = pd.read_csv(training_path)
        self.symptom_cols = self.df.columns[:-1]  # exclude prognosis
        self.symptom_to_index = {
            symptom: idx for idx, symptom in enumerate(self.symptom_cols)
        }

    def encode(self, symptoms_list):
        """Convert list of symptoms → binary vector"""
        vector = [0] * len(self.symptom_cols)

        for symptom in symptoms_list:
            if symptom in self.symptom_to_index:
                idx = self.symptom_to_index[symptom]
                vector[idx] = 1

        return vector

    def get_all_symptoms(self):
        return list(self.symptom_cols)
    
    