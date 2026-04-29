# models/environment.py

import numpy as np


class MedicalEnv:

    def __init__(self, df, symptom_cols, classifier):
        self.df = df
        self.symptom_cols = list(symptom_cols)
        self.clf = classifier

        self.n_symptoms = len(self.symptom_cols)


        self.state = None
        self.sample_symptoms = None
        self.true_disease = None


        # Precompute dataset for IG
        self.X = df[self.symptom_cols].values
        self.y = df["prognosis"].values
        self.conf_threshold = 0.6   # you can tune (0.75–0.9)

    # =========================
    # RESET ENV
    # =========================

    def reset(self):

        # pick random patient
        sample = self.df.sample(1)

        # 🔥 store symptoms (VERY IMPORTANT)
        self.sample_symptoms = sample.iloc[0, :-1].values

        # store true disease
        self.true_disease = sample.iloc[0, -1]

        # initialize state (all unknown)
        self.state = np.zeros(len(self.symptom_cols))

        self.asked = set()
        self.prev_preds = set()
        self.prev_conf = 0

        return self.state

    # =========================
    # STEP FUNCTION
    # =========================

    def step(self, action):

        reward = 0

        # =========================
        # INVALID ACTION
        # =========================
        if action in self.asked:
            return self.state, -5, False

        self.asked.add(action)

        # =========================
        # ASK SYMPTOM
        # =========================
        if self.sample_symptoms[action] == 1:
            self.state[action] = 1
            reward += 2
        else:
            reward -= 1

        # =========================
        # 🔥 CLASSIFIER PREDICTION (MOVE UP)
        # =========================
        top_preds, probs = self.clf.predict_top_k(self.state, 3)

        # =========================
        # 🔥 DIVERSITY PENALTY (NOW SAFE)
        # =========================
        if top_preds[0] in self.prev_preds:
            reward -= 2

        self.prev_preds.add(top_preds[0])

        # =========================
        # 🔥 CONFIDENCE REWARD
        # =========================
        current_conf = probs[0]
        reward += 3 * np.sqrt(current_conf)

        # confidence improvement bonus
        reward += (current_conf - self.prev_conf) * 10
        self.prev_conf = current_conf

        # =========================
        # 🔥 STOP CONDITION
        # =========================
        done = False

        if current_conf >= self.conf_threshold:
            done = True

            if top_preds[0] == self.true_disease:
                reward += 40
            else:
                reward -= 10

        elif len(self.asked) >= 12:
            done = True

            if top_preds[0] == self.true_disease:
                reward += 20
            elif self.true_disease in top_preds:
                reward += 5
            else:
                reward -= 10

        # =========================
        # 🔥 EFFICIENCY BONUS
        # =========================
        reward += 1.0 / (len(self.asked) + 1)

        if len(self.asked) > 8:
            reward -= 0.5

        return self.state, reward, done


    # =========================
    # TOP-K PREDICTION
    # =========================

    def get_top_k(self, k=3):
        return self.clf.predict_top_k(self.state.copy(), k)

    # =========================
    # INFORMATION GAIN
    # =========================

    def compute_information_gain(self, action):

        if action in self.asked:
            return -1e9

        col = self.df.iloc[:, action]
        disease = self.df.iloc[:, -1]

        total_entropy = self._entropy(disease)

        cond_entropy = 0
        for val in [0, 1]:
            subset = disease[col == val]
            if len(subset) > 0:
                cond_entropy += (len(subset)/len(disease)) * self._entropy(subset)

        ig = total_entropy - cond_entropy

        # 🔥 normalize IG
        return ig / (total_entropy + 1e-8)
    
    
    def _entropy(self, y):

        import numpy as np

        # count occurrences
        values, counts = np.unique(y, return_counts=True)

        # probabilities
        probs = counts / counts.sum()

        # avoid log(0)
        probs = probs + 1e-10

        # entropy formula
        return -np.sum(probs * np.log2(probs))
    