# utils/bert_nlp.py

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔥 Add synonym mapping
SYNONYMS = {
    "cold": "common_cold",
    "cough": "cough",
    "fever": "high_fever",
    "body pain": "muscle_pain",
    "stomach pain": "abdominal_pain",
    "tired": "fatigue",
}


def extract_symptoms(text, symptom_list):
    text = text.lower()

    found = []

    # ✅ Step 1: Direct synonyms
    for word, mapped in SYNONYMS.items():
        if word in text:
            found.append(mapped)

    # ✅ Step 2: Semantic matching
    cleaned = [s.replace("_", " ") for s in symptom_list]

    user_emb = model.encode(text, convert_to_tensor=True)
    sym_emb = model.encode(cleaned, convert_to_tensor=True)

    scores = util.cos_sim(user_emb, sym_emb)[0]

    for i, score in enumerate(scores):
        if score > 0.45:   # threshold
            found.append(symptom_list[i])

    return list(set(found))


def semantic_match(user_input, symptom_list):
    user_emb = model.encode(user_input, convert_to_tensor=True)
    symptom_emb = model.encode(symptom_list, convert_to_tensor=True)

    scores = util.cos_sim(user_emb, symptom_emb)[0]

    matched = [
        symptom_list[i]
        for i, score in enumerate(scores)
        if score > 0.5
    ]

    return matched
