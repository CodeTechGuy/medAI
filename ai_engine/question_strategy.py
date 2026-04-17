# ai_engine/question_strategy.py

import torch
import random


def get_next_question(current_symptoms, model, encoder, graph):

    state_vec = encoder.encode(current_symptoms)
    state = torch.FloatTensor(state_vec)

    with torch.no_grad():
        q_values = model(state)

    q_values = q_values + torch.randn_like(q_values) * 0.01

    all_symptoms = encoder.get_all_symptoms()

    valid_actions = []

    # 🔥 filter valid symptoms
    for i, symptom in enumerate(all_symptoms):

        # ❌ skip already known
        if symptom in current_symptoms:
            continue

        # ❌ skip irrelevant (DATA-DRIVEN via graph)
        if not graph.is_relevant(symptom, current_symptoms):
            continue

        valid_actions.append((i, q_values[i].item()))

    # 🔥 fallback if nothing valid
    if not valid_actions:
        return fallback_question(current_symptoms)

    # 🔥 sort by Q-value
    valid_actions.sort(key=lambda x: x[1], reverse=True)

    # 🔥 TOP-K sampling (fixes bias like "swelled lymph nodes")
    k = min(3, len(valid_actions))
    top_k = valid_actions[:k]

    # 🔥 weighted random (better than pure random)
    scores = [v[1] for v in top_k]
    probs = softmax(scores)

    chosen_idx = random.choices(range(k), weights=probs)[0]

    chosen_action = top_k[chosen_idx][0]
    symptom = all_symptoms[chosen_action]

    return humanize_question(symptom)


# 🔥 softmax for probabilistic selection
def softmax(x):
    import math
    exps = [math.exp(i) for i in x]
    s = sum(exps)
    return [e / s for e in exps]


# 🔥 human-friendly questions
def humanize_question(symptom):

    symptom = symptom.replace("_", " ")

    templates = [
        f"Are you experiencing {symptom}?",
        f"Have you noticed any {symptom}?",
        f"Do you have {symptom}?",
        f"Any signs of {symptom}?",
        f"Just checking — do you feel {symptom}?"
    ]

    return random.choice(templates)


# 🔥 fallback (used when DQN fails)
def fallback_question(current_symptoms):

    if not current_symptoms:
        return "Can you describe your main symptoms?"

    if "cough" in current_symptoms:
        return "Are you experiencing fever?"

    if "high_fever" in current_symptoms:
        return "Do you have headache?"

    return "Can you tell me more about your symptoms?"

