import random
import torch
import pandas as pd
import numpy as np

from models.dqn import DQN
from models.environment import MedicalEnv


# 🔥 Load data
df = pd.read_csv("data/Training.csv")
symptom_cols = df.columns[:-1]

env = MedicalEnv(df, symptom_cols)

# 🔥 Load trained DQN
model = DQN(len(symptom_cols), len(symptom_cols))
model.load_state_dict(torch.load("saved_models/dqn_model.pth"))
model.eval()


# =========================
# AGENTS
# =========================

def dqn_agent(state):
    with torch.no_grad():
        return torch.argmax(model(state)).item()


def random_agent(state):
    return random.randint(0, len(symptom_cols)-1)


def gemini_agent_mock(state, true_symptoms):
    # 🔥 simulate "intelligent" agent
    possible = [i for i, val in enumerate(true_symptoms) if val == 1]
    return random.choice(possible) if possible else random.randint(0, len(symptom_cols)-1)


# =========================
# EVALUATION
# =========================

def evaluate(agent_fn, episodes=100):

    total_steps = []
    success_count = 0

    for _ in range(episodes):

        state = env.reset()
        state = torch.FloatTensor(state)

        for step in range(20):

            if agent_fn == gemini_agent_mock:
                action = agent_fn(state, env.true_symptoms)
            else:
                action = agent_fn(state)

            next_state, reward, done = env.step(action)
            state = torch.FloatTensor(next_state)

            if done:
                total_steps.append(step+1)

                if reward > 0:
                    success_count += 1
                break

    avg_steps = np.mean(total_steps)
    success_rate = success_count / episodes

    return avg_steps, success_rate


# =========================
# RUN COMPARISON
# =========================

agents = {
    "DQN": dqn_agent,
    "Random": random_agent,
    "Gemini (mock)": gemini_agent_mock
}

results = {}

for name, agent in agents.items():
    steps, success = evaluate(agent)
    results[name] = (steps, success)
    print(f"{name}: Steps={steps:.2f}, Success={success:.2f}")



import matplotlib.pyplot as plt

names = list(results.keys())
steps = [results[n][0] for n in names]
success = [results[n][1] for n in names]

plt.figure(figsize=(10,5))

# Steps
plt.subplot(1,2,1)
plt.bar(names, steps)
plt.title("Avg Questions Asked")
plt.ylabel("Steps")

# Success
plt.subplot(1,2,2)
plt.bar(names, success)
plt.title("Success Rate")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
