# models/train.py

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import os
import joblib

from dqn import DQN
from environment import MedicalEnv
from replay_buffer import ReplayBuffer


# =========================
# LOAD DATA
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "../data/Training.csv"))

symptom_cols = df.columns[:-1]


# =========================
# LOAD CLASSIFIER
# =========================

clf = joblib.load(os.path.join(BASE_DIR, "../saved_models/classifier.pkl"))

env = MedicalEnv(df, symptom_cols, clf)

state_size = len(symptom_cols)
action_size = len(symptom_cols)


# =========================
# MODELS
# =========================

model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(model.state_dict())


# =========================
# OPTIMIZER
# =========================

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()


# =========================
# REPLAY BUFFER
# =========================

buffer = ReplayBuffer(10000)


# =========================
# HYPERPARAMETERS
# =========================

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.998

batch_size = 64
episodes = 1000

# 🔥 FIXED IG weight (very important)
IG_WEIGHT = 0.5


# =========================
# METRICS
# =========================

rewards_history = []
steps_history = []
success_history = []
loss_history = []

print("🚀 Training started...")


# =========================
# TRAINING LOOP
# =========================

for episode in range(episodes):

    state = env.reset()
    state = torch.FloatTensor(state)

    total_reward = 0
    step_count = 0
    success = 0

    for step in range(15):

        # =========================
        # ACTION SELECTION
        # =========================

        if np.random.rand() < epsilon:
            action = np.random.randint(action_size)

        else:
            with torch.no_grad():

                # 🔥 Q-values
                q_values = model(state).numpy()

                # 🔥 normalize Q-values
                q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)

                # 🔥 mask already asked
                mask = (state.numpy() == 1)
                q_values[mask] = -1e9

                # =========================
                # INFORMATION GAIN
                # =========================

                ig_scores = []

                for a in range(action_size):
                    if a in env.asked:
                        ig_scores.append(-1e9)
                    else:
                        ig_scores.append(env.compute_information_gain(a))

                ig_scores = np.array(ig_scores)

                # 🔥 normalize IG (CRITICAL FIX)
                valid_mask = ig_scores != -1e9
                if np.any(valid_mask):
                    ig_scores[valid_mask] = (
                        (ig_scores[valid_mask] - ig_scores[valid_mask].mean()) /
                        (ig_scores[valid_mask].std() + 1e-8)
                    )

                # =========================
                # COMBINE RL + IG
                # =========================

                combined = q_values + IG_WEIGHT * ig_scores

                action = int(np.argmax(combined))


        # =========================
        # STEP ENV
        # =========================

        next_state, reward, done = env.step(action)
        next_state = torch.FloatTensor(next_state)

        buffer.push(state, action, reward, next_state, done)

        state = next_state

        total_reward += reward
        step_count += 1


        # =========================
        # TRAIN FROM BUFFER
        # =========================

        if len(buffer) > batch_size:

            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states = torch.stack(states)
            next_states = torch.stack(next_states)

            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # current Q
            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

            # target Q
            with torch.no_grad():
                next_q = target_model(next_states).max(1)[0]

            targets = rewards + gamma * next_q * (1 - dones)

            # 🔥 stabilize training
            targets = torch.clamp(targets, -10, 10)

            loss = loss_fn(q_values, targets)

            optimizer.zero_grad()
            loss.backward()

            # 🔥 gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            loss_history.append(loss.item())


        if done:
            break


    # =========================
    # EVALUATION
    # =========================

    top_preds, _ = env.get_top_k(3)

    print("TRUE:", env.true_disease)
    print("TOP 3:", top_preds)
    print("------------------")

    if env.true_disease == top_preds[0]:
        success = 1
    elif env.true_disease in top_preds:
        success = 0.5


    # =========================
    # TARGET UPDATE
    # =========================

    if episode % 20 == 0:
        target_model.load_state_dict(model.state_dict())


    # =========================
    # EPSILON DECAY
    # =========================

    epsilon = max(epsilon_min, epsilon * epsilon_decay)


    # =========================
    # STORE METRICS
    # =========================

    rewards_history.append(total_reward)
    steps_history.append(step_count)
    success_history.append(success)

    if episode % 100 == 0:
        print(
            f"Episode {episode} | Reward: {total_reward:.2f} | "
            f"Steps: {step_count} | Success: {success} | Epsilon: {epsilon:.3f}"
        )


# =========================
# SAVE MODEL
# =========================

os.makedirs("saved_models", exist_ok=True)

torch.save(model.state_dict(), "saved_models/dqn_model.pth")


# =========================
# SAVE METRICS
# =========================

with open("saved_models/training_metrics.pkl", "wb") as f:
    pickle.dump({
        "rewards": rewards_history,
        "steps": steps_history,
        "success": success_history,
        "loss": loss_history
    }, f)


print("\n✅ Training Complete!")
