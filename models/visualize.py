# import pickle
# import matplotlib.pyplot as plt
# import numpy as np

# with open("saved_models/training_metrics.pkl", "rb") as f:
#     data = pickle.load(f)

# rewards = data["rewards"]
# steps = data["steps"]
# success = data["success"]
# loss = data["loss"]

# # 🔥 Smooth function
# def smooth(x, window=50):
#     return np.convolve(x, np.ones(window)/window, mode='valid')


# plt.figure(figsize=(12, 8))

# # Reward
# plt.subplot(2,2,1)
# plt.title("Reward Over Time")
# plt.plot(smooth(rewards))
# plt.xlabel("Episodes")

# # Steps
# plt.subplot(2,2,2)
# plt.title("Questions Asked")
# plt.plot(smooth(steps))
# plt.xlabel("Episodes")

# # Success
# plt.subplot(2,2,3)
# plt.title("Success Rate")
# plt.plot(smooth(success))
# plt.xlabel("Episodes")

# # Loss
# plt.subplot(2,2,4)
# plt.title("Loss")
# plt.plot(loss)
# plt.xlabel("Training Steps")

# plt.tight_layout()
# plt.show()




import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("saved_models/training_metrics.pkl", "rb") as f:
    data = pickle.load(f)

rewards = data["rewards"]
steps = data["steps"]
success = data["success"]
loss = data["loss"]

def smooth(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.figure(figsize=(14, 10))

# 🔥 Reward Trend
plt.subplot(2,2,1)
plt.title("Reward Trend (Learning Progress)")
plt.plot(smooth(rewards))
plt.xlabel("Episodes")
plt.ylabel("Reward")

# 🔥 Efficiency
plt.subplot(2,2,2)
plt.title("Questions Asked (Efficiency)")
plt.plot(smooth(steps))
plt.xlabel("Episodes")
plt.ylabel("Questions")

# 🔥 Success Rate
plt.subplot(2,2,3)
plt.title("Diagnosis Success Rate")
plt.plot(smooth(success))
plt.xlabel("Episodes")
plt.ylabel("Success (0/1)")

# 🔥 Loss Stability
plt.subplot(2,2,4)
plt.title("Loss (Training Stability)")
plt.plot(loss)
plt.xlabel("Steps")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()
