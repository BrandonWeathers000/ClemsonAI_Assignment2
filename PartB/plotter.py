import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("blackjack_training.csv")

plt.figure(figsize=(12, 6))
plt.plot(
    data["Episode"],
    data["Average_Reward_1000"],
    label="Moving average (1,000 episodes)"
)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Training episode")
plt.ylabel("Average reward per episode")
plt.title("Blackjack Q-learning training progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
