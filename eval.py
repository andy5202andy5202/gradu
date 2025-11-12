import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

eval_dir = 'evaluation_logs'
target_eps = [10,20,30,40,50,60,70,80,90,100]
colors = plt.cm.viridis(np.linspace(0, 1, len(target_eps)))
rounds = list(range(1, 16))  # 固定 x 軸為 1~20

# ====== 收集所有 loss 決定 y 軸刻度 ======
all_losses = []
for ep in target_eps:
    df = pd.read_csv(os.path.join(eval_dir, f"eval_episode_{ep}_full_rounds.csv"))
    all_losses.extend(df['global_loss'][:15])

loss_min = math.floor(min(all_losses) * 10) / 10
loss_max = math.ceil(max(all_losses) * 10) / 10
loss_ticks = np.arange(loss_min, loss_max + 0.01, 0.1)

# ====== 畫 Loss 圖 ======
plt.figure(figsize=(10, 5))
for idx, ep in enumerate(target_eps):
    df = pd.read_csv(os.path.join(eval_dir, f"eval_episode_{ep}_full_rounds.csv"))
    plt.plot(rounds, df['global_loss'][:15], marker='o', linestyle='--', color=colors[idx], label=f"Eval Ep {ep}")
plt.title("Global Loss over 15 Rounds")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.xticks(rounds)
plt.yticks(loss_ticks)
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("logs", exist_ok=True)
plt.savefig("logs/evaluation_loss_10_20_30.png")
plt.close()

# ====== 畫 Accuracy 圖 ======
plt.figure(figsize=(10, 5))
for idx, ep in enumerate(target_eps):
    df = pd.read_csv(os.path.join(eval_dir, f"eval_episode_{ep}_full_rounds.csv"))
    plt.plot(rounds, df['global_accuracy'][:15], marker='x', linestyle='-', color=colors[idx], label=f"Eval Ep {ep}")
plt.title("Global Accuracy over 15 Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.xticks(rounds)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/evaluation_accuracy_10_20_30.png")
plt.close()
