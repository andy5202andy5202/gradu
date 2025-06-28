import os
import pandas as pd
import matplotlib.pyplot as plt

# 設定
base_names = ["test", "test1"]
labels = ["Global Momentum", "Edge Momentum"]
colors = ["tab:blue", "tab:orange"]

# 畫 Loss 曲線
plt.figure(figsize=(10, 5))
for base, label, color in zip(base_names, labels, colors):
    path = f"logs/loss/{base}_loss.csv"
    df = pd.read_csv(path)
    plt.plot(df["round"], df["loss"], label=label, color=color, linestyle='--', marker='o')

plt.xlabel("Global Round")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/loss_comparison.png")
plt.close()

# 畫 Accuracy 曲線
plt.figure(figsize=(10, 5))
for base, label, color in zip(base_names, labels, colors):
    path = f"logs/accuracy/{base}_accuracy.csv"
    df = pd.read_csv(path)
    plt.plot(df["round"], df["accuracy"], label=label, color=color, linestyle='-', marker='x')

plt.xlabel("Global Round")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/accuracy_comparison.png")
plt.close()
