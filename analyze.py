import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 讀取資料
csv_path = "logs/train_stats.csv"
df = pd.read_csv(csv_path, header=None)
df.columns = [
    "vehicle_id", "compute_power", "batch_size", "epochs",
    "total_time", "avg_epoch_time", "sim_delay"
]

# seaborn 樣式
sns.set(style="whitegrid")

# 統計每台車參與訓練的次數
train_counts = df["vehicle_id"].value_counts().reset_index()
train_counts.columns = ["vehicle_id", "train_count"]

# 合併回原 df（保留 compute_power 資訊）
df_merged = pd.merge(df, train_counts, on="vehicle_id", how="left")

# 每台車只保留一筆 → 確保不重複統計 train_count
unique_vehicles = df_merged.drop_duplicates(subset=["vehicle_id"])

# 分組後計算正確平均 train_count
avg_train_count = unique_vehicles.groupby("compute_power")["train_count"].mean()

# 其他統計（用原始 df）
grouped = df_merged.groupby("compute_power")
summary = pd.DataFrame({
    "num_vehicles": grouped["vehicle_id"].nunique(),
    "epochs": grouped["epochs"].mean(),
    "total_time": grouped["total_time"].mean(),
    "avg_epoch_time": grouped["total_time"].sum() / grouped["epochs"].sum(),
    "sim_delay": grouped["sim_delay"].first(),
    "avg_train_count": avg_train_count  # ✅ 正確版本
})

print("=== Extended Summary by Compute Power ===")
print(summary)

# 建立 logs 資料夾
os.makedirs("logs", exist_ok=True)

# 畫 avg_train_count
plt.figure(figsize=(6, 4))
sns.barplot(data=summary.reset_index(), x="compute_power", y="avg_train_count")
plt.title("Avg Train Count per Vehicle vs Compute Power")
plt.xlabel("Compute Power")
plt.ylabel("Avg Train Count")
plt.tight_layout()
plt.savefig("logs/plot_avg_train_count.png")
plt.show()
