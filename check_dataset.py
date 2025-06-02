import os
import pickle
import numpy as np

# 取消警告顯示
import warnings
warnings.filterwarnings("ignore")

# === 設定區 ===
ORIG_DIR = "./cifar_noniid_4groups"
BLUR_DIR = "./cifar_noniid_blurred"
GROUPS = ['g0', 'g1', 'g2', 'g3']
THRESHOLD = 0.01  # 每張圖平均像素差異大於此閾值就視為模糊

def unnormalize(image):
    return np.clip(image * 0.5 + 0.5, 0, 1)

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compute_blur_ratio(original_data, blurred_data, threshold=0.00001):
    diffs = np.abs(original_data - blurred_data)
    pixel_diffs = diffs.reshape(diffs.shape[0], -1).mean(axis=1)
    blurred_flags = pixel_diffs > threshold
    blur_ratio = np.sum(blurred_flags) / len(blurred_flags)
    return blur_ratio

# === 主程序 ===
print("=== 模糊比例檢查開始 ===")
for group in GROUPS:
    orig_path = os.path.join(ORIG_DIR, f"{group}_train.pkl")
    if not os.path.exists(orig_path):
        print(f"[跳過] 缺少原始資料：{orig_path}")
        continue

    orig_data = load_data(orig_path)['data'].astype(np.float32)
    if orig_data.max() > 1.0:
        orig_data = (orig_data / 255.0 - 0.5) / 0.5  # 標準化

    for speed in range(1, 21):
        blur_path = os.path.join(BLUR_DIR, group, f"speed{speed:02d}_train.pkl")
        if not os.path.exists(blur_path):
            print(f"[缺失] 找不到：{blur_path}")
            continue

        blur_data = load_data(blur_path)['data'].astype(np.float32)

        if blur_data.shape != orig_data.shape:
            print(f"[錯誤] shape 不一致：{blur_path} ({blur_data.shape} ≠ {orig_data.shape})")
            continue

        ratio = compute_blur_ratio(orig_data, blur_data, threshold=THRESHOLD)
        print(f"{group} | speed={speed:2d} | 模糊比例 = {ratio:.2%}")

print("=== 檢查結束 ===")
