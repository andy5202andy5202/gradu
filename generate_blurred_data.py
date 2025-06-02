import os
import pickle
import numpy as np
import cv2

# 輸入與輸出資料夾
INPUT_PATH = "./cifar_noniid_4groups"
OUTPUT_PATH = "./cifar_noniid_blurred"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 固定模糊強度
SIGMA = 0.4

def apply_partial_blur(data, blur_ratio, seed=0):
    """隨機挑選部分樣本進行模糊處理，並回傳實際模糊比例"""
    np.random.seed(seed)
    blurred_data = []
    num_blur = int(len(data) * blur_ratio)
    blur_indices = set(np.random.choice(len(data), num_blur, replace=False))

    for idx, flat_img in enumerate(data):
        img = flat_img.reshape(3, 32, 32)
        if idx in blur_indices:
            img_uint8 = ((img * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0)
            blurred = cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=SIGMA, sigmaY=SIGMA)
            blurred = blurred.transpose(2, 0, 1).astype(np.float32)
            blurred = (blurred / 255.0 - 0.5) / 0.5
            flat_blurred = blurred.reshape(-1)
            blurred_data.append(flat_blurred)
        else:
            blurred_data.append(flat_img)

    blurred_array = np.array(blurred_data, dtype=np.float32)
    diffs = np.abs(data - blurred_array)
    pixel_diff = diffs.reshape(len(data), -1).mean(axis=1)
    true_blur_ratio = np.mean(pixel_diff > 0.01)

    return blurred_array, true_blur_ratio

def process_group(group_name, prefix):
    path = os.path.join(INPUT_PATH, f"{group_name}_{prefix}.pkl")
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    data = raw['data']
    labels = raw['labels']
    print(f"[INFO] 載入 {path} → 數量 {len(labels)}")

    data = data.astype(np.float32)
    if data.max() > 1.0:
        data = (data / 255.0 - 0.5) / 0.5

    group_out_dir = os.path.join(OUTPUT_PATH, group_name)
    os.makedirs(group_out_dir, exist_ok=True)

    for speed in range(1, 21):
        ratio = speed / 100.0
        blurred_data, real_ratio = apply_partial_blur(data, ratio, seed=42 + speed)
        out_dict = {'data': blurred_data, 'labels': np.array(labels)}
        out_path = os.path.join(group_out_dir, f"speed{speed:02d}_{prefix}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(out_dict, f)
        print(f"{group_name} | speed={speed:02d} | 模糊比例設定 = {ratio:.2f}，實際 = {real_ratio:.2%}")

# 處理所有 group 的 train 資料
for group in ['g0', 'g1', 'g2', 'g3']:
    process_group(group, 'train')

print("模糊版訓練資料已重新產生完畢。")