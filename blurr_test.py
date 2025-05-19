import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
import os

def load_image_from_pkl(pkl_path, index=None):
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)

    images = data_dict['data']  # shape (N, 3072)
    labels = data_dict['labels']

    if index is None:
        index = np.random.randint(0, len(images))

    image = images[index].reshape(3, 32, 32).astype(np.float32)
    label = labels[index]

    # Normalize to (-1, 1)
    image = (image / 255.0 - 0.5) / 0.5

    return image, label, index

def unnormalize(image):
    return np.clip(image * 0.5 + 0.5, 0, 1)

# ✅ 用 trainer.py 版本邏輯（對一張圖模擬）
def apply_motion_blur_like_trainer(data, max_speed):
    sigma = max(0, 0.1 * max_speed - 0.5)

    # 模擬 batch 處理，但這裡只處理一張圖 (N=1)
    data = data.reshape(1, 3, 32, 32)
    blurred_list = []
    for image in data:
        blurred_image = gaussian_filter(image, sigma=(0, sigma, sigma))
        blurred_list.append(blurred_image)

    return np.array(blurred_list)[0]  # 取回第一張

def visualize_compare(original, blurred, max_speed, index, label, save_path="blur_compare.png"):
    original_img = unnormalize(original)
    blurred_img = unnormalize(blurred)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(np.transpose(original_img, (1, 2, 0)))
    axs[0].set_title(f"Original (Label: {label})")
    axs[1].imshow(np.transpose(blurred_img, (1, 2, 0)))
    axs[1].set_title(f"Blurred (Speed={max_speed})")
    for ax in axs: ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"已儲存對比圖：{save_path}")

# ========== 主程式 ==========
if __name__ == "__main__":
    PKL_PATH = "./cifar_noniid_4groups/g0_train.pkl"  # 修改成你的實際路徑
    MAX_SPEED = 15.0  # 你可以在這裡改速度看看
    INDEX = None  # 可指定 index（如 0），或設為 None 隨機選一張

    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"找不到檔案：{PKL_PATH}")

    image, label, idx = load_image_from_pkl(PKL_PATH, index=INDEX)
    blurred = apply_motion_blur_like_trainer(image, MAX_SPEED)
    visualize_compare(image, blurred, MAX_SPEED, idx, label, save_path=f"blur_compare_speed{MAX_SPEED}.png")
