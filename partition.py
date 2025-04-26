import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/cifar-10-batches-py"   # 原始 CIFAR-10 資料夾
OUTPUT_PATH = "./cifar_noniid_9groups_split"   # 拆好的輸出資料夾
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 只保留 g0 ~ g8，不使用 label 9
groups = {
    'g0': [0],  # airplane
    'g1': [1],  # automobile
    'g2': [2],  # bird
    'g3': [3],  # cat
    'g4': [4],  # deer
    'g5': [5],  # dog
    'g6': [6],  # frog
    'g7': [7],  # horse
    'g8': [8],  # ship
}

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def save_data(data, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

# ======== 讀取原始 CIFAR-10 ==========
all_data = []
all_labels = []

for i in range(1, 6):  # 5 個 batch
    batch = unpickle(os.path.join(DATA_PATH, f'data_batch_{i}'))
    all_data.append(batch[b'data'])
    all_labels.extend(batch[b'labels'])

all_data = np.vstack(all_data)
all_labels = np.array(all_labels)

# ======== 分組並拆分 train/test ==========
for group_name, label_list in groups.items():
    indices = [i for i, label in enumerate(all_labels) if label in label_list]
    group_data = all_data[indices]
    group_labels = all_labels[indices]

    # 8:2 切分
    X_train, X_test, y_train, y_test = train_test_split(
        group_data, group_labels, test_size=0.2, stratify=group_labels, random_state=42
    )

    save_data(X_train, y_train, os.path.join(OUTPUT_PATH, f'{group_name}_train.pkl'))
    save_data(X_test, y_test, os.path.join(OUTPUT_PATH, f'{group_name}_test.pkl'))

    print(f"{group_name}: 訓練集 {len(y_train)} 筆，測試集 {len(y_test)} 筆（Label={label_list[0]}）")

print("✅ 拆分完成，已儲存到：", OUTPUT_PATH)
