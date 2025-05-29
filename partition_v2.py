import pickle
import numpy as np
import os

DATA_PATH = "./data/cifar-10-batches-py"  # CIFAR-10 原始資料
OUTPUT_PATH = "./cifar_noniid_4groups"    # 輸出資料夾
os.makedirs(OUTPUT_PATH, exist_ok=True)

# g0~g3 分別對應 0~2, 3~4, 5~6, 7~9 三群
group_mapping = {
    'g0': [0, 1, 2],
    'g1': [3, 4],
    'g2': [5, 6],
    'g3': [7, 8, 9]
}

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def save_group_data(data_dict, prefix):
    for group_name, content in data_dict.items():
        path = os.path.join(OUTPUT_PATH, f"{group_name}_{prefix}.pkl")
        with open(path, 'wb') as f:
            pickle.dump({'data': np.array(content['data']), 'labels': np.array(content['labels'])}, f)
        print(f"{group_name}_{prefix}.pkl → 數量: {len(content['labels'])}")

# 初始化 group 資料結構
train_groups = {g: {'data': [], 'labels': []} for g in group_mapping}
test_groups = {g: {'data': [], 'labels': []} for g in group_mapping}

# 處理訓練集（data_batch_1 ~ data_batch_5）
for i in range(1, 6):
    batch = unpickle(os.path.join(DATA_PATH, f"data_batch_{i}"))
    data = batch[b'data']
    labels = batch[b'labels']
    for x, y in zip(data, labels):
        for group_name, label_list in group_mapping.items():
            if y in label_list:
                train_groups[group_name]['data'].append(x)
                train_groups[group_name]['labels'].append(y)
                break

# 處理測試集（test_batch）
test_batch = unpickle(os.path.join(DATA_PATH, "test_batch"))
test_data = test_batch[b'data']
test_labels = test_batch[b'labels']
for x, y in zip(test_data, test_labels):
    for group_name, label_list in group_mapping.items():
        if y in label_list:
            test_groups[group_name]['data'].append(x)
            test_groups[group_name]['labels'].append(y)
            break

# 儲存結果
save_group_data(train_groups, "train")
save_group_data(test_groups, "test")

print("分群與儲存完成")
