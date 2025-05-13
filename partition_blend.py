import pickle
import numpy as np
import os
from sklearn.model_selection import KFold

INPUT_PATH = "./cifar_noniid_9groups_split"    # 你現在的 g0_train.pkl ~ g8_train.pkl
OUTPUT_PATH = "./cifar_noniid_3groups_mixed"   # 輸出的資料夾
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 每個新 group 對應到原本哪些 group
group_mapping = {
    'g0': ['g0', 'g1', 'g2'],  # labels 0-2
    'g1': ['g3', 'g4', 'g5'],  # labels 3-5
    'g2': ['g6', 'g7', 'g8'],  # labels 6-8
}

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

output_file_index = 0  # 控制輸出的 g0_train ~ g8_train

for new_group, old_groups in group_mapping.items():
    # ======== 讀取並合併舊 group ========
    merged_data = []
    merged_labels = []

    for old_group in old_groups:
        group_file = os.path.join(INPUT_PATH, f"{old_group}_train.pkl")
        group_content = load_pickle(group_file)
        merged_data.append(group_content['data'])
        labels_array = np.array(group_content['labels'], dtype=np.int64)  # ✅ 強制轉 int64
        merged_labels.append(labels_array)

    merged_data = np.vstack(merged_data)
    merged_labels = np.hstack(merged_labels)

    print(f"{new_group} 合併後：{len(merged_labels)} 筆資料")
    print(f"→ 合併後 labels unique: {np.unique(merged_labels)}")  # ✅ 這行確認 label 是否正常

    # ======== 切成三份 =========
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for _, split_indices in kf.split(merged_data):
        split_data = merged_data[split_indices]
        split_labels = merged_labels[split_indices]

        output_filename = os.path.join(OUTPUT_PATH, f"g{output_file_index}_train.pkl")
        save_pickle(split_data, split_labels, output_filename)
        print(f"  已儲存 {output_filename}，數量：{len(split_labels)} 筆，unique labels: {np.unique(split_labels)}")

        output_file_index += 1  # 每份檔案編號 +1

print("重新分組與切分完成，儲存到：", OUTPUT_PATH)
