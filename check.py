import torch
print(torch.cuda.is_available())  # 如果是 True 表示你的 PyTorch 支援 GPU
print(torch.cuda.device_count())  # 看看有幾個 GPU 可用
print(torch.cuda.get_device_name(0))  # 看看你的 GPU 名稱
