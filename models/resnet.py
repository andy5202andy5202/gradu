# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# 10層
# class SmallResNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SmallResNet, self).__init__()
        
#         self.in_planes = 16  # 初始 channel 減少為 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
        
#         # 每層只剩一個 block，channel 數量縮減
#         self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=1, stride=1)
#         self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=1, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=1, stride=2)
#         # ❌ layer4 砍掉了

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(64, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out


#15層
# class SmallResNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SmallResNet, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)

#         self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=2, stride=1)
#         self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=1, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(64, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for s in strides:
#             layers.append(block(self.in_planes, planes, s))
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out

#18層(簡化)
# class SmallResNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SmallResNet, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)

#         # 每層 2 個 block
#         self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=2, stride=1)
#         self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(64, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)  # 第一個 block 決定 stride
#         layers = []
#         for s in strides:
#             layers.append(block(self.in_planes, planes, s))
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out

# 18層(有layer4)
# class SmallResNet(nn.Module):
#     def __init__(self, num_classes=9):
#         super(SmallResNet, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)

#         self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=2, stride=1)
#         self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=2)
#         self.layer4 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)  # ← 新增

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, num_classes)  # ← 對應上方輸出 channel

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for s in strides:
#             layers.append(block(self.in_planes, planes, s))
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)  # ← 新增
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out


#完整官方版18層
import torch.nn as nn
from torchvision.models import resnet18

def convert_bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=32, num_channels=child.num_features)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.model = resnet18(weights=None)
        convert_bn_to_gn(self.model)  # ✅ 替換所有 BatchNorm 為 GroupNorm
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# simple CNN 
# class CIFAR_CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CIFAR_CNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8x8
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.fc(x)
#         return x

# MobileNet
# import torch
# import torch.nn as nn
# from torchvision.models import mobilenet_v2

# class SmallResNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SmallResNet, self).__init__()
#         # 載入 MobileNetV2 架構，但不使用預訓練權重
#         self.model = mobilenet_v2(weights=None)

#         # 調整第一層：CIFAR-10 是 32x32，官方模型是為了 224x224
#         self.model.features[0][0] = nn.Conv2d(
#             in_channels=3, out_channels=32,
#             kernel_size=3, stride=1, padding=1, bias=False
#         )

#         # 調整 classifier 輸出
#         self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

#     def forward(self, x):
#         return self.model(x)
