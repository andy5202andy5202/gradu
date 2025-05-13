import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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

#18層
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

# 完整18層
class SmallResNet(nn.Module):
    def __init__(self, num_classes=9):
        super(SmallResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)  # ← 新增

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)  # ← 對應上方輸出 channel

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # ← 新增
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


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
#         return F.relu(out)


# class SmallResNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SmallResNet, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)

#         self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=2, stride=1)
#         self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=2, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=2)
#         self.layer4 = self._make_layer(BasicBlock, 64, num_blocks=1, stride=1)  # 🔼 新增但保持 channel 不變

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.3)  # 🔼 新增 Dropout 抗 overfit
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
#         out = self.layer4(out)  
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.dropout(out)  
#         out = self.fc(out)
#         return out
