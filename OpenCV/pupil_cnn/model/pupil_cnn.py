import torch
import torch.nn as nn
import torch.nn.functional as F

class PupilCNN(nn.Module):
    def __init__(self):
        super(PupilCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # 输入图片大小为 64x64 时，计算展平后的大小
        # conv1: 64 -> 60, pool -> 30
        # conv2: 30 -> 28, pool -> 14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 1)  # 二分类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 30, 30]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 14, 14]
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
