import torch
import torch.nn as nn
import torchvision.models as models

class GazeResNet(nn.Module):
    def __init__(self, input_channels=16, output_dim=2):
        super(GazeResNet, self).__init__()
        self.encoder = models.resnet18(pretrained=False)

        self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, output_dim)

    def forward(self, x):
        return self.encoder(x)
