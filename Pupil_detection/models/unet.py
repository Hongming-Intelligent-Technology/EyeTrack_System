import torch, torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(1,64)
        self.d2 = DoubleConv(64,128)
        self.u1 = DoubleConv(128,64)
        self.outc = nn.Conv2d(64,1,1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
    def forward(self,x):
        x1 = self.d1(x)
        x2 = self.pool(x1)
        x3 = self.d2(x2)
        x4 = self.up(x3)
        x5 = self.u1(torch.cat([x4,x1],dim=1))
        return torch.sigmoid(self.outc(x5))
