import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from utils import PupilMaskDataset

dataset = PupilMaskDataset("dataset/images","dataset/masks")
loader = DataLoader(dataset,batch_size=8,shuffle=True)
model = UNet()
opt = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

for epoch in range(30):
    for img, mask in loader:
        opt.zero_grad()
        out = model(img)
        loss = loss_fn(out, mask)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}: {loss.item():.4f}")

torch.save(model.state_dict(),"models/pupil_unet.pth")
