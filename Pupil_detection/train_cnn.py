import torch, torch.optim as optim
from torch.utils.data import DataLoader
from models.pupil_cnn import PupilCNN
from utils import PupilDataset
import torch.nn as nn

dataset = PupilDataset("dataset/images","dataset/labels.txt")
loader = DataLoader(dataset,batch_size=32,shuffle=True)
model = PupilCNN()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(20):
    for imgs, coords in loader:
        opt.zero_grad()
        out = model(imgs)
        loss = loss_fn(out, coords)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}: loss {loss.item():.4f}")

torch.save(model.state_dict(),"models/pupil_cnn.pth")
