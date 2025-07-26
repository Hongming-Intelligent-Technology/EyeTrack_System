import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import GazeResNet

def train_model(
    train_path='dataset/train.npy',
    label_path='dataset/label.npy',
    model_save_path='dataset/model.pth',
    epochs=30, batch_size=32, lr=0.001
):
    # 1. 加载数据
    x = np.load(train_path)   # [N, 16, 26, 60]
    y = np.load(label_path)   # [N, 2]

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型
    model = GazeResNet()
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. 训练循环
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 4. 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 模型已保存至 {model_save_path}")

if __name__ == '__main__':
    train_model()
