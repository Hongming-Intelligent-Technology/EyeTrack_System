import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model_lstm import GazeLSTMNet

def train_lstm(
    train_path='dataset/train.npy',
    label_path='dataset/label.npy',
    model_save_path='dataset/model_lstm.pth',
    epochs=40,
    batch_size=32,
    lr=1e-3
):
    x = np.load(train_path)   # [N, C, T, W]
    y = np.load(label_path)   # [N, 2]
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GazeLSTMNet(input_channels=x.shape[1], width=x.shape[3])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train_lstm()
