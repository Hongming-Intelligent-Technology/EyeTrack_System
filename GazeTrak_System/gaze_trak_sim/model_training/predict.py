import torch
import numpy as np
from model import GazeResNet

def predict(
    model_path='dataset/model.pth',
    test_echo_path='dataset/test.npy'
):
    model = GazeResNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_data = np.load(test_echo_path)   # [N, 16, 26, 60]
    x_tensor = torch.tensor(test_data, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(x_tensor).numpy()  # [N, 2]

    return y_pred

if __name__ == '__main__':
    preds = predict()
    print("预测 gaze 坐标:", preds[:5])
