import torch
import numpy as np
from model_lstm import GazeLSTMNet

def predict_lstm(
    model_path='dataset/model_lstm.pth',
    test_path='dataset/train.npy',
    save_path='dataset/pred.npy'
):
    model = GazeLSTMNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = np.load(test_path)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(x_tensor).numpy()

    np.save(save_path, y_pred)
    print(f"Predicted gaze saved to {save_path}")
    return y_pred

if __name__ == '__main__':
    predict_lstm()
