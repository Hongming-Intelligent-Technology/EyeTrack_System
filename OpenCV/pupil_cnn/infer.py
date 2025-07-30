import torch
import cv2
import numpy as np
from torchvision import transforms
from model.pupil_cnn import PupilCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PupilCNN().to(device)
model.load_state_dict(torch.load("pupil_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def detect_pupil_from_image(img_path):
    img = cv2.imread(img_path)
    input_img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_img)
        pred = output.item()
        if pred > 0.5:
            print("Detected pupil ✅")
        else:
            print("No pupil ❌")

# 示例
detect_pupil_from_image("test_eye.jpg")
