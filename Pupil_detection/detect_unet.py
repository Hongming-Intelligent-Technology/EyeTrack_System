import cv2, torch, numpy as np
from models.unet import UNet
from torchvision import transforms

model = UNet()
model.load_state_dict(torch.load("models/pupil_unet.pth",map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inp = tf(gray).unsqueeze(0)
    with torch.no_grad():
        mask = model(inp)[0,0].numpy()
    mask = cv2.resize((mask>0.5).astype(np.uint8),(frame.shape[1],frame.shape[0]))
    # 质心
    M = cv2.moments(mask)
    if M['m00']>0:
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
    cv2.imshow("UNet Pupil", frame)
    if cv2.waitKey(1)&0xFF==27: break
