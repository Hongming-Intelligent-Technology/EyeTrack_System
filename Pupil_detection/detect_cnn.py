import torch, cv2
from torchvision import transforms
from models.pupil_cnn import PupilCNN

model = PupilCNN()
model.load_state_dict(torch.load("models/pupil_cnn.pth", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(gray,(64,64))
    inp = tf(eye).unsqueeze(0)
    with torch.no_grad():
        x,y = model(inp)[0]
    x,y = x.item()*frame.shape[1], y.item()*frame.shape[0]
    cv2.circle(frame,(int(x),int(y)),5,(0,0,255),-1)
    cv2.imshow("Pupil",frame)
    if cv2.waitKey(1)&0xFF==27: break
