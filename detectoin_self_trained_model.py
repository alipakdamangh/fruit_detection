import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IP_CAMERA_URL = 'http://192.168.1.128:8080/video'
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_INPUT_SIZE = (224, 224)
DISPLAY_SIZE = (900, 600)
CLASS_NAMES = ['freshapples', 'freshoranges', 'rottenapples', 'rottenoranges']
LABEL_MAP = {
    'freshapples': 'Fresh Apple',
    'freshoranges': 'Fresh Orange',
    'rottenapples': 'Rotten Apple',
    'rottenoranges': 'Rotten Orange'
}
CONFIDENCE_THRESHOLD = 0.7

print("Loading model...")
try:
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model weights loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    print("Please ensure MODEL_PATH is correct and the file exists.")
    exit()
except Exception as e:
    print(f"FATAL ERROR loading model weights: {e}")
    print("Ensure the model architecture (resnet50) matches the saved weights.")
    exit()

model = model.to(DEVICE).eval()
print("Model set to evaluation mode.")

print("Setting up preprocessing transforms...")
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])
print("Preprocessing transforms defined.")

WINDOW_NAME = 'Fruit Classification'
print(f"Creating display window: {WINDOW_NAME}")
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
print(f"Display window created with size {DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]}")

print(f"Opening video stream from {IP_CAMERA_URL}...")
cap = cv2.VideoCapture(IP_CAMERA_URL)

if not cap.isOpened():
    print(f"FATAL ERROR: Unable to open video stream at {IP_CAMERA_URL}")
    print("Please check the IP camera URL and ensure the stream is active.")
    exit()

print("Video stream opened. Press 'q' to exit.")

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (0, 255, 0)
LINE_TYPE = 2
TEXT_ORIGIN = (10, 30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            top_prob, top_idx = probs.max(0)

        predicted_class_name = CLASS_NAMES[top_idx.item()]
        confidence = top_prob.item()

        display_text = "Detecting..."
        if confidence >= CONFIDENCE_THRESHOLD:
            display_label = LABEL_MAP.get(predicted_class_name, predicted_class_name)
            display_text = f"{display_label}: {confidence*100:.1f}%"

        cv2.putText(frame, display_text, TEXT_ORIGIN, FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
        display_frame = cv2.resize(frame, DISPLAY_SIZE)
        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' key pressed. Exiting.")
            break

except Exception as e:
    print(f"\nAn error occurred during streaming or inference: {e}")

finally:
    print("Releasing video stream and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")
