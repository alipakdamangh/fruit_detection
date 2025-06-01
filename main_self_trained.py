# Import necessary libraries
import cv2  # OpenCV library for image processing and computer vision
import torch  # PyTorch library for tensor computations and deep learning
import torch.nn as nn  # Neural network modules from PyTorch
import torch.nn.functional as F  # Functional interface for neural network layers
from torchvision import models, transforms  # Pretrained models and image transforms
from PIL import Image  # Python Imaging Library (for image format conversion)
import time  # Used optionally for timing operations

# Define the URL of the IP camera stream
IP_CAMERA_URL = 'http://192.168.1.128:8080/video'

# Define the path to the saved model weights
MODEL_PATH = 'best_model.pth'

# Set the computation device: GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input size expected by the model (ResNet50 typically uses 224x224)
MODEL_INPUT_SIZE = (224, 224)

# Size of the window in which the video will be displayed
DISPLAY_SIZE = (900, 600)

# Define the class names corresponding to model output indices
CLASS_NAMES = ['freshapples', 'freshoranges', 'rottenapples', 'rottenoranges']

# Mapping from class name to user-friendly display label
LABEL_MAP = {
    'freshapples': 'Fresh Apple',
    'freshoranges': 'Fresh Orange',
    'rottenapples': 'Rotten Apple',
    'rottenoranges': 'Rotten Orange'
}

# Only display predictions with confidence above this threshold
CONFIDENCE_THRESHOLD = 0.7

# -----------------------------------
# Load and prepare the model
# -----------------------------------
print("Loading model...")
try:
    # Load a ResNet-50 model without pretrained weights
    model = models.resnet50(weights=None)

    # Replace the final fully connected layer to match the number of fruit classes
    num_ftrs = model.fc.in_features  # Get the number of input features to the last FC layer
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))  # Set the output layer to have 4 classes

    # Load the trained model weights from disk
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Model weights loaded successfully from {MODEL_PATH}")

except FileNotFoundError:
    # Handle case where model file is not found
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    print("Please ensure MODEL_PATH is correct and the file exists.")
    exit()

except Exception as e:
    # Handle other loading errors (e.g., architecture mismatch)
    print(f"FATAL ERROR loading model weights: {e}")
    print("Ensure the model architecture (resnet50) matches the saved weights.")
    exit()

# Move model to the computation device and set it to evaluation mode
model = model.to(DEVICE).eval()
print("Model set to evaluation mode.")

# -----------------------------------
# Define preprocessing steps
# -----------------------------------
print("Setting up preprocessing transforms...")

# Mean and standard deviation used for normalization (ImageNet standard)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Compose preprocessing pipeline:
# Resize → CenterCrop → ToTensor → Normalize
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])
print("Preprocessing transforms defined.")

# -----------------------------------
# Set up OpenCV window
# -----------------------------------
WINDOW_NAME = 'Fruit Classification'
print(f"Creating display window: {WINDOW_NAME}")

# Create resizable OpenCV window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
print(f"Display window created with size {DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]}")

# -----------------------------------
# Open video stream
# -----------------------------------
print(f"Opening video stream from {IP_CAMERA_URL}...")
cap = cv2.VideoCapture(IP_CAMERA_URL)  # Start capturing from the IP camera

# If the video stream cannot be opened, exit with an error
if not cap.isOpened():
    print(f"FATAL ERROR: Unable to open video stream at {IP_CAMERA_URL}")
    print("Please check the IP camera URL and ensure the stream is active.")
    exit()

print("Video stream opened. Press 'q' to exit.")

# Font settings for displaying text on frames
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (0, 255, 0)  # Green color
LINE_TYPE = 2
TEXT_ORIGIN = (10, 30)  # Top-left corner of the frame

# -----------------------------------
# Start real-time inference loop
# -----------------------------------
try:
    while True:
        # Read a frame from the camera stream
        ret, frame = cap.read()

        # If frame could not be read, stop the loop
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Convert BGR image (OpenCV default) to RGB (PIL expects RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image for preprocessing
        pil_img = Image.fromarray(img_rgb)

        # Apply preprocessing and add batch dimension
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

        # Perform inference without computing gradients
        with torch.no_grad():
            outputs = model(input_tensor)  # Forward pass
            probs = F.softmax(outputs, dim=1)[0]  # Convert logits to probabilities
            top_prob, top_idx = probs.max(0)  # Get the class with the highest probability

        # Map the predicted index to class name and confidence
        predicted_class_name = CLASS_NAMES[top_idx.item()]
        confidence = top_prob.item()

        # Default display text
        display_text = "Detecting..."

        # If confidence is above threshold, show prediction
        if confidence >= CONFIDENCE_THRESHOLD:
            display_label = LABEL_MAP.get(predicted_class_name, predicted_class_name)
            display_text = f"{display_label}: {confidence*100:.1f}%"
        else:
            pass  # Don't show uncertain predictions

        # Draw the prediction text on the video frame
        cv2.putText(frame, display_text, TEXT_ORIGIN, FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)

        # Resize the frame for display
        display_frame = cv2.resize(frame, DISPLAY_SIZE)

        # Show the frame in the OpenCV window
        cv2.imshow(WINDOW_NAME, display_frame)

        # Check if the user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' key pressed. Exiting.")
            break

# Catch any unexpected errors during processing
except Exception as e:
    print(f"\nAn error occurred during streaming or inference: {e}")
    print("Possible issues: Invalid IP camera URL, camera stream stopped, model file corrupted, or device issues.")

# -----------------------------------
# Clean up on exit
# -----------------------------------
finally:
    print("Releasing video stream and closing windows...")
    cap.release()  # Release the video stream resource
    cv2.destroyAllWindows()  # Close OpenCV windows
    print("Script finished.")
