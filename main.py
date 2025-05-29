import cv2          # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
from ultralytics import YOLO # YOLO library for object detection
import serial       # PySerial library for serial communication
import time         # Time library for time-related functions

# ==============================================================================
# --- User Configuration ---
# ==============================================================================

# --- System / Paths ---
IP_CAMERA_URL = 'http://192.168.105.166:8080/video' # URL of your IP Webcam stream
YOLO_MODEL_NAME = 'yolov8s.pt' # YOLOv8 model name
SERIAL_PORT = 'COM6'   # Updated to match your Arduino IDE
BAUD_RATE = 9600       # Baud rate for serial communication

# --- Detection Settings ---
TARGET_CLASS_NAMES = ['apple', 'orange'] # List of class names to detect
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence score (0.0 to 1.0)

# --- Display Configuration ---
WINDOW_NAME = 'Fruit Detection (YOLOv8)' # Name of the display window
DISPLAY_WIDTH = 900   # Initial window width
DISPLAY_HEIGHT = 600  # Initial window height
INITIAL_WINDOW_X = 100 # Initial window position X
INITIAL_WINDOW_Y = 100 # Initial window position Y

# ==============================================================================
# --- Styling Configuration ---
# ==============================================================================
# Controls appearance of bounding boxes and labels drawn by Ultralytics .plot()
BOX_LINE_WIDTH = 5       # Thickness of bounding box lines
# Note: Font size for results.plot() uses default; direct argument not supported here.
# For advanced control, consider ultralytics.utils.SETTINGS or manual drawing.
SHOW_LABELS = True       # Show class labels above boxes
SHOW_CONFIDENCE = True   # Show confidence scores next to labels

# ==============================================================================
# --- Load YOLOv8 Model ---
# ==============================================================================
print(f"Loading YOLO model: {YOLO_MODEL_NAME}...")
try:
    model = YOLO(YOLO_MODEL_NAME) # Load the YOLOv8 model
    print("Model loaded successfully.")

    # --- Dynamically Find Target Class IDs ---
    model_class_names = model.names # Get the class names from the loaded model
    target_class_ids = {} # Use a dictionary to map class name to ID
    print("Searching for target class IDs in the loaded model...")
    for name in TARGET_CLASS_NAMES: # Iterate through the target class names
        found = False
        for idx, class_name in model_class_names.items(): # Iterate through the model's class names and their IDs
            if name.lower() == class_name.lower(): # Compare target name (case-insensitive) with model name
                target_class_ids[name.lower()] = idx # Store the ID with the lowercase target name as key
                print(f"  - Found '{name}' with ID: {idx}")
                found = True
                break
        if not found: print(f"  - Warning: Target class '{name}' not found.")

    if not target_class_ids: print("FATAL ERROR: Target classes not found."); exit()
    print(f"Detection will filter for Class IDs: {list(target_class_ids.values())}")

except Exception as e: # Catch any exceptions during model loading
    print(f"FATAL ERROR: Could not load YOLO model: {e}")
    print("Install ultralytics: pip install ultralytics")
    exit()

# ==============================================================================
# --- Initialize Serial Communication ---
# ==============================================================================
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE) # Initialize serial communication with Arduino
    print(f"Serial communication established on {SERIAL_PORT} at {BAUD_RATE} baud.")
    time.sleep(2) # Give Arduino time to initialize serial
except serial.SerialException as e: # Catch exceptions during serial port opening
    print(f"FATAL ERROR: Could not open serial port {SERIAL_PORT}: {e}")
    print("Check if the Arduino is connected and the serial port is correct.")
    exit()

# ==============================================================================
# --- Create Display Window ---
# ==============================================================================
print(f"Creating display window: {WINDOW_NAME}")
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Create a resizable named window
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT) # Set the initial window size
cv2.moveWindow(WINDOW_NAME, INITIAL_WINDOW_X, INITIAL_WINDOW_Y) # Set the initial window position
print(f"Set initial window size to {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

# ==============================================================================
# --- Real-time Detection Loop ---
# ==============================================================================
print("Starting detection stream... Press 'q' to quit.")
try:
    results_stream = model.predict( # Start real-time prediction from the IP camera stream
        source=IP_CAMERA_URL, stream=True, classes=list(target_class_ids.values()), conf=CONFIDENCE_THRESHOLD
    )

    last_detected = None # To avoid sending continuous commands

    for results in results_stream: # Iterate through the detection results for each frame
        detected_class = None
        if results.boxes: # If any objects are detected in the current frame
            for box in results.boxes: # Iterate through each detected bounding box
                class_id = int(box.cls[0]) # Get the class ID of the detected object
                confidence = float(box.conf[0]) # Get the confidence score of the detection
                for name, id in target_class_ids.items(): # Iterate through the target class names and their IDs
                    if id == class_id and confidence >= CONFIDENCE_THRESHOLD: # Check if the detected object is a target and has sufficient confidence
                        detected_class = name.lower() # Store the lowercase name of the detected class
                        break
                if detected_class: # If a target class is detected
                    break # Only need to find one detected target

        # Send command to Arduino based on detection
        if detected_class == 'apple' and last_detected != 'apple': # If an apple is detected and wasn't the last detected object
            print("Detected apple. Sending 'a' to Arduino.")
            arduino.write(b'a') # Send 'a' as bytes to the Arduino
            last_detected = 'apple'
        elif detected_class == 'orange' and last_detected != 'orange': # If an orange is detected and wasn't the last detected object
            print("Detected orange. Sending 'o' to Arduino.")
            arduino.write(b'o') # Send 'o' as bytes to the Arduino
            last_detected = 'orange'
        elif not detected_class and last_detected is not None: # If no target is detected and there was a previously detected object
            arduino.write(b'n') # Send 'n' as bytes to the Arduino
            print("No target detected. Sending 'n' to Arduino.")
            last_detected = None

        # Use results.plot() to get frame with annotations drawn by Ultralytics
        annotated_frame = results.plot(
            line_width=BOX_LINE_WIDTH,
            labels=SHOW_LABELS,
            conf=SHOW_CONFIDENCE
        )

        # Display the frame
        cv2.imshow(WINDOW_NAME, annotated_frame) # Show the annotated frame in the display window

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'): # Wait for a key press (1ms) and check if it's 'q'
            print("Exit key ('q') pressed.")
            break

except Exception as e: # Catch any exceptions during the detection loop
    print(f"\nAn error occurred during detection: {e}")
    print("Check camera connection, model compatibility, and libraries.")

finally:
    # ==============================================================================
    # --- Release Resources ---
    # ==============================================================================
    print("Closing windows and serial port...")
    cv2.destroyAllWindows() # Close all OpenCV display windows
    if 'arduino' in locals() and arduino.is_open: # Check if the serial port is open
        arduino.close() # Close the serial port
        print("Serial port closed.")
    print("Script finished.")