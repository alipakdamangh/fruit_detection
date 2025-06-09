import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time

IP_CAMERA_URL = 'http://192.168.105.166:8080/video'
YOLO_MODEL_NAME = 'yolov8s.pt'
SERIAL_PORT = 'COM6'
BAUD_RATE = 9600

TARGET_CLASS_NAMES = ['apple', 'orange']
CONFIDENCE_THRESHOLD = 0.4

WINDOW_NAME = 'Fruit Detection (YOLOv8)'
DISPLAY_WIDTH = 900
DISPLAY_HEIGHT = 600
INITIAL_WINDOW_X = 100
INITIAL_WINDOW_Y = 100

BOX_LINE_WIDTH = 5
SHOW_LABELS = True
SHOW_CONFIDENCE = True

print(f"Loading YOLO model: {YOLO_MODEL_NAME}...")
try:
    model = YOLO(YOLO_MODEL_NAME)
    print("Model loaded successfully.")
    model_class_names = model.names
    target_class_ids = {}
    print("Searching for target class IDs in the loaded model...")
    for name in TARGET_CLASS_NAMES:
        found = False
        for idx, class_name in model_class_names.items():
            if name.lower() == class_name.lower():
                target_class_ids[name.lower()] = idx
                print(f"  - Found '{name}' with ID: {idx}")
                found = True
                break
        if not found: print(f"  - Warning: Target class '{name}' not found.")
    if not target_class_ids: print("FATAL ERROR: Target classes not found."); exit()
    print(f"Detection will filter for Class IDs: {list(target_class_ids.values())}")
except Exception as e:
    print(f"FATAL ERROR: Could not load YOLO model: {e}")
    print("Install ultralytics: pip install ultralytics")
    exit()

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print(f"Serial communication established on {SERIAL_PORT} at {BAUD_RATE} baud.")
    time.sleep(2)
except serial.SerialException as e:
    print(f"FATAL ERROR: Could not open serial port {SERIAL_PORT}: {e}")
    print("Check if the Arduino is connected and the serial port is correct.")
    exit()

print(f"Creating display window: {WINDOW_NAME}")
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
cv2.moveWindow(WINDOW_NAME, INITIAL_WINDOW_X, INITIAL_WINDOW_Y)
print(f"Set initial window size to {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

print("Starting detection stream... Press 'q' to quit.")
try:
    results_stream = model.predict(
        source=IP_CAMERA_URL, stream=True, classes=list(target_class_ids.values()), conf=CONFIDENCE_THRESHOLD
    )
    last_detected = None
    for results in results_stream:
        detected_class = None
        if results.boxes:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                for name, id in target_class_ids.items():
                    if id == class_id and confidence >= CONFIDENCE_THRESHOLD:
                        detected_class = name.lower()
                        break
                if detected_class:
                    break
        if detected_class == 'apple' and last_detected != 'apple':
            print("Detected apple. Sending 'a' to Arduino.")
            arduino.write(b'a')
            last_detected = 'apple'
        elif detected_class == 'orange' and last_detected != 'orange':
            print("Detected orange. Sending 'o' to Arduino.")
            arduino.write(b'o')
            last_detected = 'orange'
        elif not detected_class and last_detected is not None:
            arduino.write(b'n')
            print("No target detected. Sending 'n' to Arduino.")
            last_detected = None
        annotated_frame = results.plot(
            line_width=BOX_LINE_WIDTH,
            labels=SHOW_LABELS,
            conf=SHOW_CONFIDENCE
        )
        cv2.imshow(WINDOW_NAME, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key ('q') pressed.")
            break
except Exception as e:
    print(f"\nAn error occurred during detection: {e}")
    print("Check camera connection, model compatibility, and libraries.")
finally:
    print("Closing windows and serial port...")
    cv2.destroyAllWindows()
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Serial port closed.")
    print("Script finished.")
