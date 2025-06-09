import cv2
from ultralytics import YOLO

IP_CAMERA_URL = 'http://192.168.1.128:8080/video'
YOLO_MODEL_NAME = '../yolov8_models/yolov8l.pt'

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
                print(f"   - Found '{name}' with ID: {idx}")
                found = True
                break
        if not found: print(f"   - Warning: Target class '{name}' not found.")

    if not target_class_ids: print("FATAL ERROR: Target classes not found."); exit()
    print(f"Detection will filter for Class IDs: {list(target_class_ids.values())}")

except Exception as e:
    print(f"FATAL ERROR: Could not load YOLO model: {e}")
    print("Install ultralytics: pip install ultralytics")
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

    for results in results_stream:
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
    print("Closing windows...")
    cv2.destroyAllWindows()
    print("Script finished.")