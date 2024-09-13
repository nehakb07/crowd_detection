from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (e.g., YOLOv8n for nano, YOLOv8s for small)
model = YOLO("yolov8n.pt")  # Start with a pre-trained model

# Train the model using your dataset
model.train(
    data='C:/Users/Neha KB/Desktop/humanhead/data.yaml',  # The YAML file you created
    epochs=30,                 # Number of epochs
    batch=16,                  # Batch size (adjust based on your system's capacity)
    imgsz=640,                 # Image size for training (YOLO typically uses 640x640)
    workers=2                  # Number of workers for data loading
)
