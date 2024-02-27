from ultralytics import YOLO
yolo = YOLO("./yolov8n.pt", task="detect")
result = yolo(source="./ultralytics/assets/000105.jpg", save=True)