from ultralytics import YOLO

yolo = YOLO('./yolov8n.pt')

yolo.train(data = 'smoke_Detect.yaml', workers = 0, epochs=30, batch=16)
