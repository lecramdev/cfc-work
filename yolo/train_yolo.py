from ultralytics import YOLO

model = YOLO("yolov6n.yaml", "detect")
model.train(data='VOC.yaml', epochs=1000, imgsz=640, plots=True, device="cpu")
