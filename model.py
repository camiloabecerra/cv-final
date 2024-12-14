import os
from ultralytics import YOLO
from hyperparameters import EPOCHS, IMGSZ

data_path = "data/data.yaml"

model = YOLO("yolo11n.pt")
results = model.train(data=data_path, epochs=EPOCHS, imgsz=IMGSZ)