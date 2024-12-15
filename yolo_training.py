import os
from ultralytics import YOLO
from hyperparameters import EPOCHS, IMGSZ

data_path = "datasets/data.yaml"

model = YOLO("yolo11n.pt")
model.train(data=data_path, epochs=EPOCHS, imgsz=IMGSZ)
model.export()
