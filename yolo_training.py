import os
from ultralytics import YOLO
from hyperparameters import EPOCHS, IMGSZ

<<<<<<< HEAD:yolo_training.py
data_path = "datasets/data.yaml"
=======
data_path = "datasets/data/data.yaml"
>>>>>>> refs/remotes/origin/main:model.py

model = YOLO("yolo11n.pt")
model.train(data=data_path, epochs=EPOCHS, imgsz=IMGSZ)
model.export()
