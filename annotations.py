from tkinter.constants import W
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("finetuned.pt")
path = "videos/video2.mov"
capture = cv2.VideoCapture(path)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter("video2_annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    annotated_frame = model(frame, conf=0.6)[0].plot()
    output.write(annotated_frame)

capture.release()
output.release()
cv2.destroyAllWindows()
