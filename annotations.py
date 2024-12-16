from ultralytics import YOLO
import cv2
from team_assignments import Detector

model = YOLO("finetuned.pt")
path = "videos/video2.mov"
capture = cv2.VideoCapture(path)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter("annotated_videos/video2_annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
ball_pos = []

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    classifier = Detector(frame, model)
    ball_pos += classifier.ball
    classifier.annotate_img()

    output.write(classifier.img)

capture.release()
output.release()
cv2.destroyAllWindows()
