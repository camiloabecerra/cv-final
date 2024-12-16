from ultralytics import YOLO, Annotator
import cv2
from team_assignments import Detector

model = YOLO("finetuned.pt")
path = "videos/video2.mov"
capture = cv2.VideoCapture(path)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter("annotated_videos/video2_annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # classifier = Detector("images/football.png", model)
    # teams = classifier.assign_teams()

    annotated_frame = model.predict(frame, conf=0.1)[0].plot()

    output.write(annotated_frame)

capture.release()
output.release()
cv2.destroyAllWindows()
