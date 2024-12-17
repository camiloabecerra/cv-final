from ultralytics import YOLO
import cv2
from team_assignments import Detector

model = YOLO("finetuned.pt")
path = "videos/video3.mov"
capture = cv2.VideoCapture(path)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter("annotated_videos/video3_annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
ball_pos = []
teams = []
team_positions = {0:[],1:[]}

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    if teams == []:
        classifier = Detector(frame, model)
    else:
        classifier = Detector(frame, model, teams)

    ball_pos += classifier.ball
    classifier.annotate_img()

    print(ball_pos)

    if teams == []:
        teams = classifier.teams
    
    output.write(classifier.img)

print(ball_pos)

capture.release()
output.release()
cv2.destroyAllWindows()
