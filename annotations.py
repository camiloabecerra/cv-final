from ultralytics import YOLO
import cv2
import pandas as pd
from team_assignments import Detector

def interpolate_ball_positions(ball_positions):
    """
    Interpolates and fills missing ball position over time. 

    Parameters:
    -----------
    ball_position: list 
        List of dictionaries containing ball positions. 

    Returns: 
    -----------
    list
        List of interpolated ball prediction dictionaries
    """

    ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2','y2'])

    # interpolate missing values
    df_ball_positions = df_ball_positions.interpolate().bfill()

    ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().toList()]

    return ball_positions

def main():
    model = YOLO("finetuned.pt")
    path = "videos/video3.mov"
    capture = cv2.VideoCapture(path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    output = cv2.VideoWriter("annotated_videos/video3_annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    
    ball_pos = []
    teams = []
    # team_positions = {0:[],1:[]}

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

        if teams == []:
            teams = classifier.teams
        
        output.write(classifier.img)

    ball_pos = interpolate_ball_positions(ball_pos)
    print(ball_pos)

    capture.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()