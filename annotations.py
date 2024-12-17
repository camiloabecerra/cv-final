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

    df_ball_positions = pd.DataFrame(ball_positions, columns = ['x','y'])

    # interpolate missing values
    df_ball_positions = df_ball_positions.interpolate(method='linear').bfill()

    ball_positions = df_ball_positions.to_nump().tolist()

    return ball_positions

def main():
    model = YOLO("yolo/finetuned.pt")
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

        team0_ppositions = []
        team1_ppositions = []
        for t0p, t1p in zip(classifier.assign_teams()[0], classifier.assign_teams()[1]):
            t0x = int((t0p[2]+t0p[0])//2)
            t0y = int((t0p[3]+t0p[1])//2)
            team0_ppositions.append([t0x,t0y])

            t1x = int((t1p[2]+t1p[0])//2)
            t1y = int((t1p[3]+t1p[1])//2)
            team1_ppositions.append([t1x,t1y])

        ball_pos += classifier.ball
        team_positions[0].append(team0_ppositions)
        team_positions[1].append(team1_ppositions)

        classifier.annotate_img()
        if teams == []:
            teams = classifier.teams

        output.write(classifier.img)

    ball_pos = interpolate_ball_positions(ball_pos)

    capture.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()