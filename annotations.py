from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np

from team_assignments import Detector
from ball_assignments import PlayerBallAssigner


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

    ball_positions = df_ball_positions.to_numpy().tolist()

    return ball_positions

def calculate_speed(team_positions, fps):
    team_speeds = {0: [], 1: []}
    for team_id, positions in team_positions.items():
        speeds = []
        for player_positions in zip(*positions):
            total_distance = 0
            for i in range(1, len(player_positions)):
                x1, y1 = player_positions[i-1]
                x2, y2 = player_positions[i]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_distance += distance
            speed = (total_distance / len(player_positions)) * fps if len(player_positions) > 1 else 0
            speeds.append(speed)
        team_speeds[team_id] = speeds
    return team_speeds


def update_team_positions(team_positions, team_assignments):
    for team_id, players in team_assignments.items():
        frame_positions = []
        for player in players:
            x = int((player[2]+player[0])//2)
            y = int((player[3]+player[1])//2)
            frame_positions.append([x,y])
        team_positions[team_id].append(frame_positions)
    return team_positions

def annotate_speeds(frame, team_assignments, team_speeds):
    for team_id, players in team_assignments.items():
        for i, player in enumerate(players):
            if i >= len(team_speeds[team_id]):
                print(f"skipping player {i} in team {team_id} due to incompatible speed data")
                continue
            x = int((player[2]+player[0]) // 2)
            y = int((player[3]+player[1]) // 2)
            speed = team_speeds[team_id][i]
            cv2.putText(frame, f"{speed:.2f} px/s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
    return frame

def process_frame(frame, model, teams, team_positions, ball_pos, fps):
    if teams == []:
            classifier = Detector(frame, model)
    else:
        classifier = Detector(frame, model, teams)

    ball_pos += classifier.ball
    if not teams:
        teams = classifier.teams
    
    team_assignments = classifier.assign_teams()

    team_positions = update_team_positions(team_positions, team_assignments)
    team_speeds = calculate_speed(team_positions, fps)
    frame = annotate_speeds(classifier.img, team_assignments, team_speeds)

    classifier.annotate_img()
    return teams, team_positions, ball_pos, classifier.img


def main():
    model = YOLO("yolo/finetuned.pt")
    path = "videos/video1.mov"
    capture = cv2.VideoCapture(path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    output = cv2.VideoWriter("annotated_videos/video1_annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    ball_pos = []
    teams = []
    team_positions = {0:[],1:[]}

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        teams, team_positions, ball_pos, annotated_frame = process_frame(frame, model, teams, team_positions, ball_pos, fps)
        output.write(annotated_frame)

    ball_pos = interpolate_ball_positions(ball_pos)

    capture.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()