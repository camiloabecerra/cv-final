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
    """
    Calculate the speed of players for each team across frames.

    Parameters:
    -----------
    team_positions : dict
        Dictionary containing player positions for each team across frames.
    fps : int
        Frames per second of the video.

    Returns:
    --------
    dict
        A dictionary containing the calculated speed (px/s) for each player in each team.
        Example: {0: [speed1, speed2, ...], 1: [speed1, speed2, ...]}
    """
    team_speeds = {0: [], 1: []}
    for team_id, positions in team_positions.items():
        if len(positions) < 2:
            continue
        for frame in range(1, len(positions)):
            curr_positions = positions[frame]
            prev_positions = positions[frame-1]
            for player, (curr_pos, prev_pos) in enumerate(zip(curr_positions, prev_positions)):
                if curr_pos is None or prev_pos is None:
                    continue
                distance = measure_distance(prev_pos, curr_pos)
                time_elapsed = 1/fps
                speed_mps = distance / time_elapsed
                speed_kph = speed_mps * 3.6

                if player not in team_speeds[team_id]:
                    team_speeds[team_id][player] = []
                team_speeds[team_id][player].append(speed_kph)

                if len(team_speeds[team_id][player]) > frame_window:
                    team_speeds[team_id][player].pop(0)

                if frame % frame_window == 0:
                    avg_speed = sum(team_speeds[team_id][player])/len(team_speeds[team_id][player])
                else:
                    avg_speed = curr_positions[player][2] if len(curr_positions[player]) > 2 else None

                if len(curr_positions[player]) < 3:
                    curr_positions[player].append(None)
                curr_positions[player][2] = avg_speed
    return team_positions




    #     for player_positions in zip(*positions):
    #         total_distance = 0
    #         for i in range(1, len(player_positions)):
    #             x1, y1 = player_positions[i-1]
    #             x2, y2 = player_positions[i]
    #             distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    #             total_distance += distance
    #         speed = (total_distance / len(player_positions)) * fps if len(player_positions) > 1 else 0
    #         speeds.append(speed)
    #     team_speeds[team_id] = speeds
    # return team_speeds

def update_team_positions(team_positions, team_assignments):
    """
    Update player positions for each team in the current frame.

    Parameters:
    -----------
    team_positions : dict
        Dictionary to store cumulative player positions for each team.
    team_assignments : dict
        Dictionary containing the players detected in the current frame.

    Returns:
    --------
    dict
        Updated team positions with new frame data added.
    """
    for team_id, players in team_assignments.items():
        frame_positions = []
        for player in players:
            x = int((player[2]+player[0])//2)
            y = int((player[3]+player[1])//2)
            frame_positions.append([x,y])
        team_positions[team_id].append(frame_positions)
    return team_positions

def annotate_speeds(frame, team_assignments, team_speeds):
    """
    Annotate the frame with the speeds of detected players.

    Parameters:
    -----------
    frame : np.ndarray
        Current video frame to annotate.
    team_assignments : dict
        Dictionary containing player bounding boxes for each team.
    team_speeds : dict
        Dictionary containing calculated player speeds for each team.

    Returns:
    --------
    np.ndarray
        Annotated frame with player speeds displayed.
    """
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
    """
    Process a single video frame by detecting players, assigning teams, calculating speeds,
    and annotating ball possession and speeds.

    Parameters:
    -----------
    frame : np.ndarray
        Current video frame to process.
    model : YOLO
        YOLO object detection model.
    teams : list
        List of team centroids.
    team_positions : dict
        Dictionary to store cumulative player positions for each team.
    ball_pos : list
        List to store cumulative ball positions.
    fps : int
        Frames per second of the video.

    Returns:
    --------
    tuple
        Updated teams, team_positions, ball_pos, and annotated frame.
    """

    if teams == []:
            classifier = Detector(frame, model)
    else:
        classifier = Detector(frame, model, teams)

    # update ball positions
    ball_pos += classifier.ball

    # assign team if not already assigned
    if not teams:
        teams = classifier.teams
    
    team_assignments = classifier.assign_teams()

    # update team positions to calculate speeds 
    team_positions = update_team_positions(team_positions, team_assignments)
    team_speeds = calculate_speed(team_positions, fps)

    # assign ball possession 
    if classifier.ball:
        assigner = PlayerBallAssigner()
        assignments = assigner.assign_ball_to_player(team_positions, classifier.ball)

        for ball_id, team_id in assignments.items():
            if team_id != -1:
                # Annotate possession on the frame
                x = int(classifier.ball[ball_id][0])
                y = int(classifier.ball[ball_id][1])
                cv2.putText(
                    classifier.img,
                    f"Ball {ball_id}: Team {team_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),  # Green text
                    2,
                )

    # annotate speeds on each frame 
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