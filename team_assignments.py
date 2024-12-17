import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import numpy as np
import math

class Detector:
    def __init__(self, img, model, team_centroids=[]):
        """
        Initialize the Detector object with an image, a model, and optionally, pre-defined team centroids.

        Parameters:
        -----------
        img : np.ndarray
            The input image on which object detection and team assignment will be performed.
        model : YOLO
            A YOLO detection model used to predict bounding boxes for players and the ball.
        team_centroids : list (optional)
            A list of pre-computed team color centroids to classify players into teams. If empty, 
            team assignments will be inferred using K-Means clustering on the detected players.
        """
        self.img = img
        self.teams = team_centroids

        # run object detection using YOLO
        self.detection = model.predict(self.img, conf=0.1)[0]

        # store players and ball positions 
        self.players = list(filter(lambda elt: self.detection.names[int(elt[5])] == "player", self.detection.boxes.data))
        self.ball = list(filter(lambda elt: self.detection.names[int(elt[5])] == "ball", self.detection.boxes.data))

        # if more than two players are detected, assign the players into teams 
        if len(self.players) >= 2:
            self.team_players = self.assign_teams()
        else:
            self.team_players = {}

    def assign_teams(self):
        """
        Assign detected players to two teams based on the average color within their bounding boxes.

        If no team color centroids are provided, K-Means clustering is used to find two color-based clusters.
        If centroids are provided, players are assigned to the closest centroid.

        Returns:
        --------
        dict
            A dictionary with two keys (0 and 1), each containing a list of player bounding boxes 
            assigned to that team.
        """
        avg_colors = list(map(lambda p: np.mean(self.img[int(p[1]):int(p[3]),int(p[0]):int(p[2])], axis=(0,1,2)), self.players))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        if len(self.teams) == 0:
            _,labels,self.teams = cv2.kmeans(np.float32(avg_colors), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            self.teams = [self.teams[0][0],self.teams[1][0]]
        else:
            labels = list(map(lambda x: 0 if math.dist([self.teams[0]],[x]) <= math.dist([self.teams[1]],[x]) else 1, avg_colors))

        team0 = []
        team1 = []
        for i in range(len(labels)):
            if labels[i] == 0:
                team0.append(list(self.players[i]))
            else:
                team1.append(list(self.players[i]))

        return {0:team0, 1:team1}

    def annotate_img(self):
        """
        Draw ellipses on the detected objects (players/ball) to visually represent their location and team affiliation.
        This method modifies the input image by adding these annotations.
        """
        for detection in self.detection.boxes.data:
            self.draw_bbox(detection)

    def draw_bbox(self, detection):
        """
        Draw an ellipse on the image to highlight a detected player or ball. The ellipse color indicates the team:
        
        - Blue: Team 0
        - Green: Team 1
        - Red: Not assigned or object other than a player (e.g., the ball)

        Parameters:
        -----------
        detection : list
            A single detection bounding box represented as [x1, y1, x2, y2, confidence, class_id].
        """
        detection = list(detection)

        # compute ellipse center and axes length based on bounding box 
        center_coords = ((int(detection[2]) + int(detection[0])) // 2, int(detection[3]))
        axes_len = (int(math.dist([int(detection[0])],[int(detection[2])])), 30)

        color = (0,0,255)
        if detection in self.team_players[0]:
            color = (255,0,0)
        elif detection in self.team_players[1]:
            color = (0,255,0)

        self.img = cv2.ellipse(self.img, center_coords, axes_len, 0, 35, 235, color, 4)