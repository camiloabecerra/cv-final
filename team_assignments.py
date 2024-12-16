import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import numpy as np
import math

class Detector:
    def __init__(self, img, model, team_centroids=None):
        self.img = img
        self.detection = model.predict(self.img, conf=0.1)[0]
        self.teams = team_centroids

        names = self.detection.names
        self.players = list(filter(lambda elt: names[int(elt[5])] == "player", self.detection.boxes.data))
        self.ball = list(filter(lambda elt: names[int(elt[5])] == "ball", self.detection.boxes.data))

    def classify(self):
        return self.detection

    def assign_teams(self):
        if not self.teams:


        avg_colors = []
        for player in self.players:
            val = np.mean(self.img[int(player[1]):int(player[3]),int(player[0]):int(player[2])])
            avg_colors.append(val)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _,labels,_ = cv2.kmeans(np.float32(avg_colors), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        team0 = []
        team1 = []
        for i in range(len(labels)):
            if labels[i] == 0:
                team0.append(list(self.players[i]))
            else:
                team1.append(list(self.players[i]))

        return {0:team0, 1:team1}

    def annotate_img(self):
        for detection in self.detection.boxes.data:
            self.draw_bbox(detection)

    def draw_bbox(self, detection):
        detection = list(detection)
        center_coords = ((int(detection[2]) + int(detection[0])) // 2, int(detection[3]))
        axes_len = (int(math.dist([int(detection[0])],[int(detection[2])])), 30)

        color = (0,0,255)
        if len(self.players) >= 2:
            if detection in self.assign_teams()[0]:
                color = (255,0,0)
            elif detection in self.assign_teams()[1]:
                color = (0,255,0)

        self.img = cv2.ellipse(self.img, center_coords, axes_len, 0, 45, 255, color, 4)
