from re import A
import cv2
from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, img, model):
        self.img = cv2.imread(img)
        self.detection = model.predict(img, conf=0.1)[0]

        names = self.detection.names
        self.players = list(filter(lambda elt: names[int(elt[5])] == "player", self.detection.boxes.data))

    def classify(self):
        return self.detection

    def assign_teams(self):
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
                team0.append(self.players[i])
            else:
                team1.append(self.players[i])

        return {0:team0, 1:team1}

    # def annotate_img(self):
    #     for detection in self.detection.boxes.data:
    #         self.draw_bbox(detection)

    # def draw_bbox(self, detection):
    #     cv2.ellipse(
    #         self.img, int(detection[4])
    #     )


# classifier = Detector("images/football.png", YOLO("finetuned.pt"))
# classifier.assign_teams()
