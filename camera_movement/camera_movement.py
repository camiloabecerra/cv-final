import cv2
import numpy as np

def measure_distance(p1,p2):
    return np.linalg.norm(p1-p2)

def measure_distance_xy(p1,p2):
    return p1[0]-p2[0], p1[1]-p2[1]


class CameraMovementEstimator():
    def __init__(self, frame):
        first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7
        )

    def get_camera_movement(self, frames):
        camera_movement = [[0,0]*len(frames)]

        gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(gray, frame_gray, features, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT,10,0.03 | cv2.TERM_CRITERIA_EPS))
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new, old) in enumerate(new_features, features):
                old_features_point = old.ravel()
                new_features_point = new.ravel()
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_distance_xy(old_features_point, new_features_point)
            if max_distance < self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            gray = frame_gray.copy()
        return camera_movement
                
