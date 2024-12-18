import sys 
import math
sys.path.append('../')

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 60
    
    def measure_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def assign_ball_to_player(self, players, ball_positions):
        ball_assignments = {}

        for ball_id, ball_pos in enumerate(ball_positions):
            min_distance = float("inf")
            assigned_team = -1

            # find player closest to the ball 
            for team_id, player_arr in players.items():
                for player_pos in player_arr[-1]:
                    # measure distance between player and the ball
                    distance = self.measure_distance(player_pos, ball_pos)
                    if distance < self.max_player_ball_distance and distance < min_distance:
                        min_distance = distance 
                        assigned_team = team_id 
        
        ball_assignments[ball_id] = assigned_team
        return ball_assignments