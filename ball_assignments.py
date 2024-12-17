import sys 
sys.path.append('../')

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 60
    
    def assign_ball_to_player(self, players, ball_positions):
        ball_positions = {}

        for ball_id, ball_pos in enumerate(ball_positions):
            min_distance = float("inf")
            assigned_player = -1

            # find player closest to the ball 
            # for player_id, player_pos in enumerate(players):
