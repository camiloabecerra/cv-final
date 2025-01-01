import sys 
import math
sys.path.append('../')

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

class PlayerBallAssigner():
    def __init__(self):
        """
        Initialize the PlayerBallAssigner with a maximum allowable distance
        for assigning the ball to a player.
        """
        self.max_player_ball_distance = 60
        self.team_possession = {0:0, 1:0}
    
    def measure_distance(self, p1, p2):
        """
        Measure the Euclidean distance between two points.

        Parameters:
        -----------
        p1, p2 : tuple
            Coordinates of the two points as (x, y).

        Returns:
        --------
        float
            Euclidean distance between the points.
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def assign_ball_to_team(self, players, ball_position):
        """
        Assign the ball to the closest player and track team possession.

        Parameters:
        -----------
        players : dict
            Dictionary containing player positions by team.
        ball_positions : list
            List of ball positions as [x, y] coordinates.

        Returns:
        --------
        dict
            Dictionary mapping each ball to the assigned team.
        """
        min_distance = float("inf")
        assigned_team = -1

        # find player closest to the ball 
        for team_id, player_arr in players.items():
            for player_pos in player_arr[-1]:
                # measure distance between player and the ball

                distance = self.measure_distance(player_pos, ball_position[-1])
                if distance < self.max_player_ball_distance and distance < min_distance:
                    min_distance = distance 
                    assigned_team = team_id

        if assigned_team != -1:
            self.team_possession[assigned_team] += 1

        return self.team_possession
        
    
    def team_ball_possession(self, total_frames):
        """
        Calculate the percentage of ball posession for each team. 

        Returns:
        ----------
        dict
            Dictionary containing the possession percentage for each team. 
        """
        percentages = {
            team_id: (count / total_frames) * 100 for team_id, count in self.team_possession.items()
        }
        return percentages