import numpy as np
from scipy.optimize import linear_sum_assignment
pitch_length = 100
pitch_width = 80
#4-4-2
keeper_units = [pitch_length/20, pitch_width/2]
defense_units = [pitch_length/5, pitch_width/5]
midfield_units = [pitch_length/2, pitch_width/5]
attack_units = [pitch_length/1.25, pitch_width/3]

#4-3-3
keeper_units_2 = [pitch_length/20, pitch_width/2]
defense_units_2 = [pitch_length/5, pitch_width/5]
midfield_units_2 = [pitch_length/2, pitch_width/4]
attack_units_2 = [pitch_length/1.25, pitch_width/4]

#3-4-3
keeper_units_3 = [pitch_length/20, pitch_width/2]
defense_units_3 = [pitch_length/5, pitch_width/4]
midfield_units_3 = [pitch_length/2, pitch_width/5]
attack_units_3 = [pitch_length/1.25, pitch_width/4]

player_coords_path = "data/"
player_coords = []

formations = {
    "4-4-2": [
        # Keeper
        (keeper_units[0], keeper_units[1]),
        # Defense
        (defense_units[0],defense_units[1]),(defense_units[0],defense_units[1]*2), (defense_units[0], defense_units[1]*3), (defense_units[0], defense_units[1]*4),
        # Midfield
        (midfield_units[0], midfield_units[1]), (midfield_units[0], midfield_units[1]*2), (midfield_units[0], midfield_units[1]*3), (midfield_units[0], midfield_units[1]*4),
        # Attack
        (attack_units[0], attack_units[1]), (attack_units[0], attack_units[1]*2),
    ],
    "4-4-3": [
        # Keeper
        (keeper_units_2[0], keeper_units_2[1]),
        # Defense
        (defense_units_2[0],defense_units_2[1]),(defense_units_2[0],defense_units_2[1]*2), (defense_units_2[0], defense_units_2[1]*3), (defense_units_2[0], defense_units_2[1]*4),
        # Midfield
        (midfield_units_2[0], midfield_units_2[1]), (midfield_units_2[0], midfield_units_2[1]*2), (midfield_units_2[0], midfield_units_2[1]*3),
        # Attack
        (attack_units_2[0], attack_units_2[1]), (attack_units_2[0], attack_units_2[1]*2), (attack_units_2[0], attack_units_2[1]*3),
    ],
    "3-4-3": [
        # Keeper
        (keeper_units_3[0], keeper_units_3[1]),
        # Defense
        (defense_units_3[0], defense_units_3[1]), (defense_units_3[0], defense_units_3[1]*2), (defense_units_3[0], defense_units_3[1]*3)
        # Midfield
        (midfield_units_3[0], midfield_units_3[1]), (midfield_units_3[0], midfield_units_3[1]*2), (midfield_units_3[0], midfield_units_3[1]*3), (midfield_units_3[0], midfield_units_3[1]*4),
        # Attack
        (attack_units_3[0], attack_units_3[1]), (attack_units_3[0], attack_units_3[1]*2), (attack_units_3[0], attack_units_3[1]*3),
    ],
}


def normalize_positions(positions, pitch_length, pitch_width):
    # take in identified player positions and scale to identified pitch_length and pitch_width, to make compatible with formation coords
    return [(x/pitch_length, y/pitch_width) for x,y in positions]

def compute_distance(player_coords, formation_coords):
    len_formation_coords = len(formation_coords)
    len_player_coords = len(player_coords)
    distance_matrix = np.zeros((len_player_coords, len_formation_coords))
    for i, p in enumerate(player_coords):
        for j, f in enumerate(formation_coords):
            distance_matrix[i,j] = np.linalg.norm(np.array(p)-np.array(f))
    return distance_matrix

def find_most_matching(distance_matrix):
    row_index, col_index = linear_sum_assignment(distance_matrix)
    return row_index, col_index

def compute_similarity_score(distance_matrix, row_index, col_index):
    return sum(distance_matrix[row_index, col_index])

def identify_formation(player_coords, formation_coords, pitch_length, pitch_width):
    normalized_player_coords = normalize_positions(player_coords, pitch_length, pitch_width)
    best_formation = None
    best_score = float('inf')
    for formation, formation_coords in formations.items():
        # normalize formation coords just in case, should remain unchanged
        normalized_formation_coords = normalize_positions(formation_coords, pitch_length, pitch_width)
        # calculate distance matrix
        distance_matrix = compute_distance(normalized_player_coords, normalized_formation_coords)
        row_index, col_index = find_most_matching(distance_matrix)
        score = compute_similarity_score(distance_matrix, row_index, col_index)
        if score < best_score:
            best_score = score
            best_formation = formation
    return best_formation, best_score

def main():
    best_formation, best_score = identify_formation(player_coords, formations, pitch_length, pitch_width)
    print(best_formation, best_score)