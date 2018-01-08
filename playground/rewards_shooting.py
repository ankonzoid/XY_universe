


import numpy as np
import random
from scipy.spatial import distance_matrix

def distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def take_shots(shot_probs):
    n = len(shot_probs)
    rand_list = np.array([random.random() for i in range(n)])
    shot_results = np.array(rand_list < shot_probs, dtype=int)
    return shot_results

def get_shot_probs(allies, opponents, goals):
    D_goal = 5.0
    D_opponent = 5.0
    d_goals, d_opponents = distance_matrix(allies, goals), distance_matrix(allies, opponents)
    d_closest_goal, d_closest_opponent = np.amin(d_goals, axis=1), np.amin(d_opponents, axis=1)
    shot_probs = 1.0 - d_closest_goal/(2*D_goal) - d_closest_opponent/(2*D_opponent)
    return shot_probs

if __name__ == "__main__":
    Lx, Ly = 5.0, 5.0
    allies = np.array([[0.2, 0.2],
                       [4.8, 0.2]])
    opponents = np.array([[1.4, 1.5],
                          [2.6, 3.5]])
    goal = np.array([[2.5, 4.8]])
    shot_probs = get_shot_probs(allies, opponents, goal)
    shot_results = take_shots(shot_probs)
    print(shot_results)