import numpy as np
import random
from scipy.spatial import distance_matrix

from line_intersections import intersectLines

if __name__ == "__main__":
    Lx, Ly = 20.0, 10.0
    ind = {"x": 0, "y": 1}

    goal_line = np.array([[Lx, 0.1*Ly], [Lx, 0.9*Ly]])
    defenders = np.array([[0.9*Lx, 0.5*Ly]])
    ball = np.array([0.1*Lx, 0.5*Ly])

    twopi = 2*np.pi
    goal_line_rel_ball = goal_line - ball
    theta_goal_line_rel_ball = []
    for i in range(2):
        theta = np.arctan2(goal_line_rel_ball[i,ind["y"]], goal_line_rel_ball[i,ind["x"]])
        theta_goal_line_rel_ball.append(theta)

    #theta_sample = np.random.uniform(theta_goal_line_rel_ball[0]-0.5, theta_goal_line_rel_ball[1]+0.5)
    theta_sample = np.random.uniform(theta_goal_line_rel_ball[0], theta_goal_line_rel_ball[1])
    theta_vec_sample = np.array([np.cos(theta_sample), np.sin(theta_sample)])

    v_ball_size = 1.0
    v_ball = v_ball_size * theta_vec_sample

    n_iter = 300
    t = 0.0
    dt = 0.1
    scored = 0
    for iter in range(n_iter):
        t += dt
        ball_new = ball + v_ball * dt

        if ball_new[ind["x"]] < 0 or ball_new[ind["x"]] > Lx or \
            ball_new[ind["y"]] < 0 or ball_new[ind["y"]] > Ly:

            intersection_result = intersectLines(ball_new, ball, goal_line[0], goal_line[1])
            valid, r, s = intersection_result[2], intersection_result[3], intersection_result[4]
            if valid==1 and r>=0.0 and r<=1.0 and s>=0.0 and s<=1.0:
                scored = 1
                print("scored!")
            else:
                scored = 0
                print("missed!")

            break

        ball = ball_new
        #print(iter, t, ":", ball)


    #theta_goal_line = goal_line[0]
    #theta_particles_D = np.arctan2(goal_line, xy_D[:, 0]) % twopi
    #ball_vision =
    #goal = np.array([[2.5, 4.8]])
    #shot_probs = get_shot_probs(allies, opponents, goal)
    #shot_results = take_shots(shot_probs)
    #print(shot_results)



