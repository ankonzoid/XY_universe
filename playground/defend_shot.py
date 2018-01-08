import numpy as np
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from line_intersections import intersectLines

if __name__ == "__main__":
    Lx, Ly = 20.0, 10.0
    r_ball = 0.2
    r_defender = 0.5
    dy = 0.04
    ind = {"x": 0, "y": 1}

    goal_line = np.array([[Lx, 0.1*Ly], [Lx, 0.9*Ly]])
    defender_init = np.array([0.9*Lx, 0.5*Ly])
    ball_init = np.array([0.1*Lx, 0.5*Ly])

    # Generate episodes
    n_episodes = 1000
    n_iter = 300
    t = 0.0
    dt = 0.1
    scored = 0
    v_ball_size = 2.0
    for episode in range(n_episodes):

        # Reset position of ball and defenders
        ball, defender = ball_init.copy(), defender_init.copy()

        # Animation storage
        ball_animation, defender_animation = [], []
        ball_animation.append(ball)
        defender_animation.append(defender)

        # Set up initial conditions
        # Find the theta angle spread of the goal posts
        twopi = 2 * np.pi
        goal_line_rel_ball = goal_line - ball
        theta_goal_line_rel_ball = []
        for i in range(2):
            theta = np.arctan2(goal_line_rel_ball[i, ind["y"]], goal_line_rel_ball[i, ind["x"]])
            theta_goal_line_rel_ball.append(theta)
        if 0:  # chance to miss
            theta_sample = np.random.uniform(theta_goal_line_rel_ball[0] - 0.5, theta_goal_line_rel_ball[1] + 0.5)
        else:  # almost 100% of hitting
            theta_sample = np.random.uniform(theta_goal_line_rel_ball[0], theta_goal_line_rel_ball[1])
        theta_vec_sample = np.array([np.cos(theta_sample), np.sin(theta_sample)])
        v_ball = v_ball_size * theta_vec_sample

        # Simulate physics
        status = ""

        for iter in range(n_iter):

            # ============================
            # Make observation
            # ============================


            # ============================
            # Choose an action
            # ============================
            action = 0

            # ============================
            # Move agent
            # ============================
            if action == 0:
                defender += np.array([0,dy])  # up
            elif action == 1:
                defender += np.array([0,-dy])  # down
            else:
                exit("err: invalid action!")

            # ============================
            # Move ball
            # ============================
            t += dt
            ball_new = ball + v_ball * dt
            d_ball_to_defender = np.linalg.norm(ball_new-defender)  # scalar

            # ============================
            # Check for termination
            # ============================
            if ball_new[ind["x"]] < 0 or ball_new[ind["x"]] > Lx or \
                ball_new[ind["y"]] < 0 or ball_new[ind["y"]] > Ly:

                intersection_result = intersectLines(ball_new, ball, goal_line[0], goal_line[1])
                valid, r, s = intersection_result[2], intersection_result[3], intersection_result[4]
                if valid==1 and r>=0.0 and r<=1.0 and s>=0.0 and s<=1.0:
                    scored += 1
                    status = "scored"
                else:
                    status = "missed"
                break

            elif d_ball_to_defender < (r_defender + r_ball):
                status = "blocked"
                break

            ball = ball_new

            # Store
            defender_animation.append(defender.copy())
            ball_animation.append(ball)

        # =============================
        # Make animation
        # =============================
        fig = plt.figure()
        fig.clf()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.1, Lx+0.5), ylim=(-0.1, Ly+0.1))

        defender_particles = plt.Circle((defender_animation[0][0], defender_animation[0][1]), radius=r_defender, fc='b')
        ball_particles = plt.Circle((ball_animation[0][0], ball_animation[0][1]), radius=r_ball, fc='r')
        rect = plt.Rectangle((0.0, 0.0), Lx, Ly, ec='none', lw=2, fc='none')
        goal = plt.Rectangle((goal_line[0][0], goal_line[0][1]), 0.3, goal_line[1][1] - goal_line[0][1], ec='none', lw=2, fc='k')
        rect.set_edgecolor('k')
        goal.set_edgecolor('k')

        ax.add_patch(defender_particles)
        ax.add_patch(ball_particles)
        ax.add_patch(rect)
        ax.add_patch(goal)

        def init():
            defender_particles.center = (defender_animation[0][0], defender_animation[0][1])
            ball_particles.center = (ball_animation[0][0], ball_animation[0][1])
            ax.add_patch(defender_particles)
            ax.add_patch(ball_particles)
            ax.add_patch(rect)
            ax.add_patch(goal)
            return defender_particles, ball_particles, rect, goal

        def animate(i):
            defender_particles.center = (defender_animation[i][0], defender_animation[i][1])
            ball_particles.center = (ball_animation[i][0], ball_animation[i][1])
            return defender_particles, ball_particles, rect, goal

        ani = animation.FuncAnimation(fig, animate, len(defender_animation), init_func=init, interval=10)
        plt.axis('scaled')
        plt.show()

        # Print status
        print("{}: {} (acc={}/{})".format(episode+1, status, scored, episode+1))





