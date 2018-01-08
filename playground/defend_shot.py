import numpy as np
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random

from line_intersections import intersectLines

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

if __name__ == "__main__":
    Lx, Ly = 20.0, 10.0
    r_ball = 0.2
    r_defender = 0.5
    dy = 0.04
    ind = {"x": 0, "y": 1}

    goal_line = np.array([[Lx, 0.1*Ly], [Lx, 0.9*Ly]])
    defender_init = np.array([0.9*Lx, 0.5*Ly])
    ball_init = np.array([0.1*Lx, 0.5*Ly])

    # Setup agent
    from collections import deque
    gamma = 0.95
    training_batch_size = 200
    memory = deque(maxlen=10000)

    action_size = 2
    observation_size = 2
    model = Sequential()
    model.add(Dense(10, input_dim=observation_size, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))

    # Generate episodes
    n_episodes = 20000
    n_iter = 300
    t = 0.0
    dt = 0.1
    scored = 0
    v_ball_size = 2.0
    reward_performance = deque(maxlen=100)
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
        reward_episode = 0.0
        for iter in range(n_iter):

            # ============================
            # Make observation
            # ============================
            observation = ball - defender

            # ============================
            # Choose an action
            # ============================
            epsilon = 0.2
            observation_reshaped = np.reshape(observation, [1, len(observation)])  # reshape for model input
            if np.random.rand() < epsilon:
                action = np.random.choice(list(range(action_size)))  # explore
            else:
                Q_estimate = model.predict(observation_reshaped)[0]
                action = np.random.choice(np.flatnonzero(Q_estimate == np.amax(Q_estimate)))  # exploit

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
            ball_next = ball + v_ball * dt
            d_ball_to_defender = np.linalg.norm(ball_next-defender)  # scalar

            # ============================
            # Check for termination
            # ============================
            if ball_next[ind["x"]] < 0 or ball_next[ind["x"]] > Lx or \
                ball_next[ind["y"]] < 0 or ball_next[ind["y"]] > Ly:
                intersection_result = intersectLines(ball_next, ball, goal_line[0], goal_line[1])
                valid, r, s = intersection_result[2], intersection_result[3], intersection_result[4]
                if valid==1 and r>=0.0 and r<=1.0 and s>=0.0 and s<=1.0:
                    scored += 1
                    status = "scored"
                else:
                    status = "missed"
            elif d_ball_to_defender < (r_defender + r_ball):
                status = "blocked"

            ball = ball_next
            observation_next = ball_next - defender

            # Get reward
            reward = 0
            done = False
            if status == "":
                done = 0
            elif status == "scored":
                reward = -1
                done = True
            elif status in ["blocked", "missed"]:
                reward = 1
                done = True
            else:
                exit("err: Invalid status found!")

            # Print
            #print("[{}] obs = {}, action = {}, reward = {}, done = {}".format(iter, observation, action, reward, done))

            # Store for training
            memory.append([observation, action, reward, observation_next, done])

            # Store for animation
            defender_animation.append(defender.copy())
            ball_animation.append(ball)

            # Add to total reward
            reward_episode += reward

            # Terminate?
            if done:
                break

        reward_performance.append(reward_episode)
        reward_avg = np.sum(np.array(reward_performance))/len(reward_performance)
        print("episode {}/{}: reward = {:.3f}, reward_avg = {:.3f}".format(episode+1, n_episodes, reward_episode, reward_avg))

        # =============================
        # Train
        # =============================
        n_training_sessions = 10
        for session in range(n_training_sessions):

            training_batch_size_use = min(training_batch_size, len(memory))
            minibatch = random.sample(memory, training_batch_size_use)  # sample mini-batch from replay memory
            (observations, actions, rewards, observations_next, dones) = ([], [], [], [], [])
            for (observation, action, reward, observation_next, done) in minibatch:
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                observations_next.append(observation_next)
                dones.append(done)

            observations = np.array(observations, dtype=np.float)  # (batch_size, observation_size)
            actions = np.array(actions, dtype=np.int)  # (batch_size, )
            rewards = np.array(rewards, dtype=np.float)  # (batch_size, )
            observations_next = np.array(observations_next, dtype=np.float)  # (batch_size, observation_size)
            dones = np.array(dones, dtype=np.int)  # (batch_size, )

            # Compute Q estimate
            Q = model.predict(observations)  # (batch_size, action_size)
            Q_next = model.predict(observations_next)  # (batch_size, action_size)

            # Compute target Q
            Q_target = Q.copy()  # (batch_size, action_size)
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                Q_target[i, action] = reward
                if not done:
                    Q_target[i, action] += gamma * np.amax(Q_next[i, :])

            # Train model
            n_epochs = 1
            model.fit(observations, Q_target, epochs=n_epochs, verbose=False)

        # =============================
        # Make animation
        # =============================
        if episode % 100 == 0:
            fig = plt.figure()
            fig.clf()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.1, Lx+0.5), ylim=(-0.1, Ly+0.1))

            defender_c = (229/255, 146/255, 4/255)
            ball_c = 'k'
            defender_particles = plt.Circle((defender_animation[0][0], defender_animation[0][1]), radius=r_defender, fc=defender_c)
            ball_particles = plt.Circle((ball_animation[0][0], ball_animation[0][1]), radius=r_ball, fc=ball_c)
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

            ani = animation.FuncAnimation(fig, animate, len(defender_animation), init_func=init, interval=20, blit=True)
            plt.axis('scaled')
            #plt.show()
            animation_filename = "animations/ani{}.mp4".format(episode)
            ani.save(animation_filename, extra_args=['-vcodec', 'libx264'])

        # =============================
        # Print result
        # =============================
        #print("{}: {} (acc={}/{})".format(episode+1, status, scored, episode+1))

