import numpy as np
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random

from line_intersections import intersectLines

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque

if __name__ == "__main__":

    # make agent
    class Agent:

        def __init__(self):
            self.action_size = 2
            self.observation_size = 2
            self.gamma = 0.99
            self.epsilon = 0.1
            self._build_model()
            self._reserve_memory()

        def _reserve_memory(self):
            #self.memory = deque(maxlen=10000)
            self.memory = []

        def _build_model(self):
            model = Sequential()
            model.add(Dense(50, input_dim=self.observation_size, activation="relu"))
            model.add(Dense(50, activation="relu"))
            model.add(Dense(self.action_size, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=0.001))
            self.model = model

        def train(self, n_batch_sessions=10, training_batch_size=200):

            for session in range(n_batch_sessions):

                minibatch = random.sample(self.memory, min(training_batch_size, len(self.memory)))
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
                Q = self.model.predict(observations)  # (batch_size, action_size)
                Q_next = self.model.predict(observations_next)  # (batch_size, action_size)

                # Compute target Q
                Q_target = Q.copy()  # (batch_size, action_size)
                for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                    Q_target[i, action] = reward
                    if not done:
                        Q_target[i, action] += self.gamma * np.amax(Q_next[i, :])

                # Train model
                self.model.fit(observations, Q_target, epochs=1, verbose=False)

        def get_action(self, observation):

            observation_reshaped = np.reshape(observation, [1, len(observation)])  # reshape for model input
            if np.random.rand() < self.epsilon:
                action = np.random.choice(list(range(self.action_size)))  # explore
            else:
                Q_estimate = self.model.predict(observation_reshaped)[0]
                action = np.random.choice(np.flatnonzero(Q_estimate == np.amax(Q_estimate)))  # exploit

            return action

        def save(self, model_filename):
            self.model.save(model_filename)

    class Environment:

        def __init__(self):
            pass

        def reset(self):
            self.t = 0.0
            self.dt = 0.1
            self.Lx = 20.0
            self.Ly = 10.0
            self.v_ball_mag = 2.0
            self.r_ball = 0.2
            self.r_defender = 0.2
            self.dx, self.dy = 0.04, 0.04
            self.ind = {"x": 0, "y": 1}

            self.goal_line = np.array([[self.Lx, 0.1 * self.Ly], [self.Lx, 0.9 * self.Ly]])
            self.defender = np.array([0.9 * self.Lx, 0.5 * self.Ly])
            self.ball = np.array([0.1 * self.Lx, 0.5 * self.Ly])

            # Set up ball velocity
            goal_line_rel_ball = self.goal_line - self.ball
            theta_goal_line_rel_ball = []
            for i in range(2):
                theta = np.arctan2(goal_line_rel_ball[i, self.ind["y"]], goal_line_rel_ball[i, self.ind["x"]])
                theta_goal_line_rel_ball.append(theta)
            theta_sample = np.random.uniform(theta_goal_line_rel_ball[0], theta_goal_line_rel_ball[1])
            theta_vec_sample = np.array([np.cos(theta_sample), np.sin(theta_sample)])
            self.v_ball = self.v_ball_mag * theta_vec_sample

        def make_observation(self):
            ball_relative_location = self.ball - self.defender
            observation = ball_relative_location
            return observation

        def step(self, action):

            # Update time
            self.t += self.dt

            # Move agent
            if action == 0: self.defender += np.array([0, self.dy])  # up
            elif action == 1: self.defender += np.array([0, -self.dy])  # down
            else: exit("err: invalid action!")

            # Move environment
            ball = self.ball
            ball_next = self.ball + self.v_ball * self.dt
            self.ball = ball_next

            # Check for termination
            status = ""
            d_ball_to_defender = np.linalg.norm(self.ball - self.defender)  # scalar
            if ball_next[self.ind["x"]] < 0 or ball_next[self.ind["x"]] > self.Lx or \
                            ball_next[self.ind["y"]] < 0 or ball_next[self.ind["y"]] > self.Ly:
                intersection_result = intersectLines(ball_next, ball, self.goal_line[0], self.goal_line[1])
                valid, r, s = intersection_result[2], intersection_result[3], intersection_result[4]
                if valid == 1 and r >= 0.0 and r <= 1.0 and s >= 0.0 and s <= 1.0: status = "scored"
                else: status = "missed"
            elif d_ball_to_defender < (self.r_defender + self.r_ball):
                status = "blocked"

            # Get reward
            reward, done = 0, False
            if status == "scored":
                reward, done = -1, True
            elif status in ["blocked", "missed"]:
                reward, done = 1, True
            elif status == "":
                pass
            else:
                exit("err: Invalid status found!")

            # Make new observation
            observation_next = self.make_observation()

            return observation_next, reward, done

        def make_animation(self, animation_data, animation_filename):

            defender_animation, ball_animation  = animation_data

            print("Making animation...")
            fig = plt.figure()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.1, self.Lx + 0.5), ylim=(-0.1, self.Ly + 0.1))

            defender_c = (229 / 255, 146 / 255, 4 / 255)
            ball_c = 'k'
            defender_particles = plt.Circle((defender_animation[0][0], defender_animation[0][1]),
                                            radius=self.r_defender, fc=defender_c)
            ball_particles = plt.Circle((ball_animation[0][0], ball_animation[0][1]),
                                        radius=self.r_ball, fc=ball_c)
            rect = plt.Rectangle((0.0, 0.0), self.Lx, self.Ly, ec='none', lw=2, fc='none')
            goal = plt.Rectangle((self.goal_line[0][0], self.goal_line[0][1]), 0.3, self.goal_line[1][1] - self.goal_line[0][1],
                                 ec='none', lw=2, fc='k')
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

            ani = animation.FuncAnimation(fig, animate, len(defender_animation), init_func=init, interval=20,
                                          blit=True)
            plt.axis('scaled')
            ani.save(animation_filename, extra_args=['-vcodec', 'libx264'])
            plt.clf()


    # =====================================
    #
    # Start
    #
    # =====================================
    env = Environment()
    agent = Agent()
    n_experiments = 100
    n_episodes_per_experiment = 1000
    n_iter = 300

    n_batch_sessions = 20
    training_batch_size = 200

    for experiment in range(n_experiments):

        #agent.save("models/model_exp{}.h5".format(experiment))
        reward_experiment = []
        for episode in range(n_episodes_per_experiment):
            env.reset()
            agent.memory = []

            ball_animation, defender_animation = [], []
            ball_animation.append(env.ball.copy())  # store to animation data
            defender_animation.append(env.defender.copy())  # store to animation data

            reward_episode = 0.0
            for iter in range(n_iter):
                observation = env.make_observation()
                action = agent.get_action(observation)
                observation_next, reward, done = env.step(action)
                agent.memory.append([observation, action, reward, observation_next, done])

                defender_animation.append(env.defender.copy())  # store
                ball_animation.append(env.ball.copy())  # store
                reward_episode += reward  # sum up episodic reward
                if done: break  # terminate

            reward_experiment.append(reward_episode)

            #if episode == 0:
            #    animation_data = (defender_animation, ball_animation)
            #    env.make_animation(animation_data, "animations/ani_exp{}_ep{}.mp4".format(experiment, episode))

        reward_avg = sum(reward_experiment) / len(reward_experiment)
        print("experiment {}/{}: reward_avg = {:.3f}".format(experiment, len(reward_experiment), reward_avg))
        agent.train(n_batch_sessions = n_batch_sessions, training_batch_size = training_batch_size)
