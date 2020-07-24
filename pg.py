from collections import deque
import gym
from gym import spaces
import tensorflow as tf
import numpy as np
import random
import pickle
import copy

class PGAgent:
    def get_info(self):
        print(f"Action space: {self.action_space}, {type(self.action_space)}")
        # print(f"Action space high: {self.action_space.high}")
        # print(f"Action space low: {self.action_space.low}")
        print(f"Observation space: {self.observation_space}, {type(self.observation_space)}")
        # print(f"Observation space high: {observation_space.high}")
        # print(f"Observation space low: {observation_space.low}")
        # envs = gym.envs.registry.all()
        # for envi in envs:
        #     print(envi)

    def __init__(self, environment, expl_decay=0.995, batch_size=64, mem_limit=1_000_000, mem_init_size=64,
                 max_episodes=100, gamma=0.95, learning_rate=0.001, learn_step=True, learn_batch=None, learn_epochs=5,
                 done_factor=-1, done_penalty=-20, reward_policy='asis'):
        # Constants
        self.batch_size = batch_size
        self.memory_limit = mem_limit
        self.memory_init_size = mem_init_size
        self.max_episodes = max_episodes
        self.learn_step = learn_step
        if learn_batch is None:
            self.learn_batch = self.batch_size
        else:
            self.learn_batch = learn_batch
        self.learn_epochs = learn_epochs
        self.done_factor = done_factor
        self.done_penalty = done_penalty
        self.reward_policy = reward_policy

        # Learning parameters
        self.alpha = learning_rate
        self.expl_max = 1.0
        self.expl_min = 0.01
        self.expl_rate = self.expl_max
        self.expl_decay = expl_decay
        self.greedy_decay = 0.0001
        self.gamma = gamma

        # Environment
        self.env = environment
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.action_n = environment.action_space.n

        # Memory
        self.observations = []
        self.actions = []
        self.prob_dist = []
        self.rewards = []

        # High-score Model's copies
        self.hall_of_fame = deque(maxlen=10)
        self.hall_max_reward = -99_999_999_999.0

        # Model
        self.input_shape = self.observation_space.shape
        self.output_dim =  self.action_n
        print(f"self.input_shape={self.input_shape}, self.output_dim={self.output_dim}")
        m_input = tf.keras.Input(shape=self.input_shape)
        m = tf.keras.layers.Dense(64, activation='selu')(m_input)
        m = tf.keras.layers.Dense(64, activation='tanh')(m)
        m = tf.keras.layers.Dense(24, activation='sigmoid')(m)
        m_output = tf.keras.layers.Dense(self.output_dim , activation='softmax')(m)
        model = tf.keras.Model(m_input, m_output)
        #model.compile(optimizer='rmsprop', loss='mse')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='binary_crossentropy')
        self.model = model

    def fit(self, e_play=0):
        for episode in range(self.max_episodes):
            observation = self.env.reset()
            observation = np.reshape(observation, (1, -1))
            steps = 0
            total_reward = 0
            done = False
            while not done:
                if e_play > 0 and episode % e_play == 0:
                    self.env.render()

                # Get Action for Descrete action space
                probability_distribution = self.model.predict(observation)[0]
                probability_distribution = np.exp(probability_distribution) / np.sum(np.exp(probability_distribution))
                #print("probability_distribution", probability_distribution)
                action = np.random.choice(self.action_n, p=probability_distribution)
                self.prob_dist.append(probability_distribution)

                # Make a step with environment
                observation_next, reward, done, info = self.env.step(action)
                steps += 1
                observation_next = np.reshape(observation_next, (1, -1))

                # Done and total rewards
                if done: reward = reward * self.done_factor + self.done_penalty
                total_reward += reward

                # Calculating final reward
                if self.reward_policy=='cumulative':
                    final_reward = total_reward
                else:
                    final_reward = reward

                # Add experience to memory
                one_hot_action = np.zeros(self.action_n)
                one_hot_action[action] = 1
                self.actions.append(one_hot_action - probability_distribution)
                self.observations.append(observation)
                self.rewards.append(final_reward)

                observation = observation_next

                if done:
                    # Calculate discounted reward
                    discounted_rewards = np.zeros_like(self.rewards)
                    accumulation = 0
                    for i in reversed(range(len(self.rewards))):
                        accumulation = accumulation * self.gamma + self.rewards[i]
                        discounted_rewards[i] = accumulation
                    mean = np.mean(discounted_rewards)
                    std = np.std(discounted_rewards)
                    discounted_rewards = (discounted_rewards - mean) / std

                    # Create train batch
                    x_train = np.vstack(self.observations)
                    actions = np.vstack(self.actions)
                    prob_dist = np.vstack(self.prob_dist)
                    y_train = prob_dist + actions * np.reshape(discounted_rewards, (-1, 1))

                    # Learn the estimator
                    self.model.fit(x_train, y_train, batch_size=self.learn_batch, epochs=self.learn_epochs, verbose=0)
                    #self.model.train_on_batch(x_train, y_train)
                    self.observations, self.actions, self.rewards, self.prob_dist = [], [], [], []

                    print(f"{self.expl_rate} Episode {episode} finished. Reward {total_reward}. Steps {steps}")
                    # Copy the most successful model to hall of fame
                    if self.hall_max_reward <= final_reward:
                        self.hall_of_fame.append((tf.keras.models.clone_model(self.model), self.model.get_weights()))
                        self.hall_max_reward = final_reward
                    break

        self.env.close()
        return self

    def play(self, play_env, play_model, n_episodes=1):
        for episode in range(n_episodes):
            observation = play_env.reset()
            observation = np.reshape(observation, (1, -1))
            steps = 0
            total_reward = 0
            done = False
            while not done:
                play_env.render()
                action = np.argmax(play_model.predict(observation)[0])

                # Make a step with environment
                observation_next, reward, done, info = play_env.step(action)
                observation_next = np.reshape(observation_next, (1, -1))
                observation = observation_next

                steps += 1
                total_reward += reward
                if done:
                    print(f"Episode {episode} finished. Reward {total_reward}. Steps {steps}")
                    break

if __name__ == "__main__":
    # env_name = "CartPole-v0"
    env_name = "CartPole-v1"
    # env_name = "Acrobot-v1"
    # env_name = "MountainCar-v0"
    # env_name = "MountainCarContinuous-v0"
    # env_name = "Pendulum-v0"
    # env_name = "LunarLander-v2"
    # env_name = "LunarLanderContinuous-v2"
    # env_name = "CarRacing-v0"
    # env_name = "BipedalWalker-v3"
    # env_name = "BipedalWalkerHardcore-v3"
    # env_name = "Breakout-ram-v0"
    # env_name = "Pong-ram-v0"
    env = gym.make(env_name)

    cartpole_dqn = PGAgent(environment=env, expl_decay=0.95, batch_size=500, mem_limit=1_000_000, mem_init_size=1000,
                            max_episodes=1000, gamma=0.9, learning_rate=0.01, learn_step=False, learn_batch=200,
                            learn_epochs=2, done_factor=-1, done_penalty=0, reward_policy='asis')

    cartpole_dqn.get_info()
    cartpole_dqn.fit(e_play=1)

    env_play = gym.make(env_name)
    model, weights = cartpole_dqn.hall_of_fame[-1]
    model.set_weights(weights)
    cartpole_dqn.play(env_play, model, n_episodes=200)