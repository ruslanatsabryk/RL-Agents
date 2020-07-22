# replay Memory M with capacity N (= finite capacity)
# Initialize the DQN weights w
# for episode in max_episode:
#     s = Environment state
#     for steps in max_steps:
#         Choose action a from state s using epsilon greedy.
#         Take action a, get r (reward) and s' (next state)
#         Store experience tuple <s, a, r, s'> in M
#         s = s' (state = new_state)
#         Get random minibatch of exp tuples from M
#         Set Q_target = reward(s,a) +  γ * maxQ(s')
#         Update w =  α(Q_target - Q_value) *  ∇w Q_value

from collections import deque
import gym
from gym import spaces
import tensorflow as tf
import numpy as np
import random
import pickle
import copy

class DQNAgent:
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
                 done_factor=-1, reward_policy='asis'):
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
        self.memory = deque(maxlen=self.memory_limit)

        # High-score Model's copies
        self.hall_of_fame = deque(maxlen=10)
        self.hall_max_reward = -99_999_999_999.0

        # Model
        self.input_shape = self.observation_space.shape
        self.output_dim =  self.action_n
        print(f"self.input_shape={self.input_shape}, self.output_dim={self.output_dim}")
        m_input = tf.keras.Input(shape=self.input_shape)
        m = tf.keras.layers.Dense(256, activation='relu')(m_input)
        m = tf.keras.layers.Dense(256, activation='relu')(m)
        #m = tf.keras.layers.Dense(24, activation='relu')(m)
        m_output = tf.keras.layers.Dense(self.output_dim , activation='linear')(m)
        model = tf.keras.Model(m_input, m_output)
        #model.compile(optimizer='rmsprop', loss='mse')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        self.model = model

    def get_memory_sample(self):
        memory_batch = random.sample(self.memory, self.batch_size)
        return memory_batch

    def act(self, observation, test_env=False):
        if test_env:
            return self.action_space.sample()

        return self.action_space.sample()

    def fill_memory(self):
        action_discrete = True
        random_actions = None
        if action_discrete:
            rng = np.random.default_rng()
            random_actions = rng.integers(self.action_n, size=self.memory_init_size)

        #print("random_actions", random_actions)

        # Get data from environment
        row = 0
        while row < self.memory_init_size:
            observation = self.env.reset()
            done = False
            steps = 0
            total_reward = 0
            while not done:
                #print(f"Random action = {ra}")
                action = random_actions[row]
                observation_next, reward, done, info = self.env.step(action)
                steps += 1

                #reward = 0 if reward == -1 else reward
                if done: reward = reward * self.done_factor - 20
                total_reward += reward

                # Calculating final reward
                if self.reward_policy == 'cumulative':
                    final_reward = total_reward
                else:
                    final_reward = reward

                self.memory.append((observation, action, final_reward, observation_next, done))
                observation = observation_next
                row += 1
                if row == self.memory_init_size or done:
                    break
        print("init len(self.memory)", len(self.memory))

    def fit(self, e_play=0):
        # Fill the memory
        if self.memory_init_size > 0:
            self.fill_memory()
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
                if np.random.rand() < self.expl_rate:
                    action = random.randrange(self.action_n)
                else:
                    action = np.argmax(self.model.predict(observation)[0])

                # g_action = self.model.predict(observation)[0]
                # if np.random.rand() < self.expl_rate:
                #     action = np.argmax(1 - g_action)
                # else:
                #     action = np.argmax(g_action)


                # Make a step with environment
                observation_next, reward, done, info = self.env.step(action)
                steps += 1
                observation_next = np.reshape(observation_next, (1, -1))

                # Done and total rewards
                #reward = -reward
                #print('reward', reward)
                if done: reward = reward * self.done_factor - 20
                total_reward += reward

                # Calculating final reward
                if self.reward_policy=='cumulative':
                    final_reward = total_reward
                else:
                    final_reward = reward

                # Add experience to memory
                self.memory.append((observation, action, final_reward, observation_next, done))
                observation = observation_next


                # Experience replay
                # self.experience_replay()

                if done:

                    #self.memory.append((observation, action, final_reward, observation_next, done))

                    print(f"{self.expl_rate} Episode {episode} finished. Reward {total_reward}. Steps {steps}, {len(self.memory)}")
                    # Copy the most successful model to hall of fame
                    if self.hall_max_reward <= final_reward:
                        self.hall_of_fame.append((tf.keras.models.clone_model(self.model), self.model.get_weights()))
                        self.hall_max_reward = final_reward
                    break
                #observation = observation_next

                # Experience replay each step self.step_learn == True
                if self.learn_step:
                    self.experience_replay()

            # Experience replay each episode, self.step_learn == False
            if not self.learn_step:
                self.experience_replay()

        self.env.close()
        return self

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Unzip elements from memory
        memory_batch = self.get_memory_sample()
        observations = [raw[0] for raw in memory_batch]
        observations = np.vstack(observations)
        actions = [raw[1] for raw in memory_batch]
        actions = np.vstack(actions)
        rewards = [raw[2] for raw in memory_batch]
        rewards = np.vstack(rewards)
        observations_next = [raw[3] for raw in memory_batch]
        observations_next = np.vstack(observations_next)
        dones = [raw[4] for raw in memory_batch]
        dones = np.vstack(dones)
        #print("dones.shape", dones.shape)

        # Calculate Q-value for observations_next (q_value_next)
        predictions_next = self.model.predict_on_batch(x=observations_next)
        max_q_values = np.max(predictions_next, axis=1, keepdims=True)
        q_values_next = np.copy(rewards)
        q_values_next[dones == False] = rewards[dones == False] + self.gamma * max_q_values[dones == False]  # Поменять местами?

        # Calculate Q-value for observation (q_value)
        q_values = self.model.predict(x=observations)
        # for q_value, action, q_value_next in zip(q_values, actions, q_values_next):
        #     q_value[action[0]] = q_value_next[0]
        for i in range(self.batch_size):
            q_values[i, actions[i, 0]] = q_values_next[i, 0]

        # Train self.model (gradient descent)
        self.model.fit(x=observations, y=q_values, batch_size=self.learn_batch, epochs=self.learn_epochs, verbose=0) #=self.batch_size

        # Calculate Exploration Rate
        if len(self.memory) > self.memory_init_size:
            curr_expl_rate = self.expl_rate * self.expl_decay
            self.expl_rate = max(curr_expl_rate, self.expl_min)

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
    env = gym.make(env_name)

    # (self, environment, expl_decay=0.995, batch_size=64, mem_limit=1_000_000, mem_init_size=64,
    # max_episodes=100, gamma=0.95, learn_rate=0.001, learn_step=True, learn_batch=None, learn_epochs=5, done_factor=-1,
    # reward_policy='cumulative')

    cartpole_dqn = DQNAgent(environment=env, expl_decay=0.995, batch_size=640, mem_limit=100_000, mem_init_size=2000,
                            max_episodes=500, gamma=0.95, learning_rate=0.001, learn_step=True, learn_batch=1,
                            learn_epochs=1, done_factor=1, reward_policy='asis')

    cartpole_dqn.get_info()
    #cartpole_dqn.fill_memory()
    cartpole_dqn.fit(e_play=0)

    env_play = gym.make(env_name)
    model, weights = cartpole_dqn.hall_of_fame[-1]
    model.set_weights(weights)
    cartpole_dqn.play(env_play, model, n_episodes=200)