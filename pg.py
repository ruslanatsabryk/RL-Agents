from collections import deque
import gym
import keras
import keras.backend as K
import numpy as np

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

    def __init__(self, environment, max_episodes=100, gamma=0.95, learning_rate=0.001, learn_batch=None, learn_epochs=5,
                 done_factor=-1, done_penalty=-20, reward_policy='asis'):
        # Constants
        self.max_episodes = max_episodes
        self.learn_batch = learn_batch
        self.learn_epochs = learn_epochs
        self.done_factor = done_factor
        self.done_penalty = done_penalty
        self.reward_policy = reward_policy

        # Learning parameters
        self.alpha = learning_rate
        self.gamma = gamma

        # Environment
        self.env = environment
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.action_n = environment.action_space.n

        # Memory
        self.observations = []
        self.actions = []
        self.rewards = []
        self.total_rewards =[]

        # High-score Model's copies
        self.hall_of_fame = deque(maxlen=10)
        self.hall_max_reward = -99_999_999_999.0

        # Model's input and output shape
        self.input_shape = self.observation_space.shape
        self.output_dim =  self.action_n
        self.model, self.predictor = self.get_model()
        print(f"self.input_shape={self.input_shape}, self.output_dim={self.output_dim}")

    def get_model(self):
        g_input = keras.Input(shape=[1])

        def cepg_loss(y_true, y_pred):
            y_pred_clip = K.clip(y_pred, 1e-9, 1-1e-9)
            ce = y_true * K.log(y_pred_clip)
            loss = K.sum(-ce * g_input)
            return loss

        m_input = keras.Input(shape=self.input_shape)
        m = keras.layers.Dense(64, activation='relu')(m_input)
        m = keras.layers.Dense(64, activation='relu')(m)
        # m = tf.keras.layers.Dense(24, activation='sigmoid')(m)
        m_output = keras.layers.Dense(self.output_dim, activation='softmax')(m)

        estimator = keras.Model([m_input, g_input], m_output)
        estimator.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha), loss=cepg_loss)
        predictor = keras.Model(m_input, m_output)
        return estimator, predictor

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
                probability_distribution = self.predictor.predict(observation)[0]
                action = np.random.choice(self.action_n, p=probability_distribution)

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

                # Fill the memory
                self.actions.append(action)
                self.observations.append(observation)
                self.rewards.append(final_reward)

                observation = observation_next

                if done:
                    actions_len = len(self.actions)
                    one_hot_actions = np.zeros((actions_len, self.action_n))
                    one_hot_actions[range(actions_len), self.actions] = 1

                    self.total_rewards.append(total_reward)

                    # Calculate discounted reward
                    discounted_rewards = np.zeros_like(self.rewards)
                    accumulation = 0
                    for i in reversed(range(len(self.rewards))):
                        accumulation = accumulation * self.gamma + self.rewards[i]
                        discounted_rewards[i] = accumulation

                    mean = np.mean(discounted_rewards)
                    std_ = np.std(discounted_rewards)
                    std = std_ if std_ !=0 else 1
                    discounted_rewards = (discounted_rewards - mean) / std

                    # Create train batch
                    x_train = np.vstack(self.observations)
                    y_train = one_hot_actions

                    # Learn the estimator
                    if self.learn_epochs > 1 or self.learn_batch is not None:
                        batch_size = len(x_train) if self.learn_batch is None else self.learn_batch
                        self.model.fit([x_train, discounted_rewards], y_train, batch_size=batch_size,
                                       epochs=self.learn_epochs, verbose=0)
                    else:
                        self.model.train_on_batch([x_train, discounted_rewards], y_train)

                    self.observations, self.actions, self.rewards = [], [], []

                    print(f"Episode {episode} finished. "
                          f"Reward: {total_reward}. Average 100 reward: {np.mean(self.total_rewards[-100:])}, Steps: {steps}")

                    # Copy the most successful model to the hall of fame
                    if self.hall_max_reward <= final_reward:
                        self.hall_of_fame.append((keras.models.clone_model(self.predictor), self.predictor.get_weights()))
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
                #action = np.argmax(play_model.predict(observation)[0])
                probability_distribution = play_model.predict(observation)[0]
                action = np.random.choice(range(self.action_n), p=probability_distribution)

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
    # env_name = "CartPole-v1"
    # env_name = "Acrobot-v1"
    # env_name = "MountainCar-v0"
    # env_name = "MountainCarContinuous-v0"
    # env_name = "Pendulum-v0"
    env_name = "LunarLander-v2"
    # env_name = "LunarLanderContinuous-v2"
    # env_name = "CarRacing-v0"
    # env_name = "BipedalWalker-v3"
    # env_name = "BipedalWalkerHardcore-v3"
    # env_name = "Breakout-ram-v0"
    # env_name = "Pong-ram-v0"
    env = gym.make(env_name)

    cartpole_pg = PGAgent(environment=env, max_episodes=3000, gamma=0.99, learning_rate=0.0005, learn_batch=None,
                            learn_epochs=1, done_factor=1, done_penalty=0, reward_policy='asis')

    cartpole_pg.get_info()
    cartpole_pg.fit(e_play=0)

    env_play = gym.make(env_name)
    model, weights = cartpole_pg.hall_of_fame[-1]
    model.set_weights(weights)
    cartpole_pg.play(env_play, model, n_episodes=200)