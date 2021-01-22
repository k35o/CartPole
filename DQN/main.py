import os, sys

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.losses import huber_loss

# 定数
MAX_STEPS = 200 # 最大のステップ数
NUM_EPISODES = 200 # 最大の試行回数
NUM_DIZITIZED = 6 # 各状態の分割数

# 学習パラメータ
GAMMA = 0.99  # 時間割引率
ETA = 0.5  # 学習係数
CAPACITY = 10**4
BATCH_SIZE = 32

class ExperienceMemory:
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = deque(maxlen=self.capacity)

    def push(self, state, action, state_next, reward):
        self.memory.append((state, action, state_next, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self, num_states, num_actions):
        self.memory = ExperienceMemory()
        self.num_states = num_states
        self.num_actions  = num_actions

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(num_states, )))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(self.num_actions, activation="linear"))
        self.optimizer = Adam(lr=0.00001)
        self.model.compile(loss=huber_loss, optimizer=self.optimizer)

    def replay(self, target_q_n):
        if len(self.memory) < BATCH_SIZE:
            return
        inputs = np.zeros((BATCH_SIZE, self.num_states))
        targets = np.zeros((BATCH_SIZE, self.num_actions))
        transitions = self.memory.sample(BATCH_SIZE)

        for i, (state_batch, action_batch, state_next_batch, reward_batch) in enumerate(transitions):
            inputs[i:i+1] = state_batch
            target = reward_batch

            if not (state_next_batch == np.zeros(state_batch.shape)).all(axis=1):
                mainQ = self.model.predict(state_batch)[0]
                action_next = np.argmax(mainQ)
                target = reward_batch + GAMMA * target_q_n.model.predict(state_next_batch)[0][action_next]

            targets[i] = self.model.predict(state_batch)
            targets[i][action_batch] = target
            self.model.fit(inputs, targets, epochs=1, verbose=0)

    def decide_action(self, state, episode, target_q_n):
        # ε-greedy法で行動を選択する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.rand():
            # 最も価値の高い行動を行う。
            action = np.argmax(target_q_n.model.predict(state)[0])
        else:
            # 適当に行動する。
            action = np.random.choice(self.num_actions)

        return action

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = DQN(num_states, num_actions)
        self.target_q_n = DQN(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay(self.target_q_n)

    def get_action(self, state, episode):
        return self.brain.decide_action(state, episode, self.target_q_n)

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)


class Env():
    def __init__(self, env):
        self.env = env

    def run(self):
        # 状態数を取得
        num_states = self.env.observation_space.shape[0]
        # 行動数を取得
        num_actions = self.env.action_space.n
        agent = Agent(num_states, num_actions)

        complete_episodes = 0
        step_list = []
        mean_list = []
        std_list = []
        is_episode_final = False  # 最後の試行
        for episode in range(NUM_EPISODES):
            observation = self.env.reset()  # 環境の初期化
            state = observation
            state = np.reshape(state, [1, num_states])

            agent.target_q_n = agent.brain
            frames = []
            for step in range(MAX_STEPS):
                if is_episode_final: frames.append(self.env.render(mode='rgb_array'))
                # 行動を求める
                action = agent.get_action(state, episode)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                state_next, _, done, _ = self.env.step(action)
                state_next = np.reshape(state_next, [1, num_states])
                # 報酬を与える
                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    state_next = np.zeros(state.shape)
                    if step < 195:
                        reward = -1  # 失敗したので-1の報酬を与える
                        complete_episodes = 0  # 成功数をリセット
                    else:
                        reward = 1  # 成功したので+1の報酬を与える
                        complete_episodes += 1  # 連続成功記録を更新
                else:
                    reward = 0
                agent.memorize(state, action, state_next, reward)
                agent.update_q_function()
                # 観測値を更新する
                state = state_next

                # 終了時の処理
                if done:
                    step_list.append(step+1)
                    print('{}回目の試行は{}秒持ちました。(max:200秒)'.format(episode, step + 1))
                    break

            if is_episode_final:
                plt.figure()
                patch = plt.imshow(frames[0])
                plt.axis('off')

                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
                anim.save('movie_cartpole_v0_{}.gif'.format(episode+1), "ffmpeg")
                break

            if complete_episodes >= 10:
                print("最後の思考を行います。")
                is_episode_final = True

        es = np.arange(0, len(step_list))
        plt.clf()
        plt.plot(es, step_list)
        plt.savefig("reward.png")
        for i, s in enumerate(step_list):
            if i < 10:
                mean_list.append(np.average(step_list[:i+1]))
                std_list.append(np.std(step_list[:i+1]))
            else:
                mean_list.append(np.average(step_list[i-10:i+1]))
                std_list.append(np.std(step_list[i-10:i+1]))
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        plt.clf()
        plt.plot(es, mean_list)
        plt.fill_between(es, mean_list-std_list, mean_list+std_list, alpha=0.2)
        plt.savefig("mean_var.png")

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Env(env)
    env.run()
