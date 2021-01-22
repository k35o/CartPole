import os, sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import gym

# 定数
MAX_STEPS = 200 # 最大のステップ数
NUM_EPISODES = 2000 # 最大の試行回数
NUM_DIZITIZED = 6 # 各状態の分割数

# 学習パラメータ
GAMMA = 0.99  # 時間割引率
ETA = 0.5  # 学習係数

# 離散化
def bins(clip_min, clip_max, num):
    # 観測した状態デジタル変換する閾値を求める
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def analog2digitize(observation):
    #状態の離散化
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIZITIZED)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIZITIZED)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIZITIZED)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIZITIZED))
    ]
    return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

# 行動選択
class Actor:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.q_table = np.random.uniform(low=-1, high=1, size=(NUM_DIZITIZED**num_states, num_actions))

    def decide_action(self, observation, episode):
        state = analog2digitize(observation)
        # ε-greedy法で行動を選択する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.rand():
            # 最も価値の高い行動を行う。
            action = np.argmax(self.q_table[state][:])
        else:
            # 適当に行動する。
            action = np.random.choice(self.num_actions)
        return action

# 状態評価
class Critic:
    def __init__(self, num_states):
        self.V = np.zeros(NUM_DIZITIZED**num_states)

# Actor Critic 学習
class ActorCritic():
    def __init__(self, env, actor_class, critic_class):
        self.env = env
        self.actor_class = actor_class
        self.critic_class = critic_class

    def run(self):
        # 状態数を取得
        num_states = self.env.observation_space.shape[0]
        # 行動数を取得
        num_actions = self.env.action_space.n
        actor = self.actor_class(num_states, num_actions)
        critic = self.critic_class(num_states)

        # Qテーブル, Vの更新
        def update_Qtable_V(observation, action, reward, observation_next):
            state = analog2digitize(observation)
            state_next = analog2digitize(observation_next)
            td = reward + GAMMA * critic.V[state_next] - critic.V[state]
            actor.q_table[state, action] += ETA * td
            critic.V[state] += ETA * td

        step_list = []
        mean_list = []
        std_list = []
        for episode in range(NUM_EPISODES):
            observation = self.env.reset()  # 環境の初期化
            frames = []
            for step in range(MAX_STEPS):
                if episode == NUM_EPISODES-1: frames.append(self.env.render(mode='rgb_array'))
                # 行動を求める
                action = actor.decide_action(observation, episode)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, _, done, _ = self.env.step(action)
                # 報酬を与える
                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    if step < 195:
                        reward = -1  # 失敗したので-1の報酬を与える
                    else:
                        reward = 1  # 成功したので+1の報酬を与える
                else:
                    reward = 0
                # Qテーブル, Vを更新する
                update_Qtable_V(observation, action, reward, observation_next)
                # 観測値を更新する。
                observation = observation_next


                # 終了時の処理
                if done:
                    step_list.append(step+1)
                    print('{}回目の試行は{}秒持ちました。(max:200秒)'.format(episode, step + 1))
                    break

            if episode == NUM_EPISODES-1:
                plt.figure()
                patch = plt.imshow(frames[0])
                plt.axis('off')

                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
                anim.save('movie_cartpole_v0_{}.gif'.format(episode+1), "ffmpeg")

        es = np.arange(0, len(step_list))
        plt.clf()
        plt.plot(es, step_list)
        plt.savefig("reward.png")
        for i, s in enumerate(step_list):
            if i < 100:
                mean_list.append(np.average(step_list[:i+1]))
                std_list.append(np.std(step_list[:i+1]))
            else:
                mean_list.append(np.average(step_list[i-100:i+1]))
                std_list.append(np.std(step_list[i-100:i+1]))
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        plt.clf()
        plt.plot(es, mean_list)
        plt.fill_between(es, mean_list-std_list, mean_list+std_list, alpha=0.2)
        plt.savefig("mean_var.png")

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    ac = ActorCritic(env, Actor, Critic)
    ac.run()
